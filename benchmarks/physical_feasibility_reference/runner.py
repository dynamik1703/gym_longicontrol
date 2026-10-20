"""Fail-closed lifecycle for the physical-reference study."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .causal_controller import CAUSAL_CRUISE_CAP_M_S
from .diagnostics import (
    HERE,
    build_split_ledger,
    load_json,
    preflight_benchmark,
    source_hashes,
    verify_runtime_row,
)
from .evaluation import (
    canonical_sha256,
    paired_differences,
    rollout,
    solve_oracle,
    summarize,
)
from .feasibility import CANONICAL_TASK, physical_audit
from .oracle_controller import (
    BRAKING_POSITION_MARGIN_M,
    VELOCITY_SAFETY_MARGIN_M_S,
)

SPLIT_LEDGER = HERE / "split_ledger.json"
FREEZE = HERE / "freeze.json"
RESOURCE = HERE / "resource_measurements.json"
DEVELOPMENT = HERE / "development_results.json"
HOLDOUT = HERE / "holdout_results.json"
RESULTS = HERE / "results.json"
WITNESSES = HERE / "witnesses"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare() -> dict[str, Any]:
    ledger = build_split_ledger()
    if SPLIT_LEDGER.exists():
        if load_json(SPLIT_LEDGER) != ledger:
            raise RuntimeError("frozen split ledger differs from regeneration")
    else:
        _write_json(SPLIT_LEDGER, ledger)
    return ledger


def freeze() -> dict[str, Any]:
    if not SPLIT_LEDGER.exists():
        raise RuntimeError("prepare split ledger before controller freeze")
    ledger = load_json(SPLIT_LEDGER)
    if not ledger["route_clean_verified"]:
        raise RuntimeError("route-clean split checks failed")
    payload = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_task": {
            "route_length_m": 1000.0,
            "deadline_s": CANONICAL_TASK.max_time_s,
            "maximum_speed_violation_m_s": (
                CANONICAL_TASK.max_speed_violation_m_s
            ),
            "native_dt_s": 0.1,
            "maximum_episode_steps": 1800,
            "action_bounds": [-1.0, 1.0],
        },
        "controller_configuration": {
            "velocity_safety_margin_m_s": VELOCITY_SAFETY_MARGIN_M_S,
            "braking_position_margin_m": BRAKING_POSITION_MARGIN_M,
            "causal_cruise_cap_m_s": CAUSAL_CRUISE_CAP_M_S,
            "oracle_energy_refinement_iterations": 10,
        },
        "split_sha256": ledger["evaluation_sha256"],
        "source_sha256": source_hashes(),
    }
    payload["scientific_configuration_sha256"] = canonical_sha256(
        {
            "canonical_task": payload["canonical_task"],
            "controller_configuration": payload["controller_configuration"],
            "split_sha256": payload["split_sha256"],
            "source_sha256": payload["source_sha256"],
        }
    )
    if FREEZE.exists():
        existing = load_json(FREEZE)
        comparable = {
            key: value
            for key, value in existing.items()
            if key != "frozen_at_utc"
        }
        requested = {
            key: value
            for key, value in payload.items()
            if key != "frozen_at_utc"
        }
        if existing.get("source_sha256") and comparable != requested:
            raise RuntimeError(
                "controller freeze already exists with different content"
            )
        return existing
    _write_json(FREEZE, payload)
    return payload


def _assert_frozen() -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("controllers are not frozen")
    frozen = load_json(FREEZE)
    if frozen["source_sha256"] != source_hashes():
        raise RuntimeError("frozen reference source changed")
    ledger = prepare()
    if frozen["split_sha256"] != ledger["evaluation_sha256"]:
        raise RuntimeError("frozen split hashes changed")
    return frozen


def resource_preflight() -> dict[str, Any]:
    ledger = prepare()
    seeds = tuple(
        int(row["numeric_seed"])
        for row in ledger["evaluation_sets"]["development"]["accepted_tasks"][:3]
    )
    payload = preflight_benchmark(seeds)
    payload.update(
        {
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "execution_limit_hours": 12.0,
            "retained_artifact_limit_gb": 12.0,
        }
    )
    _write_json(RESOURCE, payload)
    return payload


def _write_witness(
    split_id: str, row: dict[str, Any], actions: list[float]
) -> dict[str, Any]:
    path = WITNESSES / split_id.lower() / f"seed-{row['track_seed']}.json"
    payload = {
        "schema_version": 1,
        "split": split_id,
        "track_seed": row["track_seed"],
        "route_replay": "StochasticTrack-v1 reset with numeric seed",
        "controller": "FullRoutePhysicsOracle",
        "action_sha256": row["action_sha256"],
        "trace_sha256": row["trace_sha256"],
        "source_sha256": load_json(FREEZE)["source_sha256"],
        "metrics": {
            "completed": row["completion"],
            "travel_time_s": row["travel_time_s"],
            "energy_kwh": row["signed_net_energy_kwh"],
            "speed_violation_count": row["speed_violation_count"],
            "max_speed_violation_m_s": row["max_speed_violation_m_s"],
            "integrated_speed_violation_m": row[
                "integrated_speed_violation_m"
            ],
            "final_position_m": row["final_position_m"],
        },
        "actions": actions,
    }
    _write_json(path, payload)
    return {
        "path": str(path.relative_to(HERE)),
        "sha256": canonical_sha256(payload),
        "action_count": len(actions),
    }


def evaluate_panel(name: str) -> dict[str, Any]:
    frozen = _assert_frozen()
    resource_report = load_json(RESOURCE) if RESOURCE.exists() else resource_preflight()
    if not (
        resource_report["execution_gate_passed"]
        and resource_report["artifact_gate_passed"]
    ):
        raise RuntimeError("mandatory runtime/resource gate failed")
    ledger = load_json(SPLIT_LEDGER)
    split = ledger["evaluation_sets"][name]
    split_id = split["id"]
    started = time.perf_counter()
    oracle_rows = []
    causal_rows = []
    witnesses = []
    for split_row in split["accepted_tasks"]:
        verify_runtime_row(split_row)
        seed = int(split_row["numeric_seed"])
        oracle = solve_oracle(seed, refinement_iterations=10)
        actions = oracle.pop("actions")
        if oracle["feasible_witness_found"]:
            witnesses.append(_write_witness(split_id, oracle, actions))
        oracle_rows.append(oracle)
        causal = rollout(seed, "causal")
        causal.pop("actions")
        causal_rows.append(causal)
    result = {
        "schema_version": 1,
        "evaluation_id": split_id,
        "split_ordered_sha256": split["ordered_sha256"],
        "scientific_configuration_sha256": frozen[
            "scientific_configuration_sha256"
        ],
        "source_sha256": frozen["source_sha256"],
        "oracle": {
            "episodes": oracle_rows,
            "summary": summarize(oracle_rows),
            "classification_counts": {
                classification: sum(
                    row["classification"] == classification for row in oracle_rows
                )
                for classification in (
                    "FEASIBLE_WITNESS",
                    "PROVEN_INFEASIBLE",
                    "UNRESOLVED",
                )
            },
            "witnesses": witnesses,
            "all_numerical_parity": all(
                row["numerical_parity"] for row in oracle_rows
            ),
        },
        "causal": {
            "episodes": causal_rows,
            "summary": summarize(causal_rows),
        },
        "paired_task_level_differences": paired_differences(
            oracle_rows, causal_rows
        ),
        "wall_time_s": time.perf_counter() - started,
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "content_fingerprints_constructed": 0,
        },
    }
    return result


def run_development() -> dict[str, Any]:
    result = evaluate_panel("development")
    _write_json(DEVELOPMENT, result)
    return result


def run_holdout() -> dict[str, Any]:
    if HOLDOUT.exists():
        raise RuntimeError("PHYSICS_REFERENCE_HOLDOUT_V1_ONE_OPEN_ONLY")
    result = evaluate_panel("holdout")
    _write_json(HOLDOUT, result)
    return result


def finalize() -> dict[str, Any]:
    if not DEVELOPMENT.exists() or not HOLDOUT.exists():
        raise RuntimeError("both development and holdout records are required")
    development = load_json(DEVELOPMENT)
    holdout = load_json(HOLDOUT)
    resources = load_json(RESOURCE)
    audit = physical_audit()
    payload = {
        "schema_version": 1,
        "study": (
            "Physical Feasibility, Optimal-Control Reference and "
            "Observation-Limited Control"
        ),
        "scope": "control/feasibility reference; not an RL algorithm comparison",
        "physical_audit": audit,
        "split_sha256": load_json(SPLIT_LEDGER)["evaluation_sha256"],
        "freeze": load_json(FREEZE),
        "resource_preflight": resources,
        "development": {
            "evaluation_id": development["evaluation_id"],
            "oracle": development["oracle"]["summary"],
            "oracle_classifications": development["oracle"][
                "classification_counts"
            ],
            "causal": development["causal"]["summary"],
            "wall_time_s": development["wall_time_s"],
        },
        "holdout": {
            "evaluation_id": holdout["evaluation_id"],
            "oracle": holdout["oracle"]["summary"],
            "oracle_classifications": holdout["oracle"][
                "classification_counts"
            ],
            "causal": holdout["causal"]["summary"],
            "all_numerical_parity": holdout["oracle"]["all_numerical_parity"],
            "paired_task_level_differences": holdout[
                "paired_task_level_differences"
            ],
            "wall_time_s": holdout["wall_time_s"],
        },
        "contextual_analysis": {
            "performed": False,
            "reason": (
                "Optional post-hoc panels were omitted to keep the independent "
                "holdout as the sole newly opened final-performance result."
            ),
        },
        "interpretation_guardrails": {
            "oracle_rate_is_achieved_not_certified_upper_bound": True,
            "physical_feasibility_not_observation_limited_controllability": True,
            "observation_limited_controllability_not_learning_feasibility": True,
        },
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "content_fingerprints_constructed": 0,
        },
    }
    _write_json(RESULTS, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "freeze",
            "resource-preflight",
            "development",
            "holdout",
            "finalize",
        ),
    )
    args = parser.parse_args(argv)
    functions = {
        "prepare": prepare,
        "freeze": freeze,
        "resource-preflight": resource_preflight,
        "development": run_development,
        "holdout": run_holdout,
        "finalize": finalize,
    }
    payload = functions[args.stage]()
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
