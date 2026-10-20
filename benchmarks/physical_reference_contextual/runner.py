"""Fail-closed contextual evaluation using frozen physical controllers.

The module never imports an RL policy implementation.  It reads complete
episode metrics and split fingerprints straight from immutable Git objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any

from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed
from benchmarks.physical_feasibility_reference.diagnostics import (
    load_json,
    source_hashes,
)
from benchmarks.physical_feasibility_reference.evaluation import (
    paired_differences,
    rollout,
    solve_oracle,
    summarize,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EPISODES = HERE / "episodes"
PROTOCOL = HERE / "PROTOCOL.md"
FREEZE = HERE / "freeze.json"
INTEGRITY = HERE / "integrity.json"
RESULTS = HERE / "results.json"
PAIRED = HERE / "paired_analysis.json"
RESULTS_MD = HERE / "RESULTS.md"

PHYSICAL_COMMIT = "3d9e8bcb72ecf4a01be192036df8f00c5ac9de8e"
PHYSICAL_FREEZE = ROOT / "benchmarks/physical_feasibility_reference/freeze.json"
REFERENCE_SOURCE = ROOT / "benchmarks/physical_feasibility_reference"
EXPECTED_REFERENCE_CONFIGURATION_SHA256 = (
    "f63717504a810c75d7080388aeb159505cf667ad13248dbe4a53ec0c4bb94cd1"
)

PANELS: dict[str, dict[str, str]] = {
    "sameinfo": {
        "evaluation_id": "SAMEINFO_VALIDATION_V1",
        "source_commit": "78a50b7ad866beb72a93f06cb1c5d6bcaa25cc12",
        "results_path": (
            "benchmarks/same_information_scalar_vs_constrained/results.json"
        ),
        "split_path": (
            "benchmarks/same_information_scalar_vs_constrained/SPLIT_LEDGER.md"
        ),
        "ordered_sha256": (
            "d8a658e96b0090f8e0cdcdfdec3b291329045dc3dac9541e0603f34d916bf10a"
        ),
        "secondary_method": "scalar",
    },
    "gcsl": {
        "evaluation_id": "GCSL_VALIDATION_V1",
        "source_commit": "5ada9402e3a2d627a7ea417cf1ffc23e1ab3789e",
        "results_path": "benchmarks/gcsl_canonical/results.json",
        "split_path": "benchmarks/gcsl_canonical/SPLIT_LEDGER.md",
        "ordered_sha256": (
            "d808fcc361be35f0c585003278f200d4013cc03718b0ceb318fbe6d55195a159"
        ),
        "secondary_method": "gcsl",
    },
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str, text: bool = False) -> bytes | str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def _git_bytes(commit: str, path: str) -> bytes:
    value = _git("show", f"{commit}:{path}")
    assert isinstance(value, bytes)
    return value


def _git_text(commit: str, path: str) -> str:
    return _git_bytes(commit, path).decode("utf-8")


def _git_blob(commit: str, path: str) -> str:
    value = _git("rev-parse", f"{commit}:{path}", text=True)
    assert isinstance(value, str)
    return value.strip()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _section(markdown: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$([\s\S]*?)(?=^## |\Z)",
        markdown,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"missing immutable split section {heading}")
    return match.group(1)


def _split_rows(markdown: str, heading: str) -> list[dict[str, Any]]:
    rows = []
    pattern = re.compile(
        r"^\|\s*(\d+)\s*\|\s*`([0-9a-f]{64})`\s*\|"
        r"\s*`([0-9a-f]{64})`\s*\|\s*$",
        flags=re.MULTILINE,
    )
    for seed, route, full in pattern.findall(_section(markdown, heading)):
        rows.append(
            {
                "numeric_seed": int(seed),
                "route_sha256": route,
                "full_task_sha256": full,
            }
        )
    if len(rows) != 18:
        raise RuntimeError(f"{heading} immutable ledger has {len(rows)} task rows")
    return rows


def _immutable_panel(name: str) -> dict[str, Any]:
    spec = PANELS[name]
    result_bytes = _git_bytes(spec["source_commit"], spec["results_path"])
    split_bytes = _git_bytes(spec["source_commit"], spec["split_path"])
    payload = json.loads(result_bytes)
    if payload["evaluation_id"] != spec["evaluation_id"]:
        raise RuntimeError(f"{name} evaluation identity changed")
    if name == "sameinfo":
        rows = payload["complete_compact_episode_metrics"]
        seeds = payload["validation_split"]["ordered_numeric_seeds"]
        ordered = payload["validation_split"]["ordered_sha256"]
    else:
        rows = payload["episodes"]
        seeds = payload["validation_split"]["seeds"]
        ordered = payload["validation_split"]["ordered_sha256"]
    if len(rows) != 108:
        raise RuntimeError(f"{name} immutable result matrix is incomplete")
    if ordered != spec["ordered_sha256"]:
        raise RuntimeError(f"{name} ordered split hash changed")
    tasks = _split_rows(split_bytes.decode("utf-8"), spec["evaluation_id"])
    if [row["numeric_seed"] for row in tasks] != seeds:
        raise RuntimeError(f"{name} result seeds differ from immutable ledger")
    return {
        "name": name,
        "evaluation_id": spec["evaluation_id"],
        "source_commit": spec["source_commit"],
        "ordered_sha256": ordered,
        "tasks": tasks,
        "rl_rows": rows,
        "results_artifact": {
            "path": spec["results_path"],
            "git_blob": _git_blob(spec["source_commit"], spec["results_path"]),
            "sha256": hashlib.sha256(result_bytes).hexdigest(),
            "episode_projection_sha256": canonical_sha256(rows),
        },
        "split_artifact": {
            "path": spec["split_path"],
            "git_blob": _git_blob(spec["source_commit"], spec["split_path"]),
            "sha256": hashlib.sha256(split_bytes).hexdigest(),
        },
    }


def _verify_runtime_task(row: dict[str, Any]) -> None:
    seed = int(row["numeric_seed"])
    if seed in PAPER_SEEDS:
        raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
    value = fingerprint_track(
        track_from_seed(seed), v2_semantics(), numeric_seed=seed
    )
    if value.route_sha256 != row["route_sha256"]:
        raise RuntimeError(f"runtime route fingerprint changed for {seed}")
    if value.full_task_sha256 != row["full_task_sha256"]:
        raise RuntimeError(f"runtime full-task fingerprint changed for {seed}")


def _reference_integrity() -> dict[str, Any]:
    frozen = load_json(PHYSICAL_FREEZE)
    observed = source_hashes()
    if frozen["source_sha256"] != observed:
        raise RuntimeError("frozen physical-controller source hash mismatch")
    if (
        frozen["scientific_configuration_sha256"]
        != EXPECTED_REFERENCE_CONFIGURATION_SHA256
    ):
        raise RuntimeError("frozen physical configuration hash mismatch")
    commit = _git("rev-parse", "HEAD", text=True)
    assert isinstance(commit, str)
    if not bool(
        _git(
            "merge-base", "--is-ancestor", PHYSICAL_COMMIT, commit.strip(), text=True
        )
        == ""
    ):
        raise RuntimeError("worktree does not descend from physical reference commit")
    return {
        "base_commit": PHYSICAL_COMMIT,
        "worktree_head": commit.strip(),
        "scientific_configuration_sha256": frozen[
            "scientific_configuration_sha256"
        ],
        "controller_configuration": frozen["controller_configuration"],
        "source_sha256_expected": frozen["source_sha256"],
        "source_sha256_observed": observed,
        "verified": True,
    }


def _prepared_payload() -> dict[str, Any]:
    reference = _reference_integrity()
    panels = {}
    for name in PANELS:
        panel = _immutable_panel(name)
        for task in panel["tasks"]:
            _verify_runtime_task(task)
        panels[name] = {key: value for key, value in panel.items() if key != "rl_rows"}
    return {
        "schema_version": 1,
        "study": "post-hoc contextual physical-reference analysis",
        "independent_heldout_benchmark": False,
        "frozen_before_contextual_execution": True,
        "protocol_sha256": file_sha256(PROTOCOL),
        "runner_sha256": file_sha256(Path(__file__)),
        "physical_reference": reference,
        "panels": panels,
        "canonical_evaluation": {
            "environment": "StochasticTrack-v1",
            "route_length_m": 1000.0,
            "deadline_s": 140.0,
            "speed_tolerance_m_s": 0.0,
            "native_dt_s": 0.1,
            "maximum_episode_steps": 1800,
            "metrics": "production EpisodeMetrics and is_feasible",
        },
        "execution_plan": {
            "panel_count": 2,
            "tasks_per_panel": 18,
            "reference_controllers": ["oracle", "causal"],
            "scientific_controller_task_pairs": 72,
            "pair_execution_limit": 1,
            "oracle_replay_checks_planned": 36,
            "oracle_refinement_iterations": 10,
            "normal_mode_refuses_existing_episode_records": True,
            "aggregation_only_reexecutes_controllers": False,
        },
        "planned_comparisons": [
            "per-panel Oracle and causal RSR/completion/deadline/speed",
            "paired feasible-only causal-minus-Oracle energy and travel time",
            "RL-failed policy-track pairs with Oracle/causal witnesses",
            "distinct RL-failed tracks with Oracle/causal witnesses",
            "RL failure modes by policy, seed, method, and panel",
            "RL-minus-reference energy only when both trajectories are feasible",
        ],
        "interpretation_limits": [
            "previously opened contextual tasks; not independent held-out evidence",
            "achieved Oracle reference; not a proven global optimum",
            "no guarantee over all possible generated tracks",
            "no controller or Hybrid design tuning from contextual outcomes",
            "policy-track pairs and distinct tracks remain separate units",
        ],
        "rl_execution": {
            "actors_loaded": False,
            "policies_rerun": False,
            "training_or_adaptation": False,
            "source": "immutable episode metrics read with git show",
        },
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_seeds": list(PAPER_SEEDS),
            "content_fingerprints_constructed": 0,
        },
    }


def prepare() -> dict[str, Any]:
    if FREEZE.exists():
        existing = load_json(FREEZE)
        requested = _prepared_payload()
        comparable = dict(existing)
        comparable.pop("frozen_at_utc", None)
        if comparable != requested:
            raise RuntimeError("contextual freeze exists with different content")
        return existing
    if any(EPISODES.rglob("*.json")) or RESULTS.exists() or PAIRED.exists():
        raise RuntimeError("cannot freeze after contextual execution artifacts exist")
    payload = _prepared_payload()
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(FREEZE, payload)
    integrity = {
        "schema_version": 1,
        "stage": "PREEXECUTION_FROZEN",
        "freeze_sha256": file_sha256(FREEZE),
        "protocol_sha256": payload["protocol_sha256"],
        "runner_sha256": payload["runner_sha256"],
        "controller_hashes_verified": True,
        "immutable_rl_artifacts_verified": True,
        "runtime_task_fingerprints_verified": 36,
        "scientific_episode_records": 0,
        "rl_policies_rerun": False,
        "paper": payload["paper"],
    }
    _atomic_json(INTEGRITY, integrity)
    return payload


def _assert_prepared() -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("run prepare and freeze the protocol before evaluation")
    frozen = load_json(FREEZE)
    if frozen["protocol_sha256"] != file_sha256(PROTOCOL):
        raise RuntimeError("contextual protocol changed after freeze")
    if frozen["runner_sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("contextual evaluator changed after freeze")
    current = _reference_integrity()
    if (
        current["source_sha256_observed"]
        != frozen["physical_reference"]["source_sha256_expected"]
    ):
        raise RuntimeError("physical reference changed after contextual freeze")
    for name in PANELS:
        immutable = _immutable_panel(name)
        expected = frozen["panels"][name]
        observed = {key: value for key, value in immutable.items() if key != "rl_rows"}
        if observed != expected:
            raise RuntimeError(f"immutable {name} inputs changed after freeze")
    return frozen


def _episode_path(panel: str, seed: int, controller: str) -> Path:
    return EPISODES / panel / f"seed-{seed}-{controller}.json"


def _existing_episode_paths() -> list[Path]:
    return sorted(EPISODES.rglob("seed-*-*.json")) if EPISODES.exists() else []


def _episode_envelope(
    panel: dict[str, Any], task: dict[str, Any], controller: str, row: dict[str, Any]
) -> dict[str, Any]:
    result = dict(row)
    result["exclusive_failure_mode"] = result.pop("failure_mode")
    return {
        "schema_version": 1,
        "evaluation_id": panel["evaluation_id"],
        "source_commit": panel["source_commit"],
        "task": task,
        "controller": controller,
        "scientific_pair_execution_ordinal": 1,
        "episode": result,
    }


def evaluate(*, recover_untouched_only: bool = False) -> dict[str, Any]:
    frozen = _assert_prepared()
    existing = _existing_episode_paths()
    if existing and not recover_untouched_only:
        raise RuntimeError(
            "contextual episode records already exist; use recover-untouched or "
            "aggregate-only, never repeat a recorded pair"
        )
    for name in PANELS:
        panel = frozen["panels"][name]
        for task in panel["tasks"]:
            _verify_runtime_task(task)
            seed = int(task["numeric_seed"])
            oracle_path = _episode_path(name, seed, "oracle")
            if not oracle_path.exists():
                row = solve_oracle(seed, refinement_iterations=10)
                row.pop("actions")
                _atomic_json(
                    oracle_path,
                    _episode_envelope(panel, task, "oracle", row),
                )
            causal_path = _episode_path(name, seed, "causal")
            if not causal_path.exists():
                row = rollout(seed, "causal")
                row.pop("actions")
                _atomic_json(
                    causal_path,
                    _episode_envelope(panel, task, "causal", row),
                )
    return aggregate_only()


def _load_episode_matrix() -> dict[str, dict[str, list[dict[str, Any]]]]:
    frozen = _assert_prepared()
    expected: set[Path] = set()
    matrix: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name in PANELS:
        matrix[name] = {"oracle": [], "causal": []}
        panel = frozen["panels"][name]
        for task in panel["tasks"]:
            seed = int(task["numeric_seed"])
            for controller in ("oracle", "causal"):
                path = _episode_path(name, seed, controller)
                expected.add(path)
                if not path.exists():
                    raise RuntimeError(f"missing immutable episode record {path}")
                envelope = load_json(path)
                if envelope["task"] != task or envelope["controller"] != controller:
                    raise RuntimeError(f"episode envelope identity mismatch: {path}")
                row = dict(envelope["episode"])
                row["failure_mode"] = row["exclusive_failure_mode"]
                matrix[name][controller].append(row)
    observed = set(_existing_episode_paths())
    if observed != expected:
        extras = sorted(str(path.relative_to(HERE)) for path in observed - expected)
        raise RuntimeError(f"unexpected/duplicate contextual episode records: {extras}")
    return matrix


def _stats(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "count": len(data),
        "mean": fmean(data) if data else None,
        "median": median(data) if data else None,
        "minimum": min(data) if data else None,
        "maximum": max(data) if data else None,
        "values": data,
    }


def _physical_pair_analysis(
    oracle: list[dict[str, Any]], causal: list[dict[str, Any]]
) -> dict[str, Any]:
    by_oracle = {int(row["track_seed"]): row for row in oracle}
    by_causal = {int(row["track_seed"]): row for row in causal}
    common = []
    for seed in sorted(by_oracle):
        left, right = by_oracle[seed], by_causal[seed]
        if left["canonical_success"] and right["canonical_success"]:
            energy = right["signed_net_energy_kwh"] - left["signed_net_energy_kwh"]
            common.append(
                {
                    "track_seed": seed,
                    "oracle_energy_kwh": left["signed_net_energy_kwh"],
                    "causal_energy_kwh": right["signed_net_energy_kwh"],
                    "causal_minus_oracle_energy_kwh": energy,
                    "causal_minus_oracle_energy_percent": (
                        100.0 * energy / left["signed_net_energy_kwh"]
                    ),
                    "oracle_travel_time_s": left["travel_time_s"],
                    "causal_travel_time_s": right["travel_time_s"],
                    "causal_minus_oracle_travel_time_s": (
                        right["travel_time_s"] - left["travel_time_s"]
                    ),
                }
            )
    oracle_mean = fmean(row["oracle_energy_kwh"] for row in common) if common else None
    causal_mean = fmean(row["causal_energy_kwh"] for row in common) if common else None
    return {
        "both_references_feasible_count": len(common),
        "per_track": common,
        "energy_causal_minus_oracle_kwh": _stats(
            row["causal_minus_oracle_energy_kwh"] for row in common
        ),
        "energy_causal_minus_oracle_percent_per_track": _stats(
            row["causal_minus_oracle_energy_percent"] for row in common
        ),
        "aggregate_mean_energy_difference_percent": (
            100.0 * (causal_mean - oracle_mean) / oracle_mean
            if common and oracle_mean is not None and causal_mean is not None
            else None
        ),
        "travel_time_causal_minus_oracle_s": _stats(
            row["causal_minus_oracle_travel_time_s"] for row in common
        ),
    }


def _normalize_rl_row(panel: str, row: dict[str, Any]) -> dict[str, Any]:
    if panel == "sameinfo":
        return {
            "track_seed": int(row["validation_task"]),
            "policy": row["policy"],
            "training_seed": int(row["training_seed"]),
            "method": str(row["method"]).lower(),
            "feasible": bool(row["feasible"]),
            "completion": bool(row["completed"]),
            "deadline_compliance": bool(row["deadline_compliant"]),
            "speed_compliance": bool(row["speed_compliant"]),
            "travel_time_s": float(row["travel_time_s"]),
            "final_position_m": float(row["final_position_m"]),
            "max_speed_violation_m_s": float(row["maximum_speed_violation_m_s"]),
            "integrated_speed_violation_m": float(
                row["integrated_speed_violation_m"]
            ),
            "signed_net_energy_kwh": float(row["signed_net_energy_kwh"]),
            "exclusive_failure_mode": row["exclusive_failure_mode"],
        }
    return {
        "track_seed": int(row["validation_task_seed"]),
        "policy": row["policy"],
        "training_seed": int(row["training_seed"]),
        "method": str(row["method"]).lower(),
        "feasible": bool(row["feasible"]),
        "completion": bool(row["completion"]),
        "deadline_compliance": bool(row["deadline_compliance"]),
        "speed_compliance": bool(row["speed_compliance"]),
        "travel_time_s": float(row["travel_time_s"]),
        "final_position_m": float(row["final_progress"]) * 1000.0,
        "max_speed_violation_m_s": float(row["max_speed_violation_m_s"]),
        "integrated_speed_violation_m": float(row["integrated_speed_violation_m"]),
        "signed_net_energy_kwh": float(row["signed_net_energy_kwh"]),
        "exclusive_failure_mode": row["exclusive_failure_mode"],
    }


def _group_summary(
    rows: list[dict[str, Any]], references: dict[int, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    failed = [row for row in rows if not row["feasible"]]
    failed_tracks = {row["track_seed"] for row in failed}
    return {
        "episode_count": len(rows),
        "rl_failed_policy_track_pairs": len(failed),
        "rl_failed_pairs_oracle_feasible": sum(
            references[row["track_seed"]]["oracle"]["canonical_success"]
            for row in failed
        ),
        "rl_failed_pairs_causal_feasible": sum(
            references[row["track_seed"]]["causal"]["canonical_success"]
            for row in failed
        ),
        "distinct_rl_failed_tracks": len(failed_tracks),
        "distinct_failed_tracks_oracle_feasible": sum(
            references[seed]["oracle"]["canonical_success"]
            for seed in failed_tracks
        ),
        "distinct_failed_tracks_causal_feasible": sum(
            references[seed]["causal"]["canonical_success"]
            for seed in failed_tracks
        ),
        "failure_components": {
            "speed_related": sum(not row["speed_compliance"] for row in failed),
            "deadline_related": sum(
                not row["deadline_compliance"] for row in failed
            ),
            "noncompletion": sum(not row["completion"] for row in failed),
        },
        "exclusive_failure_modes": dict(
            sorted(Counter(row["exclusive_failure_mode"] for row in failed).items())
        ),
    }


def _rl_energy(
    rows: list[dict[str, Any]], references: dict[int, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    result = {}
    for kind in ("oracle", "causal"):
        pairs = []
        for row in rows:
            reference = references[row["track_seed"]][kind]
            if row["feasible"] and reference["canonical_success"]:
                difference = (
                    row["signed_net_energy_kwh"]
                    - reference["signed_net_energy_kwh"]
                )
                pairs.append(
                    {
                        "policy": row["policy"],
                        "track_seed": row["track_seed"],
                        "rl_energy_kwh": row["signed_net_energy_kwh"],
                        "reference_energy_kwh": reference[
                            "signed_net_energy_kwh"
                        ],
                        "rl_minus_reference_energy_kwh": difference,
                        "rl_minus_reference_energy_percent": (
                            100.0
                            * difference
                            / reference["signed_net_energy_kwh"]
                        ),
                    }
                )
        result[kind] = {
            "paired_feasible_policy_track_count": len(pairs),
            "rl_minus_reference_energy_kwh": _stats(
                row["rl_minus_reference_energy_kwh"] for row in pairs
            ),
            "rl_minus_reference_energy_percent": _stats(
                row["rl_minus_reference_energy_percent"] for row in pairs
            ),
            "pairs": pairs,
        }
    return result


def _rl_comparison(
    name: str,
    oracle: list[dict[str, Any]],
    causal: list[dict[str, Any]],
) -> dict[str, Any]:
    immutable = _immutable_panel(name)
    rows = [_normalize_rl_row(name, row) for row in immutable["rl_rows"]]
    oracle_by_seed = {int(row["track_seed"]): row for row in oracle}
    causal_by_seed = {int(row["track_seed"]): row for row in causal}
    references = {
        seed: {"oracle": oracle_by_seed[seed], "causal": causal_by_seed[seed]}
        for seed in oracle_by_seed
    }
    policies: dict[str, list[dict[str, Any]]] = defaultdict(list)
    methods: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        policies[row["policy"]].append(row)
        methods[row["method"]].append(row)
    task_comparison = []
    for seed in sorted(references):
        task_comparison.append(
            {
                "track_seed": seed,
                "oracle": {
                    key: references[seed]["oracle"][key]
                    for key in (
                        "canonical_success",
                        "completion",
                        "deadline_compliant",
                        "speed_compliant",
                        "travel_time_s",
                        "signed_net_energy_kwh",
                    )
                },
                "causal": {
                    key: references[seed]["causal"][key]
                    for key in (
                        "canonical_success",
                        "completion",
                        "deadline_compliant",
                        "speed_compliant",
                        "travel_time_s",
                        "signed_net_energy_kwh",
                    )
                },
                "rl": [row for row in rows if row["track_seed"] == seed],
            }
        )
    return {
        "immutable_rl_artifacts": {
            "results": immutable["results_artifact"],
            "split": immutable["split_artifact"],
        },
        "by_policy_and_training_seed": {
            policy: {
                "training_seed": values[0]["training_seed"],
                "method": values[0]["method"],
                **_group_summary(values, references),
                "paired_feasible_energy": _rl_energy(values, references),
            }
            for policy, values in sorted(policies.items())
        },
        "pooled_by_method": {
            method: {
                **_group_summary(values, references),
                "paired_feasible_energy": _rl_energy(values, references),
            }
            for method, values in sorted(methods.items())
        },
        "task_comparison": task_comparison,
    }


def _render_results(payload: dict[str, Any]) -> str:
    lines = [
        "# Contextual Physical-Reference Results",
        "",
        "Status: **complete post-hoc contextual analysis**. This is not a new ",
        "independent held-out benchmark, and no RL policy was rerun.",
        "",
        "## Physical references",
        "",
        (
            "| Panel | Reference | RSR | Completion | Deadline | Speed | "
            "Feasible energy mean / median (kWh) |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, panel in payload["panels"].items():
        for kind in ("oracle", "causal"):
            summary = panel[kind]["summary"]
            energy = summary["feasible_only_energy_kwh"]
            lines.append(
                f"| `{panel['evaluation_id']}` | {kind} | "
                f"{summary['requirement_satisfaction_count']}/18 | "
                f"{summary['completion_count']}/18 | "
                f"{summary['deadline_compliance_count']}/18 | "
                f"{summary['speed_compliance_count']}/18 | "
                f"{energy['mean']:.9f} / {energy['median']:.9f} |"
            )
    lines.extend(["", "## Paired reference energy", ""])
    for name, panel in payload["panels"].items():
        paired = panel["paired_feasible_only"]
        energy = paired["energy_causal_minus_oracle_kwh"]
        time = paired["travel_time_causal_minus_oracle_s"]
        lines.append(
            f"- `{panel['evaluation_id']}`: {energy['count']} feasible pairs; "
            f"causal−Oracle energy mean `{energy['mean']:.9f}` kWh, median "
            f"`{energy['median']:.9f}`, range `{energy['minimum']:.9f}` to "
            f"`{energy['maximum']:.9f}`; aggregate mean difference "
            f"`{paired['aggregate_mean_energy_difference_percent']:.3f}%`; "
            f"travel-time difference mean `{time['mean']:.3f}` s."
        )
    lines.extend(["", "## RL failures with physical witnesses", ""])
    for name, panel in payload["panels"].items():
        lines.append(f"### {panel['evaluation_id']}")
        lines.append("")
        lines.append(
            "| Method | Failed policy-track pairs | Oracle witnesses | "
            "Causal witnesses | Distinct failed tracks | "
            "Oracle / causal solvable tracks |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for method, summary in panel["rl_comparison"]["pooled_by_method"].items():
            lines.append(
                f"| {method} | {summary['rl_failed_policy_track_pairs']} | "
                f"{summary['rl_failed_pairs_oracle_feasible']} | "
                f"{summary['rl_failed_pairs_causal_feasible']} | "
                f"{summary['distinct_rl_failed_tracks']} | "
                f"{summary['distinct_failed_tracks_oracle_feasible']} / "
                f"{summary['distinct_failed_tracks_causal_feasible']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Both frozen references were evaluated without adjustment. The precise ",
            "interpretation is recorded in `results.json`; the contextual panels do ",
            "not support a population-wide solvability guarantee. The Oracle is an ",
            "achieved energy reference, not a proof of global optimality. All Paper ",
            "content remained sealed and uninspected.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate_only() -> dict[str, Any]:
    frozen = _assert_prepared()
    matrix = _load_episode_matrix()
    panels = {}
    paired_payload = {"schema_version": 1, "panels": {}}
    total_search = 0
    total_replays = 0
    total_replay_steps = 0
    for name in PANELS:
        oracle = matrix[name]["oracle"]
        causal = matrix[name]["causal"]
        reference_pairs = _physical_pair_analysis(oracle, causal)
        rl = _rl_comparison(name, oracle, causal)
        total_search += sum(int(row["optimization_iterations"]) for row in oracle)
        total_replays += sum(bool(row["numerical_parity"]) for row in oracle)
        total_replay_steps += sum(
            int(row["replay"]["actions_used"]) for row in oracle
        )
        panels[name] = {
            "evaluation_id": frozen["panels"][name]["evaluation_id"],
            "tasks": frozen["panels"][name]["tasks"],
            "oracle": {"episodes": oracle, "summary": summarize(oracle)},
            "causal": {"episodes": causal, "summary": summarize(causal)},
            "paired_task_level_differences_oracle_minus_causal": paired_differences(
                oracle, causal
            ),
            "paired_feasible_only": reference_pairs,
            "rl_comparison": rl,
        }
        paired_payload["panels"][name] = {
            "evaluation_id": panels[name]["evaluation_id"],
            "physical_references": reference_pairs,
            "rl_failed_witness_analysis": {
                "by_policy_and_training_seed": rl["by_policy_and_training_seed"],
                "pooled_by_method": rl["pooled_by_method"],
            },
        }
    all_full = all(
        panels[name][kind]["summary"]["requirement_satisfaction_count"] == 18
        for name in panels
        for kind in ("oracle", "causal")
    )
    payload = {
        "schema_version": 1,
        "study": "post-hoc contextual physical-reference analysis",
        "scope": (
            "previously opened RL Validation tasks; not an independent held-out "
            "generalization benchmark"
        ),
        "freeze_sha256": file_sha256(FREEZE),
        "panels": panels,
        "execution_accounting": {
            "scientific_controller_task_pairs": 72,
            "unique_scientific_controller_task_pairs": 72,
            "duplicate_scientific_pairs": 0,
            "oracle_selected_outcomes": 36,
            "causal_outcomes": 36,
            "oracle_internal_search_trajectories": total_search,
            "oracle_replay_checks": total_replays,
            "oracle_replay_simulator_steps": total_replay_steps,
            "rl_policy_episodes_executed": 0,
        },
        "factual_interpretation": {
            "both_references_full_feasibility_on_both_panels": all_full,
            "oracle_solved_every_observed_rl_failed_track": all(
                summary["distinct_rl_failed_tracks"]
                == summary["distinct_failed_tracks_oracle_feasible"]
                for panel in panels.values()
                for summary in panel["rl_comparison"]["pooled_by_method"].values()
            ),
            "causal_solved_every_observed_rl_failed_track": all(
                summary["distinct_rl_failed_tracks"]
                == summary["distinct_failed_tracks_causal_feasible"]
                for panel in panels.values()
                for summary in panel["rl_comparison"]["pooled_by_method"].values()
            ),
            "hidden_future_route_information_necessary_on_observed_tasks": (
                False if all_full else None
            ),
            "hidden_information_general_irrelevance_claimed": False,
            "energy_opportunity_interpretation": (
                "An achieved causal-to-Oracle energy gap remains on these panels, "
                "which motivates but does not prove learnable improvement; the "
                "Oracle is privileged and not globally optimal, and a non-learning "
                "causal energy-aware control is required."
            ),
        },
        "integrity": {
            "controller_hashes_verified": True,
            "rl_result_artifacts_read_from_exact_commits": True,
            "rl_policies_rerun": False,
            "controllers_retuned_after_results": False,
            "paper": frozen["paper"],
        },
    }
    _atomic_json(PAIRED, paired_payload)
    _atomic_json(RESULTS, payload)
    _atomic_text(RESULTS_MD, _render_results(payload))
    integrity = {
        "schema_version": 1,
        "stage": "CONTEXTUAL_ANALYSIS_COMPLETE",
        "freeze_sha256": file_sha256(FREEZE),
        "protocol_sha256": file_sha256(PROTOCOL),
        "runner_sha256": file_sha256(Path(__file__)),
        "controller_hashes_verified": True,
        "controller_source_sha256": source_hashes(),
        "scientific_episode_records": 72,
        "unique_controller_task_pairs": 72,
        "duplicate_controller_task_pairs": 0,
        "oracle_replay_checks": total_replays,
        "oracle_replay_simulator_steps": total_replay_steps,
        "oracle_internal_search_trajectories": total_search,
        "rl_policies_rerun": False,
        "rl_training_or_adaptation": False,
        "immutable_rl_artifacts": {
            name: {
                "source_commit": frozen["panels"][name]["source_commit"],
                "results_artifact": frozen["panels"][name]["results_artifact"],
                "split_artifact": frozen["panels"][name]["split_artifact"],
            }
            for name in PANELS
        },
        "runtime_task_fingerprints_verified": 36,
        "output_sha256": {
            "results.json": file_sha256(RESULTS),
            "paired_analysis.json": file_sha256(PAIRED),
            "RESULTS.md": file_sha256(RESULTS_MD),
        },
        "paper": frozen["paper"],
    }
    _atomic_json(INTEGRITY, integrity)
    return payload


def verify() -> dict[str, Any]:
    payload = aggregate_only()
    integrity = load_json(INTEGRITY)
    if payload["execution_accounting"]["scientific_controller_task_pairs"] != 72:
        raise RuntimeError("contextual task accounting failed")
    if integrity["oracle_replay_checks"] != 36:
        raise RuntimeError("Oracle replay accounting failed")
    if integrity["paper"]["content_fingerprints_constructed"] != 0:
        raise RuntimeError("Paper seal failed")
    return integrity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "evaluate",
            "recover-untouched",
            "aggregate-only",
            "verify",
        ),
    )
    args = parser.parse_args(argv)
    functions = {
        "prepare": prepare,
        "evaluate": evaluate,
        "recover-untouched": lambda: evaluate(recover_untouched_only=True),
        "aggregate-only": aggregate_only,
        "verify": verify,
    }
    payload = functions[args.stage]()
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
