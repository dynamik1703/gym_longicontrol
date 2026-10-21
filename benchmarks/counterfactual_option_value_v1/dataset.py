"""Deterministic snapshot collection and exactly-once counterfactual dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import gymnasium as gym

import gym_longicontrol  # noqa: F401 - registers StochasticTrack-v1
from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice

from .alternative_selector import select_alternatives
from .branch_runner import (
    ADAPTIVE_HIGH_RESERVE_S,
    ADAPTIVE_LOW_RESERVE_S,
    DECISION_INTERVAL_NATIVE_STEPS,
    run_branch,
)
from .execution import ExclusiveLease, atomic_json, sha256_file, utc_now
from .instrumented_controller import InstrumentedContinueController
from .integrity import (
    BRANCH_LEDGER,
    CANONICAL,
    HERE,
    SPLIT_LEDGER,
    require_not_paper,
    verify_frozen_method,
    verify_native_records,
)
from .snapshot import capture_snapshot, read_snapshot, write_snapshot
from .split import canonical_sha256

TARGETS_M = (80.0, 180.0, 300.0, 430.0, 560.0, 700.0, 820.0, 930.0)
SNAPSHOT_DIR = HERE / "snapshots"
TRACE_DIR = HERE / "native_traces"
RECORD_DIR = HERE / "branch_records"
PAIR_RECORD_DIR = HERE / "counterfactual_records"
MANIFEST = HERE / "dataset_manifest.json"
LEASE = HERE / "dataset.lease"
PANEL_KEYS = ("fit_panel", "calibration_panel", "test_panel")


def _environment() -> gym.Env:
    return gym.make("StochasticTrack-v1", max_episode_steps=1_800)


def _snapshot_id(panel_id: str, numeric_seed: int, target_m: float) -> str:
    return f"{panel_id}__{numeric_seed}__{int(target_m):04d}m"


def _candidate_metadata(
    controller: InstrumentedContinueController,
) -> list[dict[str, Any]]:
    if controller.last_selection is None:
        raise RuntimeError("valid strategic boundary omitted selector candidates")
    return [
        {
            "action_id": candidate.high_level_action,
            "profile_id": candidate.profile_id,
            "family": candidate.family,
            "objective_kwh": candidate.objective_kwh,
            "future_option_time_s": candidate.future_option_time_s,
            "future_option_energy_kwh": candidate.future_option_energy_kwh,
            "terminal_time_s": candidate.terminal_time_s,
            "terminal_energy_kwh": candidate.terminal_energy_kwh,
            "lateness_penalty_kwh": candidate.lateness_penalty_kwh,
            "feasible": candidate.feasible,
        }
        for candidate in controller.last_selection.candidates
    ]


def _heuristic_runner_up(controller: InstrumentedContinueController) -> str | None:
    if controller.last_selection is None:
        return None
    reference = controller.last_selection.selected.high_level_action
    candidates = sorted(
        (
            candidate
            for candidate in controller.last_selection.candidates
            if candidate.feasible and candidate.high_level_action != reference
        ),
        key=lambda candidate: (
            candidate.objective_kwh,
            candidate.future_option_time_s + candidate.terminal_time_s,
            candidate.future_option_energy_kwh
            + candidate.terminal_energy_kwh
            + candidate.lateness_penalty_kwh,
            candidate.high_level_action,
        ),
    )
    return None if not candidates else candidates[0].high_level_action


def collect_route_snapshots(
    *, panel_id: str, numeric_seed: int, route_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_not_paper(numeric_seed)
    environment = _environment()
    controller = InstrumentedContinueController()
    selector = AdaptivePrice(
        high_reserve_s=ADAPTIVE_HIGH_RESERVE_S,
        low_reserve_s=ADAPTIVE_LOW_RESERVE_S,
    )
    observation, info = environment.reset(seed=int(numeric_seed))
    controller.reset()
    selector.reset()
    held_price: float | None = None
    target_index = 0
    rows: list[dict[str, Any]] = []
    unavailable: list[float] = []
    step = 0
    ended = False
    while not ended:
        boundary = step % DECISION_INTERVAL_NATIVE_STEPS == 0
        target: float | None = None
        snapshot_payload: dict[str, Any] | None = None
        if (
            boundary
            and target_index < len(TARGETS_M)
            and float(info["position_m"]) >= TARGETS_M[target_index]
        ):
            target = TARGETS_M[target_index]
            snapshot_payload = capture_snapshot(
                environment=environment,
                controller=controller,
                observation=observation,
                info=info,
                held_price=held_price,
                native_step=step,
                selector_last_action=selector.last_action,
            )
        option_observation = None
        if boundary:
            base = controller.prepare_strategy_step(observation)
            option_observation = controller.rl_observation(base)
            held_price = float(selector.action(base))
        if held_price is None:
            raise RuntimeError("normal trajectory lacks held price")
        action = controller.action(observation, held_price)
        decision = controller.last_decision
        if decision is None:
            raise RuntimeError("normal trajectory omitted native decision")
        if target is not None and snapshot_payload is not None:
            if controller.last_selection is None or decision.high_level_action is None:
                unavailable.append(target)
            else:
                assert option_observation is not None
                alternatives = select_alternatives(
                    reference_action_id=decision.high_level_action,
                    candidates=controller.last_selection.candidates,
                )
                snapshot_id = _snapshot_id(panel_id, numeric_seed, target)
                snapshot_path = SNAPSHOT_DIR / f"{snapshot_id}.pkl.gz"
                snapshot_meta = write_snapshot(snapshot_path, snapshot_payload)
                rows.append(
                    {
                        "snapshot_id": snapshot_id,
                        "panel_id": panel_id,
                        "numeric_seed": int(numeric_seed),
                        "route_sha256": route_sha256,
                        "nominal_target_m": target,
                        "actual_position_m": float(info["position_m"]),
                        "native_step": step,
                        "causal_observation_70d": option_observation.vector.tolist(),
                        "valid_action_mask": (
                            option_observation.valid_action_mask.tolist()
                        ),
                        "active_option": snapshot_payload["controller"][
                            "active_option"
                        ],
                        "previous_termination": snapshot_payload["controller"][
                            "previous_termination"
                        ],
                        "reference_action_id": decision.high_level_action,
                        "alternative_action_ids": list(
                            alternatives.alternative_action_ids
                        ),
                        "valid_action_ids": list(alternatives.valid_action_ids),
                        "heuristic_runner_up_action_id": _heuristic_runner_up(
                            controller
                        ),
                        "candidate_metadata": _candidate_metadata(controller),
                        "snapshot_path": str(snapshot_path.relative_to(HERE)),
                        "snapshot_sha256": snapshot_meta["sha256"],
                        "snapshot_size_bytes": snapshot_meta["size_bytes"],
                    }
                )
            target_index += 1
        observation, _, terminated, truncated, info = environment.step([action])
        ended = bool(terminated or truncated)
        controller.observe_executed_action(action, episode_ended=ended)
        step += 1
    unavailable.extend(TARGETS_M[target_index:])
    route_result = {
        "panel_id": panel_id,
        "numeric_seed": int(numeric_seed),
        "route_sha256": route_sha256,
        "snapshot_count": len(rows),
        "unavailable_targets_m": unavailable,
        "normal_trajectory_native_steps": step,
        "final_episode_metrics": info["episode_metrics"],
    }
    environment.close()
    return rows, route_result


def collect_snapshots() -> dict[str, Any]:
    verify_frozen_method(require_labels_authorized=True)
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if not canonical["execution"]["preflight_complete"]:
        raise PermissionError("preflight must pass before opening study panels")
    if MANIFEST.exists() or BRANCH_LEDGER.exists():
        raise PermissionError("snapshot sampling was already materialized")
    split = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    route_results: list[dict[str, Any]] = []
    with ExclusiveLease(LEASE) as lease:
        for panel_key in PANEL_KEYS:
            panel = split[panel_key]
            for task in panel["accepted_tasks"]:
                lease.heartbeat(f"SNAPSHOT:{panel['id']}:{task['numeric_seed']}")
                selected, route = collect_route_snapshots(
                    panel_id=panel["id"],
                    numeric_seed=int(task["numeric_seed"]),
                    route_sha256=task["route_sha256"],
                )
                rows.extend(selected)
                route_results.append(route)
    if len(rows) > 288:
        raise RuntimeError("snapshot budget exceeded")
    manifest = {
        "schema_version": 1,
        "state": "SNAPSHOTS_FROZEN_BRANCHES_UNOPENED",
        "targets_m": list(TARGETS_M),
        "sampling_rule": "first valid 1-Hz boundary at or after each target",
        "snapshot_count": len(rows),
        "snapshots": rows,
        "route_results": route_results,
        "ordered_snapshot_sha256": canonical_sha256(
            [
                {
                    "snapshot_id": row["snapshot_id"],
                    "snapshot_sha256": row["snapshot_sha256"],
                    "reference_action_id": row["reference_action_id"],
                    "alternative_action_ids": row["alternative_action_ids"],
                }
                for row in rows
            ]
        ),
        "selection_used_counterfactual_outcomes": False,
        "paper_state": "SEALED_UNINSPECTED",
    }
    atomic_json(MANIFEST, manifest)
    identities = []
    for row in rows:
        identities.append(
            {
                "branch_id": f"{row['snapshot_id']}::REFERENCE",
                "snapshot_id": row["snapshot_id"],
                "role": "REFERENCE",
                "action_id": row["reference_action_id"],
            }
        )
        identities.extend(
            {
                "branch_id": f"{row['snapshot_id']}::{action_id}",
                "snapshot_id": row["snapshot_id"],
                "role": "ALTERNATIVE",
                "action_id": action_id,
            }
            for action_id in row["alternative_action_ids"]
        )
    ledger = {
        "schema_version": 1,
        "state": "PREPARED_UNOPENED",
        "created_utc": utc_now(),
        "attempt_id": None,
        "snapshot_manifest_sha256": sha256_file(MANIFEST),
        "identity_count": len(identities),
        "identities_sha256": canonical_sha256(identities),
        "branches": {
            row["branch_id"]: {
                **row,
                "state": "PENDING",
                "record_path": None,
                "record_sha256": None,
            }
            for row in identities
        },
    }
    atomic_json(BRANCH_LEDGER, ledger)
    canonical["execution"]["panels_opened"] = True
    canonical["execution"]["snapshots_frozen"] = True
    atomic_json(CANONICAL, canonical)
    return manifest


def _update_ledger(branch_id: str, state: str, **updates: Any) -> dict[str, Any]:
    ledger = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
    branch = ledger["branches"][branch_id]
    expected = "PENDING" if state == "STARTED" else "STARTED"
    if branch["state"] != expected:
        raise RuntimeError(
            f"branch {branch_id} expected {expected}, observed {branch['state']}"
        )
    branch["state"] = state
    branch.update(updates)
    atomic_json(BRANCH_LEDGER, ledger)
    return ledger


def run_dataset() -> dict[str, Any]:
    verify_frozen_method(require_labels_authorized=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ledger = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
    if ledger["state"] != "PREPARED_UNOPENED":
        raise RuntimeError("prior dataset attempt requires audit")
    if ledger["snapshot_manifest_sha256"] != sha256_file(MANIFEST):
        raise RuntimeError("snapshot manifest changed")
    ledger["state"] = "RUNNING"
    ledger["attempt_id"] = hashlib.sha256(
        f"{utc_now()}:{ledger['identities_sha256']}".encode()
    ).hexdigest()
    ledger["opened_utc"] = utc_now()
    atomic_json(BRANCH_LEDGER, ledger)
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    canonical["execution"]["scientific_labels_opened"] = True
    atomic_json(CANONICAL, canonical)
    rows_by_id = {row["snapshot_id"]: row for row in manifest["snapshots"]}
    with ExclusiveLease(LEASE) as lease:
        for branch_id in list(ledger["branches"]):
            ledger = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
            branch = ledger["branches"][branch_id]
            if branch["state"] != "PENDING":
                raise RuntimeError(
                    "non-pending branch encountered during fresh attempt"
                )
            lease.heartbeat(branch_id)
            _update_ledger(branch_id, "STARTED", started_utc=utc_now())
            snapshot_row = rows_by_id[branch["snapshot_id"]]
            snapshot_payload = read_snapshot(
                HERE / snapshot_row["snapshot_path"],
                snapshot_row["snapshot_sha256"],
            )
            record_path = (
                RECORD_DIR / f"{hashlib.sha256(branch_id.encode()).hexdigest()}.json"
            )
            trace_path = (
                TRACE_DIR / f"{hashlib.sha256(branch_id.encode()).hexdigest()}.json.gz"
            )
            result = run_branch(
                snapshot_payload,
                branch_id=branch_id,
                action_id=(
                    None if branch["role"] == "REFERENCE" else branch["action_id"]
                ),
                trace_path=trace_path,
            )
            from .branch_runner import read_trace

            trace_records = read_trace(trace_path, result["trace"]["sha256"])
            audit = verify_native_records(trace_records)
            if (
                abs(audit["signed_energy_kwh"] - result["remaining_signed_energy_kwh"])
                > 1.0e-12
            ):
                raise RuntimeError("trace and branch remaining energy disagree")
            result["integrity_audit"] = audit
            atomic_json(record_path, result)
            _update_ledger(
                branch_id,
                "COMPLETE",
                completed_utc=utc_now(),
                record_path=str(record_path.relative_to(HERE)),
                record_sha256=sha256_file(record_path),
                trace_path=str(trace_path.relative_to(HERE)),
                trace_sha256=result["trace"]["sha256"],
            )
            current = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
            usage = current.get("resource_usage", {})
            completed_count = int(usage.get("completed_branch_count", 0)) + 1
            active_seconds = float(
                usage.get("branch_active_process_time_s", 0.0)
            ) + float(result["active_process_time_s"])
            artifact_bytes = (
                int(usage.get("retained_branch_artifact_bytes", 0))
                + record_path.stat().st_size
                + trace_path.stat().st_size
            )
            current["resource_usage"] = {
                "completed_branch_count": completed_count,
                "branch_active_process_time_s": active_seconds,
                "retained_branch_artifact_bytes": artifact_bytes,
            }
            atomic_json(BRANCH_LEDGER, current)
            if active_seconds > 12.0 * 3600.0:
                raise RuntimeError("12-hour active-compute limit exceeded; stopped")
            if artifact_bytes > 4_000_000_000:
                raise RuntimeError("4 GB retained-artifact limit exceeded; stopped")
    ledger = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
    if any(row["state"] != "COMPLETE" for row in ledger["branches"].values()):
        raise RuntimeError("dataset branches incomplete")
    pair_records = []
    for snapshot_id, snapshot_row in rows_by_id.items():
        reference_ledger = ledger["branches"][f"{snapshot_id}::REFERENCE"]
        reference = json.loads(
            (HERE / reference_ledger["record_path"]).read_text(encoding="utf-8")
        )
        alternatives = []
        for action_id in snapshot_row["alternative_action_ids"]:
            branch = ledger["branches"][f"{snapshot_id}::{action_id}"]
            alternative = json.loads(
                (HERE / branch["record_path"]).read_text(encoding="utf-8")
            )
            alternatives.append(
                {
                    "action_id": action_id,
                    "branch_id": alternative["branch_id"],
                    "feasible": alternative["feasible"],
                    "remaining_signed_energy_kwh": alternative[
                        "remaining_signed_energy_kwh"
                    ],
                    "remaining_travel_time_s": alternative["remaining_travel_time_s"],
                    "delta_energy_kwh": alternative["remaining_signed_energy_kwh"]
                    - reference["remaining_signed_energy_kwh"],
                    "delta_arrival_time_s": alternative["remaining_travel_time_s"]
                    - reference["remaining_travel_time_s"],
                    "introduced_infeasibility": bool(
                        reference["feasible"] and not alternative["feasible"]
                    ),
                    "first_block_executed_behavior_differs": (
                        alternative["first_block_action_sha256"]
                        != reference["first_block_action_sha256"]
                    ),
                    "first_block_option_memory_differs": (
                        alternative["first_block_option_memory_sha256"]
                        != reference["first_block_option_memory_sha256"]
                    ),
                    "intervention_count": alternative["intervention_count"],
                    "physics_prediction_calls": alternative["physics_prediction_calls"],
                    "native_step_count": alternative["native_step_count"],
                    "active_process_time_s": alternative["active_process_time_s"],
                }
            )
        pair = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "panel_id": snapshot_row["panel_id"],
            "numeric_seed": snapshot_row["numeric_seed"],
            "route_sha256": snapshot_row["route_sha256"],
            "nominal_target_m": snapshot_row["nominal_target_m"],
            "actual_position_m": snapshot_row["actual_position_m"],
            "reference_action_id": snapshot_row["reference_action_id"],
            "reference_feasible": reference["feasible"],
            "reference_remaining_signed_energy_kwh": reference[
                "remaining_signed_energy_kwh"
            ],
            "reference_remaining_travel_time_s": reference["remaining_travel_time_s"],
            "alternatives": alternatives,
        }
        pair_path = PAIR_RECORD_DIR / f"{snapshot_id}.json"
        atomic_json(pair_path, pair)
        pair_records.append(
            {
                "snapshot_id": snapshot_id,
                "path": str(pair_path.relative_to(HERE)),
                "sha256": sha256_file(pair_path),
            }
        )
    ledger["state"] = "COMPLETE"
    ledger["completed_utc"] = utc_now()
    ledger["pair_records"] = pair_records
    ledger["pair_records_sha256"] = canonical_sha256(pair_records)
    atomic_json(BRANCH_LEDGER, ledger)
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    canonical["execution"]["dataset_complete"] = True
    atomic_json(CANONICAL, canonical)
    return {"snapshot_count": len(rows_by_id), "branch_count": len(ledger["branches"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("snapshots", "branches"))
    args = parser.parse_args()
    result = collect_snapshots() if args.stage == "snapshots" else run_dataset()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
