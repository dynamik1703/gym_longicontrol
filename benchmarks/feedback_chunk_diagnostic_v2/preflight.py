"""Mandatory A--K gates; this module never runs a scientific episode."""

from __future__ import annotations

import gzip
import json
import math
import os
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from .checkpoint_audit import verify_v1_incident_immutable
from .constants import (
    ABSOLUTE_FLOAT_TOLERANCE,
    AUTHORIZATION,
    C_CHECKPOINTS,
    CANONICAL,
    CONTROLLER_ORDER,
    EXECUTION_MANIFEST,
    H_CHECKPOINTS,
    PAIR_COUNT,
    PAIR_LEDGER,
    PREPARATION_STATUS,
    RESULTS,
    ROOT,
    RUNTIME_MANIFESTS,
    SPLIT_LEDGER,
)
from .evaluation import not_applicable
from .integrity import repository_identity, source_bundle
from .ledger import (
    build_pair_ledger,
    mark_completed,
    mark_started,
)
from .master import INTERPRETERS, WorkerClient
from .split import materialize, verify_namespace_unused
from .utils import (
    atomic_json,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    load_json,
    utc_now,
)
from .verifier import verify_native_evidence, verify_outcome

PREFLIGHT_RESULTS = (
    ROOT / "benchmarks/feedback_chunk_diagnostic_v2/preflight_results.json"
)
RUNTIME_PARITY = ROOT / "benchmarks/feedback_chunk_diagnostic_v2/runtime_parity.json"
RESOURCE_MEASUREMENTS = ROOT / (
    "benchmarks/feedback_chunk_diagnostic_v2/resource_measurements.json"
)


def _native_command(runtime_id: str, command: str) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    completed = subprocess.run(
        (
            str(INTERPRETERS[runtime_id]),
            "-m",
            "benchmarks.feedback_chunk_diagnostic_v2.checkpoint_audit",
            command,
        ),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _core_versions(manifest: dict[str, Any]) -> dict[str, list[str]]:
    wanted = {"numpy", "gymnasium", "torch", "gym-longicontrol"}
    versions: dict[str, list[str]] = {}
    for row in manifest["distributions"]:
        name = row["name"].lower()
        if name in wanted:
            versions.setdefault(name, []).append(row["version"])
    return {name: sorted(set(values)) for name, values in sorted(versions.items())}


def capture_runtime_manifests() -> dict[str, Any]:
    manifests = {
        runtime_id: _native_command(runtime_id, "runtime-manifest")
        for runtime_id in ("C", "H")
    }
    expected = {
        "C": {
            "python": "3.11.13",
            "numpy": ["2.4.6"],
            "gymnasium": ["1.4.0"],
            "torch": ["2.14.1"],
            "gym-longicontrol": ["1.0.0"],
        },
        "H": {
            "python": "3.13.16",
            "numpy": ["2.5.4"],
            "gymnasium": ["1.4.0"],
            "torch": ["2.14.1"],
            "gym-longicontrol": ["1.0.0"],
        },
    }
    for runtime_id, manifest in manifests.items():
        observed = {"python": manifest["python"], **_core_versions(manifest)}
        if observed != expected[runtime_id]:
            raise RuntimeError(
                f"native runtime {runtime_id} core versions differ: {observed}"
            )
        if Path(manifest["executable"]) != INTERPRETERS[runtime_id]:
            raise RuntimeError(f"native runtime {runtime_id} interpreter changed")
    atomic_json(RUNTIME_MANIFESTS, manifests)
    return manifests


def _compare_values(left: Any, right: Any, path: str = "$") -> dict[str, Any]:
    report = {"passed": True, "maximum_absolute_difference": 0.0, "mismatch": None}

    def compare(a: Any, b: Any, location: str) -> None:
        if not report["passed"]:
            return
        if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            if a != b or type(a) is not type(b):
                report.update(passed=False, mismatch=location)
            return
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            difference = abs(float(a) - float(b))
            report["maximum_absolute_difference"] = max(
                report["maximum_absolute_difference"], difference
            )
            if not math.isclose(
                float(a), float(b), rel_tol=0.0, abs_tol=ABSOLUTE_FLOAT_TOLERANCE
            ):
                report.update(passed=False, mismatch=location)
            return
        if type(a) is not type(b):
            report.update(passed=False, mismatch=location)
            return
        if isinstance(a, dict):
            if set(a) != set(b):
                report.update(passed=False, mismatch=location)
                return
            for key in sorted(a):
                compare(a[key], b[key], f"{location}.{key}")
            return
        if isinstance(a, list):
            if len(a) != len(b):
                report.update(passed=False, mismatch=location)
                return
            for index, (a_item, b_item) in enumerate(zip(a, b)):
                compare(a_item, b_item, f"{location}[{index}]")
            return
        if a != b:
            report.update(passed=False, mismatch=location)

    compare(left, right, path)
    return report


def _without_timing(suite: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(suite)
    result.pop("runtime_id", None)
    result.pop("elapsed_seconds", None)
    for row in result["rollouts"]:
        row.pop("elapsed_seconds", None)
    return result


def _schema_gate() -> dict[str, Any]:
    na = not_applicable("synthetic_causal_controller_does_not_compute_bound")
    row = {
        "schema_version": 1,
        "controller_id": "CAUSAL60",
        "native_step": 0,
        "causal_pre_state": {
            "position_m": 0.0,
            "velocity_m_s": 0.0,
            "acceleration_m_s2": 0.0,
            "jerk_m_s3": 0.0,
            "elapsed_time_s": 0.0,
            "total_energy_kwh": 0.0,
        },
        "ordinary_observation": [0.0] * 8,
        "visible_speed_limits": {
            "current_m_s": 0.0,
            "future_m_s": [0.0, 0.0],
            "transition_distances_m": [150.0, 150.0],
        },
        "proposed_controller_action": 0.0,
        "proposed_residual": na,
        "executed_residual": na,
        "final_actuator_action": 0.0,
        "native_action_influence": 0.0,
        "signed_step_energy_kwh": 0.0,
        "endpoint_speed_check": na,
        "known_transition_braking_check": na,
        "unknown_preview_safety_check": na,
        "deadline_completion_bound_s": na,
        "viability_flag": na,
        "shield_intervention": na,
        "intervention_or_cancellation_reason": na,
        "actual_successor_state": {
            "position_m": 1000.0,
            "velocity_m_s": 0.0,
            "acceleration_m_s2": 0.0,
            "elapsed_time_s": 0.1,
            "total_energy_kwh": 0.0,
            "speed_limit_m_s": 0.0,
        },
        "terminated": True,
        "truncated": False,
        "native_decision": {},
    }
    reconstruction = verify_native_evidence([row])
    outcome = {
        key: reconstruction[key]
        for key in (
            "completed",
            "travel_time_s",
            "energy_kwh",
            "speed_violation_count",
            "max_speed_violation_m_s",
            "integrated_speed_violation_m",
            "feasible",
            "feasible_energy_kwh",
            "deadline_compliant",
            "speed_compliant",
        )
    }
    verify_outcome(outcome, reconstruction)
    missing_rejected = False
    bad = deepcopy(row)
    del bad["deadline_completion_bound_s"]
    try:
        verify_native_evidence([bad])
    except RuntimeError:
        missing_rejected = True
    null_applicable_rejected = False
    bad = deepcopy(row)
    bad["deadline_completion_bound_s"] = {"status": "APPLICABLE", "value": None}
    try:
        verify_native_evidence([bad])
    except RuntimeError:
        null_applicable_rejected = True
    return {
        "normalized_schema_accepted": True,
        "independent_accounting_matches": True,
        "missing_completion_bound_rejected": missing_rejected,
        "applicable_null_completion_bound_rejected": null_applicable_rejected,
        "not_applicable_fields_explicit": all(
            row[field]["status"] == "NOT_APPLICABLE"
            for field in (
                "endpoint_speed_check",
                "known_transition_braking_check",
                "unknown_preview_safety_check",
                "deadline_completion_bound_s",
                "viability_flag",
            )
        ),
    }


def _ledger_crash_gate(ledger: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="chunk-v2-ledger-") as directory:
        path = Path(directory) / "ledger.json"
        atomic_json(path, ledger)
        pair_id = ledger["pairs"][0]["pair_id"]
        token = "synthetic-preflight-ownership-token"
        mark_started(
            path, pair_id=pair_id, ownership_token=token, started_at_utc=utc_now()
        )
        persisted_started = load_json(path)["pairs"][0]["state"] == "STARTED"
        duplicate_start_rejected = False
        try:
            mark_started(
                path,
                pair_id=pair_id,
                ownership_token=token,
                started_at_utc=utc_now(),
            )
        except RuntimeError:
            duplicate_start_rejected = True
        mark_completed(
            path,
            pair_id=pair_id,
            ownership_token=token,
            completed_at_utc=utc_now(),
            result_path="synthetic-result.json",
            evidence_path="synthetic-evidence.jsonl.gz",
            result_sha256="0" * 64,
            evidence_sha256="1" * 64,
        )
        persisted_completed = load_json(path)["pairs"][0]["state"] == "COMPLETED"
        duplicate_completion_rejected = False
        try:
            mark_completed(
                path,
                pair_id=pair_id,
                ownership_token=token,
                completed_at_utc=utc_now(),
                result_path="synthetic-result.json",
                evidence_path="synthetic-evidence.jsonl.gz",
                result_sha256="0" * 64,
                evidence_sha256="1" * 64,
            )
        except RuntimeError:
            duplicate_completion_rejected = True
    return {
        "started_persisted_before_reset_mock": persisted_started,
        "ambiguous_started_is_not_fresh": duplicate_start_rejected,
        "completed_persisted": persisted_completed,
        "completed_cannot_repeat": duplicate_completion_rejected,
    }


def run() -> dict[str, Any]:
    started = time.perf_counter()
    repository = repository_identity()
    incident = verify_v1_incident_immutable()
    namespace = verify_namespace_unused()
    split = materialize()
    ledger = build_pair_ledger(split)
    atomic_json(PAIR_LEDGER, ledger)
    runtimes = capture_runtime_manifests()
    workers: dict[str, WorkerClient] = {}
    try:
        ready = {}
        fixture_suites = {}
        production = {}
        fingerprints: dict[str, list[dict[str, Any]]] = {"C": [], "H": []}
        for runtime_id in ("C", "H"):
            workers[runtime_id] = WorkerClient(
                runtime_id,
                stderr_path=ROOT
                / "runs/feedback-chunk-diagnostic-v2"
                / f"preflight-worker-{runtime_id}.stderr.log",
            )
            ready[runtime_id] = workers[runtime_id].ask({"command": "READY"})
            production[runtime_id] = workers[runtime_id].ask(
                {"command": "PRODUCTION_IDENTITY"}
            )["identity"]
            fixture_suites[runtime_id] = workers[runtime_id].ask(
                {"command": "SYNTHETIC_FIXTURES"}
            )
            for task in split["panel"]["accepted_tasks"]:
                fingerprints[runtime_id].append(
                    workers[runtime_id].ask(
                        {
                            "command": "FINGERPRINT",
                            "numeric_seed": task["numeric_seed"],
                        }
                    )
                )
    finally:
        for worker in workers.values():
            worker.close()
    expected_checkpoint_count = {"C": len(C_CHECKPOINTS), "H": len(H_CHECKPOINTS)}
    checkpoint_gate = {
        runtime_id: {
            "loaded_count": len(ready[runtime_id]["loaded_checkpoints"]),
            "expected_count": expected_checkpoint_count[runtime_id],
            "metadata": ready[runtime_id]["loaded_checkpoints"],
            "passed": len(ready[runtime_id]["loaded_checkpoints"])
            == expected_checkpoint_count[runtime_id],
        }
        for runtime_id in ("C", "H")
    }
    production_comparison = _compare_values(production["C"], production["H"])
    fixture_c = _without_timing(fixture_suites["C"])
    fixture_h = _without_timing(fixture_suites["H"])
    fixture_comparison = _compare_values(fixture_c, fixture_h)
    parity = {
        "frozen_tolerance": {
            "absolute": ABSOLUTE_FLOAT_TOLERANCE,
            "relative": 0.0,
            "discrete_and_hash": "exact",
        },
        "production_identity": production,
        "production_comparison": production_comparison,
        "synthetic_fixture_comparison": fixture_comparison,
        "C_trace_sha256": canonical_sha256(fixture_c),
        "H_trace_sha256": canonical_sha256(fixture_h),
        "passed": production_comparison["passed"] and fixture_comparison["passed"],
    }
    atomic_json(RUNTIME_PARITY, parity)
    fingerprint_gate = []
    for task, c_row, h_row in zip(
        split["panel"]["accepted_tasks"], fingerprints["C"], fingerprints["H"]
    ):
        passed = all(
            row["route_sha256"] == task["route_sha256"]
            and row["full_task_sha256"] == task["full_task_sha256"]
            for row in (c_row, h_row)
        )
        fingerprint_gate.append(
            {
                "numeric_seed": task["numeric_seed"],
                "route_sha256": task["route_sha256"],
                "full_task_sha256": task["full_task_sha256"],
                "C_matches": passed,
                "H_matches": passed,
                "cross_runtime_exact": c_row["route_sha256"] == h_row["route_sha256"]
                and c_row["full_task_sha256"] == h_row["full_task_sha256"],
            }
        )
    schema = _schema_gate()
    crash = _ledger_crash_gate(ledger)
    rollout_count = len(fixture_suites["C"]["rollouts"])
    seconds_per_rollout = max(
        fixture_suites[runtime_id]["elapsed_seconds"] / rollout_count
        for runtime_id in ("C", "H")
    )
    projected_seconds = seconds_per_rollout * PAIR_COUNT * 2.0 * 1.1
    compressed_per_rollout = max(
        max(
            len(gzip.compress(canonical_bytes(row), compresslevel=6, mtime=0))
            for row in fixture_suites[runtime_id]["rollouts"]
        )
        for runtime_id in ("C", "H")
    )
    projected_bytes = int(compressed_per_rollout * PAIR_COUNT * 2.0 + 32 * 1024 * 1024)
    resources = {
        "scope": "synthetic non-reserved physical fixtures in both native runtimes",
        "runtime_uncertainty_factor": 2.0,
        "write_and_verification_overhead_factor": 1.1,
        "measured_fixture_rollouts_per_runtime": rollout_count,
        "measured_elapsed_seconds": {
            runtime_id: fixture_suites[runtime_id]["elapsed_seconds"]
            for runtime_id in ("C", "H")
        },
        "conservative_seconds_per_rollout": seconds_per_rollout,
        "projected_active_evaluation_seconds": projected_seconds,
        "active_evaluation_gate_seconds": 12 * 60 * 60,
        "projected_retained_bytes": projected_bytes,
        "retained_gate_bytes": 4 * 1024**3,
        "runtime_gate_passed": projected_seconds <= 12 * 60 * 60,
        "retained_gate_passed": projected_bytes <= 4 * 1024**3,
    }
    atomic_json(RESOURCE_MEASUREMENTS, resources)
    source = source_bundle()
    execution_manifest = {
        "schema_version": 1,
        "study": "FEEDBACK_CHUNK_RUNTIME_ISOLATED_DIAGNOSTIC_V2",
        "repository": repository,
        "canonical_sha256": file_sha256(CANONICAL),
        "split_ledger_sha256": file_sha256(SPLIT_LEDGER),
        "ordered_task_sha256": split["panel"]["ordered_sha256"],
        "pair_schedule_sha256": ledger["schedule_sha256"],
        "controller_order": list(CONTROLLER_ORDER),
        "scientific_source_bundle": source,
        "runtime_manifests": runtimes,
        "checkpoint_metadata": {
            runtime_id: ready[runtime_id]["loaded_checkpoints"]
            for runtime_id in ("C", "H")
        },
        "cross_runtime_parity_sha256": file_sha256(RUNTIME_PARITY),
        "statistical_analysis": load_json(CANONICAL)["statistics"],
        "training_transitions": 0,
        "gradient_updates": 0,
        "validation_performance_episodes": 0,
        "paper_performance_episodes": 0,
    }
    atomic_json(EXECUTION_MANIFEST, execution_manifest)
    gates = {
        "A_checkpoint_sha_and_attempt_ids": all(
            row["passed"] for row in checkpoint_gate.values()
        ),
        "B_model_optimizer_source_provenance": all(
            metadata["native_training_transitions"] == 300_000
            and metadata["gradient_updates"] == 30_000
            for row in checkpoint_gate.values()
            for metadata in row["metadata"].values()
        ),
        "C_all_final_300k_checkpoints_native_loaded": all(
            row["passed"] for row in checkpoint_gate.values()
        ),
        "D_deterministic_actor_inference": all(
            metadata["deterministic_probe_sha256"]
            == canonical_sha256(metadata["deterministic_probe"])
            for row in checkpoint_gate.values()
            for metadata in row["metadata"].values()
        ),
        "E_production_identity_cross_runtime": production_comparison["passed"],
        "F_synthetic_physical_trace_parity": fixture_comparison["passed"],
        "G_task_fingerprints_cross_runtime": all(
            row["C_matches"] and row["H_matches"] and row["cross_runtime_exact"]
            for row in fingerprint_gate
        ),
        "H_schema_evidence_independent_verifier": all(schema.values()),
        "I_exactly_once_crash_mocks": all(crash.values()),
        "J_missing_completion_bound_semantics": schema[
            "missing_completion_bound_rejected"
        ]
        and schema["applicable_null_completion_bound_rejected"]
        and schema["not_applicable_fields_explicit"],
        "K_resource_projection": resources["runtime_gate_passed"]
        and resources["retained_gate_passed"],
        "route_clean_panel": split["route_clean_verified"],
        "namespace_unused_before_materialization": namespace["unused"],
        "v1_incident_immutable": incident["pair_counts"]
        == {"COMPLETED": 252, "RUNNING": 1, "NOT_STARTED": 647},
        "validation_unopened": split["validation"]["performance_episode_count"] == 0,
        "paper_sealed_uninspected": split["paper"]["content_fingerprints_constructed"]
        == 0,
    }
    preflight = {
        "schema_version": 1,
        "scope": (
            "synthetic/unreserved fixtures and pure Diagnostic task fingerprints only"
        ),
        "scientific_performance_episode_count": 0,
        "checkpoint_gate": checkpoint_gate,
        "cross_runtime_parity": parity,
        "task_fingerprint_gate": fingerprint_gate,
        "schema_gate": schema,
        "exactly_once_gate": crash,
        "resource_gate": resources,
        "gates": gates,
        "all_mandatory_gates_passed": all(gates.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": utc_now(),
    }
    atomic_json(PREFLIGHT_RESULTS, preflight)
    preparation = {
        "schema_version": 1,
        "study": "FEEDBACK_CHUNK_RUNTIME_ISOLATED_DIAGNOSTIC_V2",
        "all_mandatory_gates_passed": preflight["all_mandatory_gates_passed"],
        "preflight_results_sha256": file_sha256(PREFLIGHT_RESULTS),
        "execution_manifest_sha256": file_sha256(EXECUTION_MANIFEST),
        "pair_ledger_sha256_before_execution": file_sha256(PAIR_LEDGER),
        "pair_schedule_sha256": ledger["schedule_sha256"],
        "performance_episode_count": 0,
        "validation_performance_episode_count": 0,
        "paper_performance_episode_count": 0,
        "prepared_at_utc": utc_now(),
    }
    atomic_json(PREPARATION_STATUS, preparation)
    authorization = {
        "schema_version": 1,
        "study": "FEEDBACK_CHUNK_RUNTIME_ISOLATED_DIAGNOSTIC_V2",
        "preparation_lock": True,
        "diagnostic_v2_authorized": False,
        "training_authorized": False,
        "validation_authorized": False,
        "paper_authorized": False,
        "preparation_status_sha256": file_sha256(PREPARATION_STATUS),
        "pair_schedule_sha256": ledger["schedule_sha256"],
        "note": "unlock only after the frozen-protocol commit",
    }
    atomic_json(AUTHORIZATION, authorization)
    placeholder = {
        "schema_version": 1,
        "study": "FEEDBACK_CHUNK_RUNTIME_ISOLATED_DIAGNOSTIC_V2",
        "state": "NOT_EXECUTED",
        "pair_count": PAIR_COUNT,
        "performance_episode_count": 0,
    }
    atomic_json(RESULTS, placeholder)
    if not preflight["all_mandatory_gates_passed"]:
        raise RuntimeError(f"mandatory V2 preflight failed: {gates}")
    return preflight


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
