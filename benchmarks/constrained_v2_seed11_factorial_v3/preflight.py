"""Materialize every static and measured V3 pre-training gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_v2_repro_audit.harness import (
    current_track_fingerprints,
)

from .checkpointing import (
    capture_rng_states,
    load_verified,
    write_verified_checkpoint,
)
from .config import EXPECTED_STREAM_FINGERPRINTS, load_configuration
from .execution import (
    apply_collector_schedule,
    atomic_json,
    runtime_versions,
    source_hashes,
)
from .initialization import verify_initialization
from .resource_preflight import run_resource_preflight
from .split import (
    build_split_ledger,
    opened_split_rows,
    stream_rows,
    v2_failure_regression,
)

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
REPRO_AUDIT = REPOSITORY / "benchmarks/constrained_v2_repro_audit/results.json"


def _audit_current_prefix_match() -> bool:
    audit = json.loads(REPRO_AUDIT.read_text(encoding="utf-8"))
    expected = audit["rng_and_tracks"]["by_seed"]["11"][
        "current_first_100_used_track_fingerprints"
    ]
    return current_track_fingerprints(11, count=100) == expected


def _collision_evidence(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in stream_rows()["T_CURRENT"]
        if row["route_sha256"] == candidate["route_sha256"]
        and row["full_task_sha256"] == candidate["full_task_sha256"]
    ]


def _split_checks(ledger: dict[str, Any]) -> dict[str, bool]:
    accepted = ledger["accepted_tasks"]
    streams = stream_rows()
    opened = opened_split_rows()
    routes = {
        name: {row["route_sha256"] for row in rows}
        for name, rows in streams.items()
    }
    full = {
        name: {row["full_task_sha256"] for row in rows}
        for name, rows in streams.items()
    }
    opened_routes = {
        row["route_sha256"] for rows in opened.values() for row in rows
    }
    opened_full = {
        row["full_task_sha256"] for rows in opened.values() for row in rows
    }
    return {
        "accepted_count_18": len(accepted) == 18,
        "numeric_unique": len({row["numeric_seed"] for row in accepted}) == 18,
        "route_unique": len({row["route_sha256"] for row in accepted}) == 18,
        "full_task_unique": len({row["full_task_sha256"] for row in accepted}) == 18,
        "clean_against_both_stream_routes": all(
            row["route_sha256"] not in routes["T_HIST"]
            and row["route_sha256"] not in routes["T_CURRENT"]
            for row in accepted
        ),
        "clean_against_both_stream_full_tasks": all(
            row["full_task_sha256"] not in full["T_HIST"]
            and row["full_task_sha256"] not in full["T_CURRENT"]
            for row in accepted
        ),
        "clean_against_opened_routes": all(
            row["route_sha256"] not in opened_routes for row in accepted
        ),
        "clean_against_opened_full_tasks": all(
            row["full_task_sha256"] not in opened_full for row in accepted
        ),
        "paper_sealed_uninspected": (
            ledger["paper"]["state"] == "SEALED_UNINSPECTED"
            and ledger["paper"]["content_fingerprints_constructed"] == 0
        ),
    }


def _atomic_markdown(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _FixtureLag:
    def __init__(self):
        self.value = 0.0

    def get_lag(self):
        return self.value


class _FixturePolicy:
    def __init__(self):
        self.events: list[str] = []
        self.lag_optims = [_FixtureLag(), _FixtureLag()]

    def pre_update_fn(self, *, stats_train):
        self.events.append("PID")
        self.lag_optims[0].value += float(stats_train["cost"][0])
        self.lag_optims[1].value += float(stats_train["cost"][1])

    def update(self, _batch_size, _replay_buffer):
        self.events.append("RL")

    def post_update_fn(self, *, stats_train):
        del stats_train
        self.events.append("POST")


def _collector_semantics_self_check() -> bool:
    expected = [
        "TRANSITION_INSERTED",
        "PID_UPDATED",
        "RL_UPDATE",
        "POST_UPDATE_HOOK",
    ]
    for collector_name in ("C_HIST", "C_CURRENT"):
        policy = _FixturePolicy()
        counters = {"real_transitions": 10, "rl_gradient_updates": 0}
        result = apply_collector_schedule(
            collector_name=collector_name,
            completed_costs=np.asarray([1.0, 2.0]),
            policy=policy,
            replay_buffer=[None] * 256,
            counters=counters,
            target_updates=1,
            batch_size=256,
        )
        if result["events"] != expected or policy.events != ["PID", "RL", "POST"]:
            return False
    policy = _FixturePolicy()
    counters = {"real_transitions": 9, "rl_gradient_updates": 0}
    result = apply_collector_schedule(
        collector_name="C_HIST",
        completed_costs=None,
        policy=policy,
        replay_buffer=[None] * 256,
        counters=counters,
        target_updates=1,
        batch_size=256,
    )
    return result["events"] == ["TRANSITION_INSERTED"] and not policy.events


def _recovery_self_check(configuration) -> bool:
    payload = {
        "checkpoint_schema_version": 1,
        "condition": configuration.conditions[0].name,
        "attempt_id": "preflight-recovery-fixture",
        "configuration_sha256": configuration.configuration_sha256,
        "source_hashes": source_hashes(),
        "runtime_versions": runtime_versions(),
        "policy": {},
        "policy_optimizers": {},
        "lagrangian_states": [],
        "replay_buffer": [],
        "environment_state": {},
        "collector_state": {},
        "counters": {"real_transitions": 10_000, "rl_gradient_updates": 1_000},
        "episode_diagnostics": [],
        "diagnostic_checkpoints_completed": [],
        "rng_states": capture_rng_states(),
        "previous_completed_episodes": 5,
        "pid_update_count": 5,
        "current_episode_start_actor_sha256": "fixture",
        "current_episode_start_transition": 9_000,
        "walltime_state": {"completed_segments_s": 1.0},
    }
    with tempfile.TemporaryDirectory(prefix="longicontrol-v3-recovery-") as tmp:
        checkpoint = Path(tmp) / "checkpoint.pt"
        metadata = write_verified_checkpoint(checkpoint, payload)
        restored = load_verified(
            checkpoint,
            expected_sha256=metadata["sha256"],
            expected_configuration_sha256=configuration.configuration_sha256,
        )
    return restored["counters"] == payload["counters"]


def materialize_preparation() -> dict[str, Any]:
    configuration = load_configuration()
    ledger = build_split_ledger()
    atomic_json(HERE / "split_ledger.json", ledger)
    atomic_json(HERE / "stream_ledger.json", stream_rows())
    initialization = verify_initialization(configuration)
    atomic_json(HERE / "initialization.json", initialization)
    failed = v2_failure_regression()
    regression = {
        "candidate": failed,
        "current_stream_matches": _collision_evidence(failed),
        "passed": (
            failed["accepted"] is False
            and _collision_evidence(failed)
            == [stream_rows()["T_CURRENT"][559]]
        ),
    }
    atomic_json(HERE / "v2_regression.json", regression)
    split_checks = _split_checks(ledger)
    resource = run_resource_preflight()
    atomic_json(HERE / "resource_measurements.json", resource)
    collector_semantics = _collector_semantics_self_check()
    recovery = _recovery_self_check(configuration)
    evaluation_non_interference = (
        resource["diagnostic_evaluation_pilot"]["training_rng_unchanged"]
        and resource["diagnostic_evaluation_pilot"]["actor_unchanged"]
    )
    all_static = (
        ledger["stream_fingerprints"] == EXPECTED_STREAM_FINGERPRINTS
        and all(split_checks.values())
        and regression["passed"]
        and _audit_current_prefix_match()
    )
    status = {
        "schema_version": 1,
        "configuration_sha256": configuration.configuration_sha256,
        "runtime": runtime_versions(),
        "stream_fingerprints": ledger["stream_fingerprints"],
        "diagnostic_set_sha256": ledger["ordered_diagnostic_set_sha256"],
        "split_checks": split_checks,
        "v2_failure_regression_pass": regression["passed"],
        "current_first_100_matches_archive_audit": _audit_current_prefix_match(),
        "initialization_pass": initialization["passed"],
        "collector_semantics_pass": collector_semantics,
        "recovery_checkpoint_roundtrip_pass": recovery,
        "evaluation_non_interference_pass": evaluation_non_interference,
        "resource_gate_pass": resource["passed"],
        "paper": "SEALED_UNINSPECTED",
        "all_programmatic_preparation_gates_pass": (
            all_static
            and initialization["passed"]
            and collector_semantics
            and recovery
            and evaluation_non_interference
            and resource["passed"]
        ),
        "ready_for_main_training": (
            all_static
            and initialization["passed"]
            and collector_semantics
            and recovery
            and evaluation_non_interference
            and resource["passed"]
        ),
        "scientific_source_hashes": source_hashes(),
    }
    atomic_json(HERE / "preparation_status.json", status)
    rejected = "\n".join(
        f"- `{row['numeric_seed']}`: {', '.join(row['rejection_reasons'])}"
        for row in ledger["rejected_candidates"]
    )
    accepted = "\n".join(
        f"- `{row['numeric_seed']}` — route `{row['route_sha256']}`, full task "
        f"`{row['full_task_sha256']}`"
        for row in ledger["accepted_tasks"]
    )
    _atomic_markdown(
        HERE / "SPLIT_LEDGER.md",
        "# V3 route-clean split ledger\n\n"
        "Candidates were scanned in ascending order from `9100000`; the first 18 "
        "passing every frozen rule were accepted.\n\n"
        "## Rejected before the eighteenth acceptance\n\n"
        f"{rejected}\n\n## Accepted tasks\n\n{accepted}\n\n"
        f"Ordered diagnostic-set SHA-256: "
        f"`{ledger['ordered_diagnostic_set_sha256']}`.\n",
    )
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)
    if not args.materialize:
        raise ValueError("Use --materialize to run every V3 preparation gate")
    status = materialize_preparation()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ready_for_main_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
