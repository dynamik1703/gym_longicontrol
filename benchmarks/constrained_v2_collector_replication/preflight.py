"""Materialize every static and measured pre-training hard gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .checkpointing import capture_rng_states, load_verified, write_verified_checkpoint
from .config import TRAINING_SEEDS, load_configuration
from .execution import (
    apply_collector_schedule,
    atomic_json,
    runtime_versions,
    source_hashes,
)
from .initialization import verify_initialization
from .resource_preflight import run_resource_preflight
from .split import build_split_ledger, opened_split_rows

HERE = Path(__file__).resolve().parent


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


def _collector_semantics_self_check() -> dict[str, Any]:
    terminal_results = {}
    for collector in ("EPISODIC", "TRANSITIONAL"):
        policy = _FixturePolicy()
        counters = {"real_transitions": 10, "rl_gradient_updates": 0}
        terminal_results[collector] = apply_collector_schedule(
            collector_name=collector,
            completed_costs=np.asarray([1.0, 2.0]),
            policy=policy,
            replay_buffer=[None] * 256,
            counters=counters,
            target_updates=1,
            batch_size=256,
        )
        if policy.events != ["PID", "RL", "POST"]:
            return {
                "passed": False,
                "reason": f"terminal order changed for {collector}",
            }
    episodic_policy = _FixturePolicy()
    episodic_counters = {"real_transitions": 9, "rl_gradient_updates": 0}
    episodic = apply_collector_schedule(
        collector_name="EPISODIC",
        completed_costs=None,
        policy=episodic_policy,
        replay_buffer=[None] * 256,
        counters=episodic_counters,
        target_updates=1,
        batch_size=256,
    )
    transitional_policy = _FixturePolicy()
    transitional_counters = {"real_transitions": 9, "rl_gradient_updates": 0}
    transitional = apply_collector_schedule(
        collector_name="TRANSITIONAL",
        completed_costs=None,
        policy=transitional_policy,
        replay_buffer=[None] * 256,
        counters=transitional_counters,
        target_updates=1,
        batch_size=256,
    )
    passed = (
        episodic["events"] == ["TRANSITION_INSERTED"]
        and not episodic_policy.events
        and transitional["events"] == ["TRANSITION_INSERTED", "RL_UPDATE"]
        and transitional_policy.events == ["RL"]
        and all(
            row["events"]
            == ["TRANSITION_INSERTED", "PID_UPDATED", "RL_UPDATE", "POST_UPDATE_HOOK"]
            for row in terminal_results.values()
        )
    )
    return {
        "passed": passed,
        "episodic_nonterminal_events": episodic["events"],
        "transitional_nonterminal_events": transitional["events"],
        "terminal_events": {
            name: row["events"] for name, row in terminal_results.items()
        },
    }


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
        "current_episode_update_events": [],
        "walltime_state": {"completed_segments_s": 1.0},
    }
    with tempfile.TemporaryDirectory(
        prefix="longicontrol-collector-recovery-"
    ) as temporary:
        checkpoint = Path(temporary) / "checkpoint.pt"
        metadata = write_verified_checkpoint(checkpoint, payload)
        restored = load_verified(
            checkpoint,
            expected_sha256=metadata["sha256"],
            expected_configuration_sha256=configuration.configuration_sha256,
        )
    return (
        restored["counters"] == payload["counters"]
        and restored["attempt_id"] == payload["attempt_id"]
    )


def _split_checks(ledger: dict[str, Any]) -> dict[str, bool]:
    diagnostic = ledger["evaluation_sets"]["diagnostic"]["accepted_tasks"]
    validation = ledger["evaluation_sets"]["validation"]["accepted_tasks"]
    combined = diagnostic + validation
    opened = [row for rows in opened_split_rows().values() for row in rows]
    opened_routes = {row["route_sha256"] for row in opened}
    opened_full = {row["full_task_sha256"] for row in opened}
    return {
        "both_sets_have_18": len(diagnostic) == len(validation) == 18,
        "cross_set_numeric_unique": len({row["numeric_seed"] for row in combined})
        == 36,
        "cross_set_route_unique": len({row["route_sha256"] for row in combined}) == 36,
        "cross_set_full_task_unique": len({row["full_task_sha256"] for row in combined})
        == 36,
        "opened_route_separation": all(
            row["route_sha256"] not in opened_routes for row in combined
        ),
        "opened_full_task_separation": all(
            row["full_task_sha256"] not in opened_full for row in combined
        ),
        "three_streams_have_865": all(
            len(ledger["canonical_streams"][str(seed)]["accepted_tasks"]) == 865
            for seed in TRAINING_SEEDS
        ),
        "final_static_route_clean_gate": ledger["final_static_route_clean_gate"][
            "passed"
        ],
        "paper_sealed_uninspected": ledger["paper"]["state"] == "SEALED_UNINSPECTED"
        and ledger["paper"]["content_fingerprints_constructed"] == 0,
    }


def _write_reports(ledger: dict[str, Any], resource: dict[str, Any]) -> None:
    evaluation_lines = []
    for name, split in ledger["evaluation_sets"].items():
        evaluation_lines.append(f"## {split['id']}\n")
        evaluation_lines.extend(
            f"- `{row['numeric_seed']}` — route `{row['route_sha256']}`, "
            f"full `{row['full_task_sha256']}`"
            for row in split["accepted_tasks"]
        )
        evaluation_lines.append(f"\nOrdered SHA-256: `{split['ordered_sha256']}`.\n")
    stream_lines = [
        f"- `CANONICAL_{seed}`: "
        f"`{ledger['stream_fingerprints'][str(seed)]}`; duplicates "
        f"`{ledger['canonical_streams'][str(seed)]['duplicates']}`"
        for seed in TRAINING_SEEDS
    ]
    _atomic_markdown(
        HERE / "SPLIT_LEDGER.md",
        "# Collector replication split ledger\n\n"
        "Evaluation sets were frozen before canonical training streams. "
        "Paper remained numerically excluded and uninspected.\n\n"
        + "\n".join(evaluation_lines)
        + "\n## Canonical streams\n\n"
        + "\n".join(stream_lines)
        + "\n\nFinal static route-clean gate: `PASS`.\n",
    )
    projection = resource["projection"]
    pilots = resource["collector_pilots"]
    _atomic_markdown(
        HERE / "RESOURCE_REPORT.md",
        "# Resource preflight\n\n"
        f"Hard gate: `{'PASS' if resource['passed'] else 'FAIL'}`.\n\n"
        f"- Projected total serial walltime: "
        f"{projection['total_serial_walltime_h']:.3f} h (limit 18 h).\n"
        f"- Projected retained disk: "
        f"{projection['retained_disk_gb']:.3f} GB (limit 15 GB).\n"
        f"- EPISODIC pilot: {pilots['EPISODIC']['walltime_s']:.3f} s.\n"
        f"- TRANSITIONAL pilot: {pilots['TRANSITIONAL']['walltime_s']:.3f} s.\n"
        "- Fixture performance was discarded; no evaluation-set or canonical-"
        "stream task was used.\n",
    )


def materialize_preparation() -> dict[str, Any]:
    configuration = load_configuration()
    ledger = build_split_ledger()
    atomic_json(HERE / "split_ledger.json", ledger)
    atomic_json(HERE / "stream_ledger.json", ledger["canonical_streams"])
    initialization = verify_initialization(configuration)
    atomic_json(HERE / "initialization.json", initialization)
    collector_semantics = _collector_semantics_self_check()
    recovery = _recovery_self_check(configuration)
    split_checks = _split_checks(ledger)
    resource = run_resource_preflight()
    atomic_json(HERE / "resource_measurements.json", resource)
    evaluation_non_interference = (
        resource["diagnostic_evaluation_pilot"]["training_rng_unchanged"]
        and resource["diagnostic_evaluation_pilot"]["actor_unchanged"]
    )
    ready = (
        all(split_checks.values())
        and initialization["passed"]
        and collector_semantics["passed"]
        and recovery
        and evaluation_non_interference
        and resource["passed"]
    )
    status = {
        "schema_version": 1,
        "configuration_sha256": configuration.configuration_sha256,
        "runtime": runtime_versions(),
        "construction_order": ledger["construction_order"],
        "evaluation_fingerprints": ledger["evaluation_fingerprints"],
        "stream_fingerprints": ledger["stream_fingerprints"],
        "split_checks": split_checks,
        "initialization_pass": initialization["passed"],
        "collector_semantics": collector_semantics,
        "recovery_checkpoint_roundtrip_pass": recovery,
        "evaluation_non_interference_pass": evaluation_non_interference,
        "resource_gate_pass": resource["passed"],
        "paper": "SEALED_UNINSPECTED",
        "all_programmatic_preparation_gates_pass": ready,
        "ready_for_main_training": ready,
        "scientific_source_hashes": source_hashes(),
    }
    atomic_json(HERE / "preparation_status.json", status)
    atomic_json(
        HERE / "freeze.json",
        {
            "schema_version": 1,
            "configuration_sha256": configuration.configuration_sha256,
            "source_commit": configuration.raw["source_commit"],
            "condition_order": [
                condition.name for condition in configuration.conditions
            ],
            "evaluation_fingerprints": ledger["evaluation_fingerprints"],
            "stream_fingerprints": ledger["stream_fingerprints"],
            "paired_initialization_hashes": {
                seed: row["combined_learner_and_replay_sha256"]
                for seed, row in initialization["by_seed_pair"].items()
            },
            "scientific_source_hashes": status["scientific_source_hashes"],
            "paper": "SEALED_UNINSPECTED",
        },
    )
    _write_reports(ledger, resource)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)
    if not args.materialize:
        raise ValueError("Use --materialize to run every preparation gate")
    status = materialize_preparation()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ready_for_main_training"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
