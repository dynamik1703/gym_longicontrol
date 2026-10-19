"""Scientific and execution gates for the collector replication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.constrained_v2_collector_replication.checkpointing import (
    capture_rng_states,
    load_verified,
    write_verified_checkpoint,
)
from benchmarks.constrained_v2_collector_replication.config import (
    CONDITION_NAMES,
    TRAINING_SEEDS,
    load_configuration,
)
from benchmarks.constrained_v2_collector_replication.evaluation import (
    evaluate_legacy_or_paper,
)
from benchmarks.constrained_v2_collector_replication.execution import (
    apply_collector_schedule,
    runtime_versions,
    source_hashes,
)
from benchmarks.constrained_v2_collector_replication.finalize import _classification
from benchmarks.constrained_v2_collector_replication.initialization import (
    verify_initialization,
)
from benchmarks.constrained_v2_collector_replication.split import (
    build_split_ledger,
    diagnostic_seeds,
    reachable_horizon,
    validation_seeds,
)


def test_frozen_condition_order_and_reachable_horizon():
    assert CONDITION_NAMES == (
        "E11_episodic_seed11",
        "T11_transitional_seed11",
        "E29_episodic_seed29",
        "T29_transitional_seed29",
        "E47_episodic_seed47",
        "T47_transitional_seed47",
    )
    assert reachable_horizon() == {
        "real_transition_budget": 300_000,
        "minimum_physically_possible_episode_length": 347,
        "max_completed_episodes": 864,
        "max_exposed_tasks": 865,
    }


def test_evaluation_sets_are_frozen_unique_and_created_before_streams():
    ledger = build_split_ledger()
    assert ledger["construction_order"][:2] == [
        "COLLECTOR_DIAGNOSTIC_V1",
        "COLLECTOR_VALIDATION_V1",
    ]
    assert diagnostic_seeds() == tuple(range(9_200_000, 9_200_018))
    assert validation_seeds() == tuple(range(9_300_000, 9_300_018))
    rows = [
        row
        for split in ledger["evaluation_sets"].values()
        for row in split["accepted_tasks"]
    ]
    assert len(rows) == 36
    assert len({row["numeric_seed"] for row in rows}) == 36
    assert len({row["route_sha256"] for row in rows}) == 36
    assert len({row["full_task_sha256"] for row in rows}) == 36
    assert (
        ledger["evaluation_fingerprints"]
        == load_configuration().raw["expected_evaluation_fingerprints"]
    )


def test_all_three_canonical_streams_are_frozen_865_and_route_clean():
    ledger = build_split_ledger()
    assert (
        ledger["stream_fingerprints"]
        == load_configuration().raw["expected_stream_fingerprints"]
    )
    assert ledger["final_static_route_clean_gate"] == {"passed": True, "collisions": []}
    for seed in TRAINING_SEEDS:
        stream = ledger["canonical_streams"][str(seed)]
        assert len(stream["accepted_tasks"]) == 865
        assert all(
            row["accepted_task_index"] == index
            for index, row in enumerate(stream["accepted_tasks"])
        )
        assert stream["duplicates"]["route_repetitions"] >= 0


def test_paper_is_sealed_and_fail_closed():
    assert build_split_ledger()["paper"]["state"] == "SEALED_UNINSPECTED"
    with pytest.raises(PermissionError):
        evaluate_legacy_or_paper()


class _Lag:
    def __init__(self):
        self.value = 0.0

    def get_lag(self):
        return self.value


class _Policy:
    def __init__(self):
        self.events = []
        self.lag_optims = [_Lag(), _Lag()]

    def pre_update_fn(self, *, stats_train):
        self.events.append("PID")
        self.lag_optims[0].value += float(stats_train["cost"][0])
        self.lag_optims[1].value += float(stats_train["cost"][1])

    def update(self, batch_size, replay_buffer):
        assert batch_size == len(replay_buffer) == 256
        self.events.append("RL")

    def post_update_fn(self, *, stats_train):
        del stats_train
        self.events.append("POST")


def test_exact_collector_order_and_within_episode_difference():
    replay = [None] * 256
    for collector in ("EPISODIC", "TRANSITIONAL"):
        policy = _Policy()
        counters = {"real_transitions": 10, "rl_gradient_updates": 0}
        row = apply_collector_schedule(
            collector_name=collector,
            completed_costs=np.asarray([1.0, 2.0]),
            policy=policy,
            replay_buffer=replay,
            counters=counters,
            target_updates=1,
            batch_size=256,
        )
        assert row["events"] == [
            "TRANSITION_INSERTED",
            "PID_UPDATED",
            "RL_UPDATE",
            "POST_UPDATE_HOOK",
        ]
        assert policy.events == ["PID", "RL", "POST"]
    episodic = _Policy()
    e_counters = {"real_transitions": 9, "rl_gradient_updates": 0}
    e = apply_collector_schedule(
        collector_name="EPISODIC",
        completed_costs=None,
        policy=episodic,
        replay_buffer=replay,
        counters=e_counters,
        target_updates=1,
        batch_size=256,
    )
    transitional = _Policy()
    t_counters = {"real_transitions": 9, "rl_gradient_updates": 0}
    t = apply_collector_schedule(
        collector_name="TRANSITIONAL",
        completed_costs=None,
        policy=transitional,
        replay_buffer=replay,
        counters=t_counters,
        target_updates=1,
        batch_size=256,
    )
    assert e["events"] == ["TRANSITION_INSERTED"]
    assert t["events"] == ["TRANSITION_INSERTED", "RL_UPDATE"]


def test_all_seed_pairs_initialize_byte_identically():
    result = verify_initialization(load_configuration())
    assert result["passed"] is True
    assert result["all_pairs_byte_identical"] is True
    assert result["seeds_initialize_differently"] is True


def test_atomic_recovery_checkpoint_round_trip(tmp_path: Path):
    configuration = load_configuration()
    payload = {
        "checkpoint_schema_version": 1,
        "condition": configuration.conditions[0].name,
        "attempt_id": "fixture-attempt",
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
    checkpoint = tmp_path / "checkpoint.pt"
    metadata = write_verified_checkpoint(checkpoint, payload)
    restored = load_verified(
        checkpoint,
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256=configuration.configuration_sha256,
    )
    assert restored["attempt_id"] == "fixture-attempt"
    assert restored["counters"] == payload["counters"]


def _comparison(seed_deltas, pooled):
    return {
        "by_seed": {
            str(seed): {
                "episodic_minus_transitional": {"requirement_satisfaction": value}
            }
            for seed, value in zip(TRAINING_SEEDS, seed_deltas)
        },
        "pooled_out_of_54": {
            "episodic_minus_transitional": {"requirement_satisfaction": pooled}
        },
    }


@pytest.mark.parametrize(
    ("deltas", "pooled", "expected"),
    [
        ((2, 2, 2), 6, "EPISODIC_ROBUST_ADVANTAGE"),
        ((-2, -2, -2), -6, "TRANSITIONAL_ROBUST_ADVANTAGE"),
        ((1, 0, -1), 0, "NO_MATERIAL_DIFFERENCE"),
        ((3, -3, 0), 0, "SEED_HETEROGENEOUS"),
    ],
)
def test_frozen_classification(deltas, pooled, expected):
    assert _classification(_comparison(deltas, pooled))["label"] == expected
