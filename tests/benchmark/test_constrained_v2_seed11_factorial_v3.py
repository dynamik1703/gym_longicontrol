"""Scientific and execution gates for the route-clean V3 factorial."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.constrained_v2_seed11_factorial_v3.checkpointing import (
    capture_rng_states,
    load_verified,
    write_verified_checkpoint,
)
from benchmarks.constrained_v2_seed11_factorial_v3.config import (
    EXPECTED_STREAM_FINGERPRINTS,
    load_configuration,
)
from benchmarks.constrained_v2_seed11_factorial_v3.evaluation import (
    evaluate_legacy_or_paper,
)
from benchmarks.constrained_v2_seed11_factorial_v3.execution import (
    apply_collector_schedule,
    runtime_versions,
    source_hashes,
)
from benchmarks.constrained_v2_seed11_factorial_v3.initialization import (
    EXPECTED,
    verify_initialization,
)
from benchmarks.constrained_v2_seed11_factorial_v3.split import (
    PAPER_SEEDS,
    _candidate,
    build_split_ledger,
    opened_split_rows,
    reachable_horizon,
    stream_fingerprints,
    stream_rows,
    v2_failure_regression,
)


def test_reachable_horizon_derives_865_with_active_task():
    assert reachable_horizon() == {
        "real_transition_budget": 300_000,
        "minimum_physically_possible_episode_length": 347,
        "max_completed_episodes": 864,
        "max_exposed_tasks": 865,
    }


def test_both_complete_reachable_stream_fingerprints_match_v2_evidence():
    rows = stream_rows()
    assert len(rows["T_HIST"]) == 865
    assert len(rows["T_CURRENT"]) == 865
    assert stream_fingerprints() == EXPECTED_STREAM_FINGERPRINTS


def test_v2_9000005_failure_is_detected_at_current_episode_559():
    failed = v2_failure_regression()
    assert failed["accepted"] is False
    assert "T_CURRENT_ROUTE" in failed["rejection_reasons"]
    assert "T_CURRENT_FULL_TASK" in failed["rejection_reasons"]
    matches = [
        row
        for row in stream_rows()["T_CURRENT"]
        if row["route_sha256"] == failed["route_sha256"]
        and row["full_task_sha256"] == failed["full_task_sha256"]
    ]
    assert matches == [
        {
            "episode_index": 559,
            "reset_seed": 1_141_367_263,
            "rng_draw_index": 559,
            "route_sha256": (
                "37421e43f27b4e72f5516f962be36e23c86f15be602f66ecd23151c561b6369c"
            ),
            "full_task_sha256": (
                "3a04942965fea88d7969485a38756d84bb473f0ee60d1757572557d31f094c6b"
            ),
        }
    ]


def test_diagnostic_selection_is_deterministic_and_records_all_rejections():
    ledger = build_split_ledger()
    assert [row["numeric_seed"] for row in ledger["accepted_tasks"]] == [
        9_100_000,
        9_100_001,
        9_100_003,
        9_100_004,
        9_100_005,
        9_100_006,
        9_100_007,
        9_100_008,
        9_100_009,
        9_100_010,
        9_100_011,
        9_100_012,
        9_100_013,
        9_100_014,
        9_100_015,
        9_100_016,
        9_100_018,
        9_100_019,
    ]
    assert [
        (row["numeric_seed"], row["rejection_reasons"])
        for row in ledger["rejected_candidates"]
    ] == [
        (9_100_002, ["T_HIST_ROUTE", "T_HIST_FULL_TASK"]),
        (
            9_100_017,
            [
                "T_HIST_ROUTE",
                "T_HIST_FULL_TASK",
                "T_CURRENT_ROUTE",
                "T_CURRENT_FULL_TASK",
            ],
        ),
    ]
    assert ledger["ordered_diagnostic_set_sha256"] == (
        "a44cc85367db4a21c5c69a8f209fd8628bf5e42b6ef3cace9bdf8a54df79583f"
    )


def test_all_diagnostic_tasks_are_clean_unique_and_opened_split_separated():
    accepted = build_split_ledger()["accepted_tasks"]
    assert len(accepted) == 18
    assert len({row["numeric_seed"] for row in accepted}) == 18
    assert len({row["route_sha256"] for row in accepted}) == 18
    assert len({row["full_task_sha256"] for row in accepted}) == 18
    for row in accepted:
        repeated = _candidate(
            row["numeric_seed"],
            accepted=[],
            streams=stream_rows(),
            opened=opened_split_rows(),
        )
        assert repeated["accepted"] is True
        assert repeated["rejection_reasons"] == []


def test_paper_is_fail_closed_and_only_numerically_excluded():
    ledger = build_split_ledger()
    assert ledger["paper"] == {
        "state": "SEALED_UNINSPECTED",
        "content_fingerprints_constructed": 0,
        "numeric_exclusion_only": list(PAPER_SEEDS),
    }
    with pytest.raises(RuntimeError, match="fail-closed"):
        _candidate(
            PAPER_SEEDS[0],
            accepted=[],
            streams=stream_rows(),
            opened=opened_split_rows(),
        )
    with pytest.raises(PermissionError):
        evaluate_legacy_or_paper()


class _Lag:
    def __init__(self):
        self.value = 0.0

    def get_lag(self):
        return self.value


class _FakePolicy:
    def __init__(self):
        self.events = []
        self.lag_optims = [_Lag(), _Lag()]

    def pre_update_fn(self, *, stats_train):
        assert np.asarray(stats_train["cost"]).shape == (2,)
        self.events.append("PID")
        self.lag_optims[0].value += 1.0
        self.lag_optims[1].value += 2.0

    def update(self, batch_size, replay_buffer):
        assert batch_size == 256 and len(replay_buffer) >= batch_size
        self.events.append("RL")

    def post_update_fn(self, *, stats_train):
        assert np.asarray(stats_train["cost"]).shape == (2,)
        self.events.append("POST")


def test_collector_order_and_pid_semantics():
    replay = [None] * 256
    for collector_name in ("C_HIST", "C_CURRENT"):
        policy = _FakePolicy()
        counters = {"real_transitions": 10, "rl_gradient_updates": 0}
        result = apply_collector_schedule(
            collector_name=collector_name,
            completed_costs=np.asarray([3.0, 4.0]),
            policy=policy,
            replay_buffer=replay,
            counters=counters,
            target_updates=1,
            batch_size=256,
        )
        assert result["events"] == [
            "TRANSITION_INSERTED",
            "PID_UPDATED",
            "RL_UPDATE",
            "POST_UPDATE_HOOK",
        ]
        assert policy.events == ["PID", "RL", "POST"]
        assert result["lambda_before"] == (0.0, 0.0)
        assert result["lambda_after"] == (1.0, 2.0)
    historical = _FakePolicy()
    counters = {"real_transitions": 9, "rl_gradient_updates": 0}
    result = apply_collector_schedule(
        collector_name="C_HIST",
        completed_costs=None,
        policy=historical,
        replay_buffer=replay,
        counters=counters,
        target_updates=1,
        batch_size=256,
    )
    assert result["events"] == ["TRANSITION_INSERTED"]
    assert counters["rl_gradient_updates"] == 0


def test_exact_300k_to_30k_update_budgets_for_both_collectors():
    replay = [None] * 256
    current_policy = _FakePolicy()
    current = {"real_transitions": 300_000, "rl_gradient_updates": 0}
    for target in range(1, 30_001):
        apply_collector_schedule(
            collector_name="C_CURRENT",
            completed_costs=None,
            policy=current_policy,
            replay_buffer=replay,
            counters=current,
            target_updates=target,
            batch_size=256,
        )
    historical_policy = _FakePolicy()
    historical = {"real_transitions": 300_000, "rl_gradient_updates": 0}
    endpoints = list(range(1800, 300_000, 1800)) + [300_000]
    for endpoint in endpoints:
        apply_collector_schedule(
            collector_name="C_HIST",
            completed_costs=np.asarray([0.0, 0.0]),
            policy=historical_policy,
            replay_buffer=replay,
            counters=historical,
            target_updates=endpoint // 10,
            batch_size=256,
        )
    assert current["rl_gradient_updates"] == 30_000
    assert historical["rl_gradient_updates"] == 30_000


def test_all_four_learner_initializations_are_identical_and_expected():
    result = verify_initialization(load_configuration())
    assert result["passed"] is True
    assert result["all_identical"] is True
    for row in result["by_condition"].values():
        for key, value in EXPECTED.items():
            assert row[key] == value
        assert row["alpha"]["effective_alpha"] == 1.0
        assert row["pid_multipliers"] == [0.0, 0.0]


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
    sidecar = json.loads(
        checkpoint.with_name("checkpoint.pt.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["sha256"] == metadata["sha256"]
