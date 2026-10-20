from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.hybrid_rl_execution.evaluation import (
    evaluate_seed,
    evaluate_validation,
    paired_three_arm_summary,
    verify_frozen_splits_without_opening,
)
from benchmarks.hybrid_rl_execution.execution import (
    RunLease,
    prepare_launchd_job,
    require_main_training_authorized,
)
from benchmarks.hybrid_rl_execution.runner import (
    HybridRunner,
    UpdateBudgetAccumulator,
    cumulative_update_target,
)
from benchmarks.hybrid_rl_execution.verification import exact_resume_probe


def _runner(**overrides):
    arguments = {
        "training_seed": 11,
        "attempt_id": "RUNNER-TEST",
        "configuration_sha256": "test",
        "source_hashes": {"test": "test"},
        "native_budget": 30,
        "update_budget": 3,
        "warmup_native_transitions": 20,
        "batch_size": 1,
        "replay_capacity": 16,
    }
    arguments.update(overrides)
    return HybridRunner(**arguments)


def test_strategy_action_is_sampled_once_and_held_ten_native_steps():
    runner = _runner()
    records = [runner.advance_native() for _ in range(10)]
    try:
        assert runner.actor_sample_count == 1
        assert runner.macro_transition_count == 1
        assert len(runner.replay) == 1
        assert len({row["strategy_action"] for row in records}) == 1
        assert runner.replay.action[0, 0] == np.float32(records[0]["strategy_action"])
    finally:
        runner.close()


def test_replay_is_not_inserted_and_actor_not_resampled_mid_block():
    runner = _runner()
    try:
        for _ in range(9):
            runner.advance_native()
        assert len(runner.replay) == 0
        assert runner.macro_transition_count == 0
        assert runner.actor_sample_count == 1
        assert runner.remaining_native_steps_in_block == 1
    finally:
        runner.close()


def test_intervention_deviation_compares_two_native_actuator_commands():
    runner = _runner()
    try:
        record = runner.advance_native()
        shield = record["shield"]
        assert record["intervention_deviation"] == abs(
            shield["proposed_native_action"] - shield["executed_action"]
        )
        assert -1.0 <= shield["proposed_native_action"] <= 1.0
        assert -1.0 <= shield["executed_action"] <= 1.0
    finally:
        runner.close()


def test_exact_native_and_gradient_update_budget_accounting():
    budget = UpdateBudgetAccumulator()
    assert budget.cumulative_target(9_999) == 0
    assert budget.cumulative_target(10_000) == 0
    assert budget.cumulative_target(10_009) == 0
    assert budget.cumulative_target(10_010) == 1
    assert budget.cumulative_target(50_000) == 4_137
    assert budget.cumulative_target(100_000) == 9_310
    assert budget.cumulative_target(150_000) == 14_482
    assert budget.cumulative_target(200_000) == 19_655
    assert budget.cumulative_target(250_000) == 24_827
    assert budget.cumulative_target(299_999) == 29_999
    assert budget.cumulative_target(300_000) == 30_000
    assert all(
        cumulative_update_target(native_transitions)
        == budget.cumulative_target(native_transitions)
        for native_transitions in range(300_001)
    )
    runner = _runner(warmup_native_transitions=10)
    try:
        runner.run_to_budget()
        assert runner.native_transition_count == 30
        assert runner.gradient_update_count == 3
    finally:
        runner.close()


def test_first_update_occurs_at_exact_post_warmup_native_target():
    runner = _runner(
        native_budget=40,
        update_budget=4,
        warmup_native_transitions=10,
    )
    try:
        for _ in range(17):
            runner.advance_native()
        assert runner.gradient_update_count == 0
        assert runner.macro_accumulator is not None
        runner.advance_native()
        assert runner.native_transition_count == 18
        assert runner.gradient_update_count == 1
        assert runner.macro_accumulator is not None
    finally:
        runner.close()


def test_exact_resume_inside_partially_executed_block():
    result = exact_resume_probe()
    assert result["partial_block_at_checkpoint"]["native_steps_in_accumulator"] == 3
    assert result["partial_block_at_checkpoint"]["remaining_native_steps_in_block"] == 7
    assert result["update_counts"] == {
        "at_checkpoint": 0,
        "uninterrupted_after_checkpoint": 1,
        "restored_after_checkpoint": 1,
    }
    assert result["preparation_interactions"]["gradient_updates"] == 2
    assert result["future_action_reward_cost_trace_exact"]
    assert result["network_optimizer_replay_environment_controller_rng_exact"]
    assert result["passed"]


def test_frozen_split_hashes_and_fail_closed_boundaries():
    splits = verify_frozen_splits_without_opening()
    assert splits["diagnostic"]["count"] == 18
    assert splits["validation"]["count"] == 36
    assert splits["validation"]["performance_episodes"] == 0
    assert splits["paper"]["state"] == "SEALED_UNINSPECTED"
    with pytest.raises(PermissionError, match="PAPER_TEST_SEALED"):
        evaluate_seed(numeric_seed=4000, arm="causal")
    with pytest.raises(PermissionError, match="frozen and unopened"):
        evaluate_validation(None)  # type: ignore[arg-type]
    with pytest.raises(PermissionError, match="not authorized"):
        require_main_training_authorized()


def test_final_authorization_flags_are_fail_closed():
    from benchmarks.hybrid_rl_execution.evaluation import CANONICAL

    authorization = json.loads(CANONICAL.read_text(encoding="utf-8"))["authorization"]
    assert authorization == {
        "ready_for_main_training": True,
        "main_training_authorized": False,
        "main_training_enabled": False,
        "validation_authorized": False,
        "paper_authorized": False,
        "preparation_only": True,
    }


def test_three_arm_pairing_is_feasible_only_and_thresholds_are_frozen():
    def arm(feasible, energy):
        return {"feasible": feasible, "energy_kwh": energy}

    outcomes = [
        {
            "causal": arm(True, 10.0),
            "time_aware": arm(True, 9.0),
            "hybrid_rl": arm(True, 8.0),
        },
        {
            "causal": arm(True, 10.0),
            "time_aware": arm(True, 9.0),
            "hybrid_rl": arm(False, 1.0),
        },
    ]
    summary = paired_three_arm_summary(outcomes)
    assert summary["hybrid_rsr"] == 1
    assert summary["paired_feasible_count_vs_causal"] == 1
    assert summary["paired_feasible_count_vs_nonlearning"] == 1
    assert summary["mean_reduction_vs_causal_percent"] == 20.0
    assert summary["mean_reduction_vs_nonlearning_percent"] > 11.0


def test_launchd_is_prepared_not_launched_and_duplicate_lease_is_blocked(tmp_path):
    plist = prepare_launchd_job(tmp_path, tmp_path / "run", "python")
    launcher = json.loads((tmp_path / "run/launcher.json").read_text())
    assert plist.exists()
    assert launcher["state"] == "PREPARED_NOT_LAUNCHED"
    assert not launcher["main_training_authorized"]
    with RunLease(tmp_path / "run"):
        with pytest.raises(RuntimeError, match="duplicate"):
            with RunLease(tmp_path / "run"):
                pass
