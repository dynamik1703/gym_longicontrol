from __future__ import annotations

import math

import numpy as np

from benchmarks.slacktrader_v1.macro_transition import (
    GAMMA_DECISION,
    GAMMA_NATIVE,
    MacroAccumulator,
    cost_bellman_target,
    reward_bellman_target,
)
from benchmarks.slacktrader_v1.replay import MacroReplay
from benchmarks.slacktrader_v1.runner import (
    SlackTraderRunner,
    UpdateBudgetAccumulator,
    cumulative_update_target,
)


def _runner(**overrides) -> SlackTraderRunner:
    arguments = {
        "training_seed": 11,
        "attempt_id": "SLACKTRADER-RECOVERY-TEST",
        "configuration_sha256": "test-config",
        "source_hashes": {"planner": "test-source"},
        "native_budget": 30,
        "update_budget": 3,
        "warmup_native_transitions": 20,
        "batch_size": 1,
        "replay_capacity": 16,
    }
    arguments.update(overrides)
    return SlackTraderRunner(**arguments)


def test_correct_smdp_discount_targets_and_price_replay_action() -> None:
    accumulator = MacroAccumulator(np.zeros(16), 0.37)
    for index in range(10):
        accumulator.add_native_step(
            reward=float(index + 1),
            speed_cost=0.1,
            terminal_cost=0.0,
            intervention_cost=0.2,
        )
    transition = accumulator.finish(np.ones(16))
    assert math.isclose(transition.continuation_discount, GAMMA_DECISION)
    assert math.isclose(GAMMA_NATIVE**10, GAMMA_DECISION, abs_tol=1e-15)
    replay = MacroReplay(capacity=2)
    replay.append(transition)
    assert replay.action[0, 0] == np.float32(0.37)
    reward = np.array([[3.0]])
    discount = np.array([[GAMMA_NATIVE**4]])
    next_q = np.array([[5.0]])
    next_log_pi = np.array([[-0.7]])
    assert np.array_equal(
        reward_bellman_target(reward, discount, next_q, next_log_pi, 0.2),
        reward + discount * (next_q - 0.2 * next_log_pi),
    )
    assert np.array_equal(
        cost_bellman_target(np.array([[0.4]]), discount, next_q),
        np.array([[0.4]]) + discount * next_q,
    )


def test_revision_three_update_counts_are_exact() -> None:
    budget = UpdateBudgetAccumulator()
    assert budget.cumulative_target(10_000) == 0
    assert budget.cumulative_target(10_010) == 1
    assert budget.cumulative_target(300_000) == 30_000
    assert all(
        cumulative_update_target(step) == budget.cumulative_target(step)
        for step in (0, 9_999, 10_000, 10_010, 50_000, 299_999, 300_000)
    )


def test_exact_resume_inside_one_second_price_block(tmp_path) -> None:
    uninterrupted = _runner()
    checkpoint = tmp_path / "checkpoint.pkl"
    try:
        for _ in range(3):
            uninterrupted.advance_native()
        assert uninterrupted.macro_accumulator is not None
        assert uninterrupted.macro_accumulator.native_steps == 3
        metadata = uninterrupted.save(checkpoint)
        future_a = [uninterrupted.advance_native() for _ in range(12)]
        restored = SlackTraderRunner.restore(
            checkpoint,
            expected_sha256=metadata["sha256"],
            configuration_sha256="test-config",
            source_hashes={"planner": "test-source"},
            attempt_id="SLACKTRADER-RECOVERY-TEST",
        )
        try:
            future_b = [restored.advance_native() for _ in range(12)]
            assert future_a == future_b
            assert (
                uninterrupted.native_transition_count
                == restored.native_transition_count
            )
            assert uninterrupted.gradient_update_count == restored.gradient_update_count
            assert np.array_equal(uninterrupted.observation, restored.observation)
            assert uninterrupted.controller.current_price_kwh_per_s == (
                restored.controller.current_price_kwh_per_s
            )
            assert uninterrupted.controller._segment_actions == (
                restored.controller._segment_actions
            )
        finally:
            restored.close()
    finally:
        uninterrupted.close()
