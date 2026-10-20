from __future__ import annotations

import math

import numpy as np

from benchmarks.hybrid_rl_execution.macro_transition import (
    GAMMA_DECISION,
    GAMMA_NATIVE,
    MacroAccumulator,
    cost_bellman_target,
    reward_bellman_target,
)
from benchmarks.hybrid_rl_execution.replay import MacroReplay


def test_full_block_discount_and_discounted_signals():
    accumulator = MacroAccumulator(np.zeros(16), 0.4)
    for index in range(10):
        accumulator.add_native_step(
            reward=float(index + 1),
            speed_cost=0.1,
            terminal_cost=0.0,
            intervention_cost=0.2,
        )
    transition = accumulator.finish(np.ones(16))
    expected_reward = sum(GAMMA_NATIVE**index * (index + 1) for index in range(10))
    assert math.isclose(GAMMA_NATIVE**10, GAMMA_DECISION, abs_tol=1e-15)
    assert math.isclose(transition.reward, expected_reward, abs_tol=1e-12)
    assert math.isclose(transition.continuation_discount, GAMMA_DECISION, abs_tol=1e-15)
    assert transition.native_steps == 10


def test_short_terminal_block_has_actual_duration_and_no_bootstrap():
    accumulator = MacroAccumulator(np.zeros(16), -0.7)
    for index in range(4):
        accumulator.add_native_step(
            reward=1.0,
            speed_cost=0.0,
            terminal_cost=float(index == 3),
            intervention_cost=0.0,
            terminated=index == 3,
        )
    transition = accumulator.finish(np.zeros(16))
    assert transition.native_steps == 4
    assert transition.terminated
    assert not transition.truncated
    assert transition.continuation_discount == 0.0
    assert transition.terminal_cost == GAMMA_NATIVE**3


def test_truncation_has_no_terminal_bootstrap_or_implicit_failure_cost():
    accumulator = MacroAccumulator(np.zeros(16), 0.0)
    for index in range(3):
        accumulator.add_native_step(
            reward=0.0,
            speed_cost=0.0,
            terminal_cost=0.0,
            intervention_cost=0.0,
            truncated=index == 2,
        )
    transition = accumulator.finish(np.zeros(16))
    assert transition.truncated
    assert transition.terminal_cost == 0.0
    assert transition.continuation_discount == 0.0


def test_hand_calculated_reward_and_cost_targets():
    reward = np.array([[3.0], [-1.0]])
    discount = np.array([[GAMMA_NATIVE**4], [0.0]])
    next_q = np.array([[5.0], [99.0]])
    next_log_pi = np.array([[-0.7], [-1.2]])
    observed_reward = reward_bellman_target(reward, discount, next_q, next_log_pi, 0.2)
    expected_reward = reward + discount * (next_q - 0.2 * next_log_pi)
    assert np.array_equal(observed_reward, expected_reward)
    cost = np.array([[0.4], [2.0]])
    observed_cost = cost_bellman_target(cost, discount, next_q)
    assert np.array_equal(observed_cost, cost + discount * next_q)
    assert observed_reward[1, 0] == reward[1, 0]
    assert observed_cost[1, 0] == cost[1, 0]


def test_replay_stores_proposed_strategy_action_at_macro_boundary():
    accumulator = MacroAccumulator(np.zeros(16), 0.37)
    for _ in range(10):
        accumulator.add_native_step(
            reward=0.0,
            speed_cost=0.0,
            terminal_cost=0.0,
            intervention_cost=0.0,
        )
    replay = MacroReplay(capacity=4)
    replay.append(accumulator.finish(np.ones(16)))
    batch = replay.sample(1, np.random.default_rng(1))
    assert batch["action"][0, 0] == np.float32(0.37)
    assert batch["native_steps"][0, 0] == 10
