from __future__ import annotations

import math

import numpy as np
import torch

from benchmarks.feedback_chunk_rl_v1.macro_transition import (
    GAMMA_NATIVE,
    MacroAccumulator,
    cost_bellman_target,
    discounted_sum,
    reward_bellman_target,
)
from benchmarks.feedback_chunk_rl_v1.sac_agent import ConstrainedResidualSAC
from benchmarks.feedback_chunk_rl_v1.training import cumulative_update_target


def test_independent_discount_and_targets_for_ten_and_thirty_steps() -> None:
    for action, duration in ((np.array([0.25]), 10), (np.array([0.1, 0.2, 0.3]), 30)):
        rewards = np.linspace(-0.4, 0.7, duration)
        costs = np.linspace(0.0, 0.05, duration)
        accumulator = MacroAccumulator(np.zeros(16), action, duration)
        for reward, cost in zip(rewards, costs, strict=True):
            accumulator.add_native_step(
                reward=float(reward),
                speed_cost=float(cost),
                terminal_cost=0.0,
                intervention_cost=float(cost * 2.0),
            )
        transition = accumulator.finish(np.ones(16))
        independent_reward = sum(
            (0.99 ** (index / 10.0)) * value for index, value in enumerate(rewards)
        )
        assert math.isclose(transition.reward, independent_reward, abs_tol=1e-14)
        assert math.isclose(transition.reward, discounted_sum(rewards), abs_tol=1e-14)
        assert math.isclose(
            transition.continuation_discount,
            0.99 ** (duration / 10.0),
            abs_tol=1e-14,
        )
        reward_target = reward_bellman_target(
            np.array([[transition.reward]]),
            np.array([[transition.continuation_discount]]),
            np.array([[2.0]]),
            np.array([[-0.5]]),
            0.2,
        )
        assert np.allclose(
            reward_target,
            transition.reward + transition.continuation_discount * 2.1,
        )
        cost_target = cost_bellman_target(
            np.array([[transition.speed_cost]]),
            np.array([[transition.continuation_discount]]),
            np.array([[3.0]]),
        )
        assert np.allclose(
            cost_target,
            transition.speed_cost + transition.continuation_discount * 3.0,
        )
    assert math.isclose(GAMMA_NATIVE**10, 0.99, abs_tol=1e-14)
    assert math.isclose(GAMMA_NATIVE**30, 0.99**3, abs_tol=1e-14)


def test_terminal_and_dataset_truncations_do_not_bootstrap() -> None:
    for field in ("terminated", "environment_truncated", "dataset_truncated"):
        accumulator = MacroAccumulator(np.zeros(16), np.zeros(3), 30)
        kwargs = {field: True}
        accumulator.add_native_step(
            reward=1.0,
            speed_cost=0.0,
            terminal_cost=0.0,
            intervention_cost=0.0,
            **kwargs,
        )
        transition = accumulator.finish(np.zeros(16))
        assert transition.native_steps == 1
        assert transition.continuation_discount == 0.0


def test_revision_three_update_schedule_is_exact() -> None:
    assert cumulative_update_target(10_000) == 0
    assert cumulative_update_target(10_009) == 0
    assert cumulative_update_target(10_010) == 1
    assert cumulative_update_target(300_000) == 30_000


def test_actor_dimensions_initialization_and_update_are_finite() -> None:
    rng = np.random.default_rng(7)
    for method, dim in (("RESIDUAL1S_SAC", 1), ("CHUNK3_RESIDUAL_SAC", 3)):
        agent = ConstrainedResidualSAC(method, seed=11)
        observation = np.zeros(16, dtype=np.float64)
        deterministic = agent.act(observation, deterministic=True)
        assert deterministic.shape == (dim,)
        assert np.max(np.abs(deterministic)) < 0.01
        generator = torch.Generator().manual_seed(123)
        stochastic = agent.act(observation, deterministic=False, generator=generator)
        assert stochastic.shape == (dim,)
        assert np.isfinite(stochastic).all()
        batch = {
            "observation": rng.normal(size=(8, 16)).astype(np.float32),
            "action": rng.uniform(-1, 1, size=(8, dim)).astype(np.float32),
            "reward": rng.normal(size=(8, 1)).astype(np.float32),
            "speed_cost": np.zeros((8, 1), dtype=np.float32),
            "terminal_cost": np.zeros((8, 1), dtype=np.float32),
            "intervention_cost": rng.uniform(0, 0.1, size=(8, 1)).astype(np.float32),
            "next_observation": rng.normal(size=(8, 16)).astype(np.float32),
            "continuation_discount": np.full((8, 1), 0.99**dim, dtype=np.float32),
        }
        diagnostics = agent.update(batch, generator=torch.Generator().manual_seed(456))
        assert all(np.isfinite(value) for value in diagnostics.values())
        assert agent.update_count == 1
