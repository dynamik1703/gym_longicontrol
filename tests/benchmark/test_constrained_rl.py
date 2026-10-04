from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

import gym_longicontrol  # noqa: F401
from benchmarks.constrained_rl.adapter import split_reward_and_cost_metrics
from benchmarks.constrained_rl.config import (
    configuration_sha256,
    load_configuration,
)
from benchmarks.constrained_rl.costs import (
    COST_NAMES,
    ConstrainedTaskWrapper,
    objective_reward,
    speed_cost,
    terminal_task_cost,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification


class ScriptedPhysicalEnvironment(gym.Env):
    def __init__(self, steps):
        self.steps = tuple(steps)
        self.index = 0
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (8,), dtype=np.float64)

    @staticmethod
    def _metrics(*, completed=False, time=0.0, energy=0.0, integral=0.0):
        return asdict(
            EpisodeMetrics(
                completed=completed,
                travel_time_s=time,
                energy_kwh=energy,
                speed_violation_count=int(integral > 0),
                max_speed_violation_m_s=float(integral > 0),
                integrated_speed_violation_m=integral,
            )
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        return np.zeros(8), {
            "elapsed_time_s": 0.0,
            "episode_metrics": self._metrics(),
        }

    def step(self, action):
        del action
        item = self.steps[self.index]
        self.index += 1
        energy = sum(step[0] for step in self.steps[: self.index])
        integral = sum(step[1] * step[2] for step in self.steps[: self.index])
        terminated, truncated = item[3], item[4]
        return (
            np.full(8, self.index / 10),
            123.0,
            terminated,
            truncated,
            {
                "elapsed_time_s": sum(step[2] for step in self.steps[: self.index]),
                "step_energy_kwh": item[0],
                "speed_excess_m_s": item[1],
                "episode_metrics": self._metrics(
                    completed=terminated,
                    time=sum(step[2] for step in self.steps[: self.index]),
                    energy=energy,
                    integral=integral,
                ),
            },
        )


def test_objective_reward_is_only_fixed_scaled_negative_net_energy():
    assert objective_reward(0.05, 0.25) == pytest.approx(-0.2)
    assert objective_reward(-0.05, 0.25) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        objective_reward(0.0, 0.0)


def test_speed_cost_matches_integrated_violation_contribution():
    assert speed_cost(2.5, 0.1) == pytest.approx(0.25)
    assert speed_cost(0.0, 0.1) == 0.0
    with pytest.raises(ValueError):
        speed_cost(-0.1, 0.1)


@pytest.mark.parametrize(
    ("completed", "time_s", "expected"),
    [(True, 139.9, 0.0), (True, 140.0, 0.0), (True, 140.1, 1.0), (False, 1.0, 1.0)],
)
def test_terminal_task_cost_is_completion_and_deadline_only(
    completed, time_s, expected
):
    metrics = EpisodeMetrics(completed, time_s, 1.0, 4, 9.0, 12.0)
    assert terminal_task_cost(metrics, TaskSpecification(140.0)) == expected


def test_wrapper_emits_ordered_separate_costs_and_terminal_failure():
    base = ScriptedPhysicalEnvironment(
        [
            (0.01, 2.0, 0.1, False, False),
            (-0.005, 3.0, 0.1, False, True),
        ]
    )
    wrapped = ConstrainedTaskWrapper(
        base, task=TaskSpecification(140.0), energy_scale_kwh=0.25
    )
    wrapped.reset(seed=2000)
    _, reward_1, terminated_1, truncated_1, info_1 = wrapped.step([0.0])
    _, reward_2, terminated_2, truncated_2, info_2 = wrapped.step([0.0])
    assert reward_1 == pytest.approx(-0.04)
    assert reward_2 == pytest.approx(0.02)
    assert not terminated_1 and not truncated_1
    assert not terminated_2 and truncated_2
    assert info_1["constraint_cost_names"] == COST_NAMES
    np.testing.assert_allclose(info_1["cost"], [0.2, 0.0])
    np.testing.assert_allclose(info_2["cost"], [0.3, 1.0])
    assert wrapped.last_completed_episode is not None
    assert wrapped.last_completed_episode.costs == pytest.approx((0.5, 1.0))
    assert wrapped.last_completed_episode.objective_return == pytest.approx(-0.02)


def test_budget_truncation_is_exact_and_not_labeled_successful():
    base = ScriptedPhysicalEnvironment(
        [(0.0, 0.0, 0.1, False, False)] * 3
    )
    wrapped = ConstrainedTaskWrapper(
        base,
        task=TaskSpecification(140.0),
        energy_scale_kwh=0.25,
        max_simulator_steps=2,
    )
    wrapped.reset()
    wrapped.step([0.0])
    _, _, terminated, truncated, info = wrapped.step([0.0])
    assert not terminated and truncated
    assert info["training_budget_truncated"] is True
    np.testing.assert_allclose(info["cost"], [0.0, 1.0])
    assert wrapped.simulator_steps == 2
    with pytest.raises(gym.error.ResetNeeded):
        wrapped.step([0.0])


def test_costs_are_deterministic_and_have_no_split_metadata_input():
    steps = [(0.01, 1.0, 0.1, False, True)]
    left = ConstrainedTaskWrapper(
        ScriptedPhysicalEnvironment(steps),
        task=TaskSpecification(140.0),
        energy_scale_kwh=0.25,
    )
    right = ConstrainedTaskWrapper(
        ScriptedPhysicalEnvironment(steps),
        task=TaskSpecification(140.0),
        energy_scale_kwh=0.25,
    )
    left.reset(seed=2000)
    right.reset(seed=3000)
    left_result = left.step([0.2])
    right_result = right.step([0.2])
    assert left_result[1] == right_result[1]
    np.testing.assert_array_equal(left_result[4]["cost"], right_result[4]["cost"])


def test_vector_cost_adapter_preserves_frozen_column_order():
    batch = SimpleNamespace(
        rew=np.array([1.0, 2.0]),
        info={"cost": np.array([[0.1, 1.0], [0.2, 0.0]])},
    )
    reward, speed, task = split_reward_and_cost_metrics(batch, 2)
    np.testing.assert_array_equal(reward, [1.0, 2.0])
    np.testing.assert_array_equal(speed, [0.1, 0.2])
    np.testing.assert_array_equal(task, [1.0, 0.0])
    with pytest.raises(ValueError):
        split_reward_and_cost_metrics(
            SimpleNamespace(rew=np.array([1.0]), info={"cost": np.array([0.1])}), 2
        )


def test_wrapper_preserves_real_environment_spaces_and_physics():
    original = gym.make("StochasticTrack-v1", max_episode_steps=10)
    wrapped_base = gym.make("StochasticTrack-v1", max_episode_steps=10)
    wrapped = ConstrainedTaskWrapper(
        wrapped_base,
        task=TaskSpecification(140.0),
        energy_scale_kwh=0.25,
    )
    try:
        original_observation, _ = original.reset(seed=123)
        wrapped_observation, _ = wrapped.reset(seed=123)
        assert wrapped.observation_space == original.observation_space
        assert wrapped.action_space == original.action_space
        np.testing.assert_array_equal(wrapped_observation, original_observation)
        for action in (np.array([0.5]), np.array([-0.25]), np.array([0.0])):
            original_step = original.step(action)
            wrapped_step = wrapped.step(action)
            np.testing.assert_allclose(wrapped_step[0], original_step[0])
            assert wrapped_step[2:4] == original_step[2:4]
            for key in (
                "position_m",
                "velocity_m_s",
                "elapsed_time_s",
                "total_energy_kwh",
                "reward_components",
                "episode_metrics",
            ):
                if isinstance(original_step[4][key], dict):
                    assert wrapped_step[4][key] == original_step[4][key]
                else:
                    assert wrapped_step[4][key] == pytest.approx(original_step[4][key])
    finally:
        original.close()
        wrapped.close()


def test_canonical_protocol_is_frozen_to_native_300k_study():
    configuration = load_configuration()
    assert configuration.training_seeds == (11, 29, 47)
    assert configuration.simulator_step_budget == 300_000
    assert configuration.algorithm.action_repeat == 1
    assert configuration.constraints.names == COST_NAMES
    assert configuration.constraints.cost_limits == (0.0, 0.0)
    assert set(configuration.track_splits.paper_final_test_reserved) == set(
        range(4000, 4018)
    )
    assert len(configuration_sha256(configuration)) == 64
    with pytest.raises(ValueError):
        replace(
            configuration.constraints,
            names=("task_failure", "speed_integral_m"),
        )
