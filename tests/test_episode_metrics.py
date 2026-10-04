import json
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pytest

from gym_longicontrol import DeterministicTrack, StochasticTrack
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.state import SimulationConfig
from gym_longicontrol.domain.task import TaskSpecification, is_feasible


@pytest.mark.parametrize("env_type", [DeterministicTrack, StochasticTrack])
def test_metrics_match_physical_info_and_are_reward_independent(env_type):
    first = env_type()
    second = env_type(reward_weights=(0, 0, 0, 0))
    try:
        first.reset(seed=42)
        second.reset(seed=42)
        count = 0
        was_above = False
        integral = maximum = 0.0
        for action in np.r_[np.ones(300), np.full(100, -1), np.ones(100)]:
            a, b = first.step([action]), second.step([action])
            np.testing.assert_array_equal(a[0], b[0])
            assert a[0].shape == (8,) and first.action_space.shape == (1,)
            assert isinstance(a[1], float) and b[1] == 0
            assert a[2:] == b[2:]  # Includes identical metrics and reward components.
            info = a[4]
            assert tuple(info["reward_components"]) == (
                "forward",
                "energy",
                "jerk",
                "shock",
            )
            assert a[1] == first.reward_weights @ list(
                info["reward_components"].values()
            )
            excess = max(0, info["velocity_m_s"] - info["speed_limit_m_s"])
            above = excess > 0
            count += above and not was_above
            was_above = above
            integral += excess * first.config.dt_s
            maximum = max(maximum, excess)
            assert info["speed_excess_m_s"] == excess
            metrics = EpisodeMetrics(**info["episode_metrics"])
            assert metrics == first.episode_metrics == second.episode_metrics
            assert (
                metrics.speed_violation_count == info["speed_violation_count"] == count
            )
            assert (
                metrics.max_speed_violation_m_s
                == info["max_speed_violation_m_s"]
                == maximum
            )
            assert (
                metrics.integrated_speed_violation_m
                == info["integrated_speed_violation_m"]
                == integral
            )
            assert metrics.travel_time_s == info["elapsed_time_s"]
            assert metrics.energy_kwh == info["total_energy_kwh"]
            assert metrics.completed == a[2]
            json.dumps(info, allow_nan=False)
            if a[2]:
                break
        assert count > 0  # Exercise violations, not just the all-zero path.
    finally:
        first.close()
        second.close()


def test_info_is_an_independent_snapshot_and_reset_clears_history():
    env = DeterministicTrack(speed_limit_positions=[0], speed_limits=[1])
    try:
        _, initial = env.reset(seed=1)
        for _ in range(30):
            _, _, _, _, info = env.step([1])
        assert info["speed_violation_count"] == 1
        assert info["max_speed_violation_m_s"] > 0
        before = env.episode_metrics
        info["episode_metrics"]["speed_violation_count"] = 999
        info["max_speed_violation_m_s"] = 999
        assert env.episode_metrics == before
        env.step([1])
        assert before != env.episode_metrics
        _, reset = env.reset(seed=1)
        assert reset == initial
        assert reset["episode_metrics"] == asdict(EpisodeMetrics(False, 0, 0, 0, 0, 0))
        assert reset["speed_excess_m_s"] == 0
        # Reading metrics repeatedly does not accumulate time or events.
        assert env.episode_metrics == env.episode_metrics
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["DeterministicTrack-v1", "StochasticTrack-v1"])
def test_external_time_limit_keeps_final_metrics_and_incomplete_route(env_id):
    env = gym.make(env_id, max_episode_steps=2)
    try:
        env.reset(seed=1)
        env.step([1])
        _, _, terminated, truncated, info = env.step([1])
        assert not terminated and truncated
        metrics = EpisodeMetrics(**info["episode_metrics"])
        assert metrics == env.unwrapped.episode_metrics
        assert metrics.travel_time_s == 0.2
        assert not metrics.completed
        assert not is_feasible(metrics, TaskSpecification(100, 100))
    finally:
        env.close()


@pytest.mark.parametrize("max_episode_steps", [1, 10])
def test_completion_and_simultaneous_time_limit(max_episode_steps):
    env = gym.make(
        "DeterministicTrack-v1",
        max_episode_steps=max_episode_steps,
        config=SimulationConfig(track_length_m=0.001),
        speed_limit_positions=[0],
        speed_limits=[30],
    )
    try:
        env.reset(seed=1)
        _, _, terminated, truncated, info = env.step([1])
        assert terminated and truncated == (max_episode_steps == 1)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        assert metrics.completed and metrics.travel_time_s == 0.1
        assert is_feasible(metrics, TaskSpecification(0.1))
        with pytest.raises(gym.error.ResetNeeded):
            env.step([1])
    finally:
        env.close()


def test_task_bounds_do_not_stop_the_environment_or_change_observations():
    env = DeterministicTrack()
    try:
        task = TaskSpecification(0.05)
        env.reset(seed=1)
        obs, _, terminated, truncated, info = env.step([1])
        assert obs.shape == (8,)
        assert not terminated and not truncated
        assert not is_feasible(EpisodeMetrics(**info["episode_metrics"]), task)
        before = env.episode_metrics
        with pytest.raises(ValueError):
            env.step([np.nan])
        assert env.episode_metrics == before
    finally:
        env.close()
