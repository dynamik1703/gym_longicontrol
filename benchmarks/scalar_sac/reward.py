"""Interpretable scalar reward adapter for the sensitivity experiment."""

from __future__ import annotations

from math import isfinite

import gymnasium as gym

from gym_longicontrol.domain.task import TaskSpecification

from .config import RewardParameters


class ScalarBenchmarkReward(gym.Wrapper):
    """Replace only the training reward, leaving the environment trajectory intact.

    The four entries in ``benchmark_reward_components`` are already weighted.
    Their sum is the returned scalar reward. The wrapped historical reward is
    retained in ``info["historical_reward"]`` for debugging only.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        parameters: RewardParameters,
        task: TaskSpecification,
        energy_normalization_kwh: float,
        speed_violation_normalization_m: float,
    ):
        super().__init__(env)
        if energy_normalization_kwh <= 0 or not isfinite(energy_normalization_kwh):
            raise ValueError("energy_normalization_kwh must be finite and positive")
        if (
            speed_violation_normalization_m <= 0
            or not isfinite(speed_violation_normalization_m)
        ):
            raise ValueError(
                "speed_violation_normalization_m must be finite and positive"
            )
        self.parameters = parameters
        self.task = task
        self.energy_normalization_kwh = float(energy_normalization_kwh)
        self.speed_violation_normalization_m = float(
            speed_violation_normalization_m
        )
        self._previous_position_m = 0.0
        self._previous_time_s = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous_position_m = float(info["position_m"])
        self._previous_time_s = float(info["elapsed_time_s"])
        return observation, info

    def step(self, action):
        observation, historical_reward, terminated, truncated, info = self.env.step(
            action
        )
        position_m = float(info["position_m"])
        elapsed_time_s = float(info["elapsed_time_s"])
        distance_m = position_m - self._previous_position_m
        duration_s = elapsed_time_s - self._previous_time_s
        if distance_m < 0 or duration_s <= 0:
            raise ValueError("Benchmark reward requires forward, positive-time steps")

        track_length_m = float(self.env.unwrapped.config.track_length_m)
        parameters = self.parameters
        components = {
            "progress": parameters.progress_weight * distance_m / track_length_m,
            "energy": -parameters.energy_weight
            * float(info["step_energy_kwh"])
            / self.energy_normalization_kwh,
            "time": -parameters.time_weight * duration_s / self.task.max_time_s,
            "speed_violation": -parameters.speed_violation_weight
            * float(info["speed_excess_m_s"])
            * duration_s
            / self.speed_violation_normalization_m,
        }
        reward = float(sum(components.values()))
        if not isfinite(reward):
            raise ValueError("Benchmark reward must remain finite")

        self._previous_position_m = position_m
        self._previous_time_s = elapsed_time_s
        benchmark_info = dict(info)
        benchmark_info["historical_reward"] = float(historical_reward)
        benchmark_info["benchmark_reward_components"] = components
        benchmark_info["benchmark_reward_configuration_id"] = (
            parameters.configuration_id
        )
        return observation, reward, terminated, truncated, benchmark_info
