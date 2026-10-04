"""Interpretable scalar reward designed after the V1 scale failure."""

from __future__ import annotations

from math import isfinite

import gymnasium as gym

from .v2_config import V2RewardParameters


def approximate_episode_return(
    parameters: V2RewardParameters,
    *,
    completed: bool,
    on_time: bool,
    progress_fraction: float,
    travel_time_s: float,
    energy_kwh: float,
    integrated_speed_violation_m: float,
    had_speed_violation: bool,
    max_time_s: float = 140.0,
    energy_normalization_kwh: float = 0.25,
    speed_violation_normalization_m: float = 1.0,
) -> float:
    """Return the undiscounted episode-scale value used in design checks."""

    route = parameters.progress_weight * progress_fraction
    if completed and on_time:
        route += parameters.on_time_completion_bonus
    return float(
        route
        - parameters.energy_weight * energy_kwh / energy_normalization_kwh
        - parameters.time_weight * travel_time_s / max_time_s
        - parameters.speed_integral_weight
        * integrated_speed_violation_m
        / speed_violation_normalization_m
        - (
            parameters.speed_violation_event_penalty
            if had_speed_violation
            else 0.0
        )
    )


class ScalarBenchmarkRewardV2(gym.Wrapper):
    """Add terminal task signals while preserving dynamics and evaluation.

    There are four conceptual components: route achievement (dense progress and
    an on-time completion bonus), energy, elapsed time, and speed violation
    (dense integrated excess and a terminal any-violation penalty).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        parameters: V2RewardParameters,
        max_time_s: float,
        max_speed_violation_m_s: float,
        energy_normalization_kwh: float,
        speed_violation_normalization_m: float,
    ):
        super().__init__(env)
        for name, value in (
            ("max_time_s", max_time_s),
            ("energy_normalization_kwh", energy_normalization_kwh),
            (
                "speed_violation_normalization_m",
                speed_violation_normalization_m,
            ),
        ):
            if value <= 0 or not isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
        if max_speed_violation_m_s < 0 or not isfinite(max_speed_violation_m_s):
            raise ValueError(
                "max_speed_violation_m_s must be finite and nonnegative"
            )
        self.parameters = parameters
        self.max_time_s = float(max_time_s)
        self.max_speed_violation_m_s = float(max_speed_violation_m_s)
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
            raise ValueError("V2 reward requires forward, positive-time steps")

        parameters = self.parameters
        episode_finished = bool(terminated or truncated)
        metrics = info["episode_metrics"]
        on_time_completion = bool(
            episode_finished
            and metrics["completed"]
            and metrics["travel_time_s"] <= self.max_time_s
        )
        any_speed_violation = bool(
            episode_finished
            and metrics["max_speed_violation_m_s"]
            > self.max_speed_violation_m_s
        )
        route_achievement = (
            parameters.progress_weight
            * distance_m
            / float(self.env.unwrapped.config.track_length_m)
        )
        if on_time_completion:
            route_achievement += parameters.on_time_completion_bonus
        speed_violation = (
            -parameters.speed_integral_weight
            * float(info["speed_excess_m_s"])
            * duration_s
            / self.speed_violation_normalization_m
        )
        if any_speed_violation:
            speed_violation -= parameters.speed_violation_event_penalty
        components = {
            "route_achievement": route_achievement,
            "energy": -parameters.energy_weight
            * float(info["step_energy_kwh"])
            / self.energy_normalization_kwh,
            "time": -parameters.time_weight * duration_s / self.max_time_s,
            "speed_violation": speed_violation,
        }
        reward = float(sum(components.values()))
        if not isfinite(reward):
            raise ValueError("V2 benchmark reward must remain finite")

        self._previous_position_m = position_m
        self._previous_time_s = elapsed_time_s
        benchmark_info = dict(info)
        benchmark_info["historical_reward"] = float(historical_reward)
        benchmark_info["benchmark_reward_components"] = components
        benchmark_info["benchmark_reward_configuration_id"] = (
            parameters.configuration_id
        )
        benchmark_info["benchmark_reward_formula_version"] = "scalar-v2"
        return observation, reward, terminated, truncated, benchmark_info
