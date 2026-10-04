"""Typed, deliberately small API exposed to generated reward functions."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any

import gymnasium as gym
import numpy as np

from gym_longicontrol.domain.task import TaskSpecification

COMPONENT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


@dataclass(frozen=True)
class RewardContext:
    """Whitelisted physical values for one post-transition reward decision."""

    position_m: float
    previous_position_m: float
    delta_position_m: float
    velocity_m_s: float
    previous_velocity_m_s: float
    acceleration_m_s2: float
    previous_acceleration_m_s2: float
    action: float
    speed_limit_m_s: float
    future_speed_limit_1_m_s: float
    future_speed_limit_2_m_s: float
    distance_to_future_limit_1_m: float
    distance_to_future_limit_2_m: float
    step_energy_kwh: float
    cumulative_energy_kwh: float
    elapsed_time_s: float
    dt_s: float
    route_length_m: float
    time_budget_s: float
    completed: bool
    episode_ended: bool

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"completed", "episode_ended"}:
                if not isinstance(value, bool):
                    raise ValueError(f"{field.name} must be boolean")
            else:
                object.__setattr__(self, field.name, _finite_number(field.name, value))
        if self.dt_s <= 0 or self.route_length_m <= 0 or self.time_budget_s <= 0:
            raise ValueError("dt, route length and time budget must be positive")
        if not -1.0 <= self.action <= 1.0:
            raise ValueError("action must lie in [-1, 1]")
        if (
            min(
                self.position_m,
                self.previous_position_m,
                self.velocity_m_s,
                self.previous_velocity_m_s,
                self.speed_limit_m_s,
                self.future_speed_limit_1_m_s,
                self.future_speed_limit_2_m_s,
                self.distance_to_future_limit_1_m,
                self.distance_to_future_limit_2_m,
                self.elapsed_time_s,
            )
            < 0
        ):
            raise ValueError(
                "distance, time, position, velocity and limits are nonnegative"
            )


REWARD_CONTEXT_FIELDS = tuple(field.name for field in fields(RewardContext))


@dataclass(frozen=True)
class RewardOutput:
    reward: float
    components: Mapping[str, float]

    def __post_init__(self):
        reward = _finite_number("reward", self.reward)
        if not isinstance(self.components, Mapping):
            raise ValueError("components must be a mapping")
        normalized = {}
        for name, value in self.components.items():
            if not isinstance(name, str) or COMPONENT_NAME.fullmatch(name) is None:
                raise ValueError(f"Invalid reward component name: {name!r}")
            normalized[name] = _finite_number(f"component {name}", value)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "components", MappingProxyType(normalized))


RewardFunction = Callable[[RewardContext], RewardOutput]


class CandidateRewardWrapper(gym.Wrapper):
    """Replace training reward without changing physics, observations or endings."""

    def __init__(
        self,
        environment: gym.Env,
        *,
        compute_reward: RewardFunction,
        task: TaskSpecification,
        candidate_id: str,
        source_sha256: str,
    ):
        super().__init__(environment)
        self.compute_reward = compute_reward
        self.task = task
        self.candidate_id = candidate_id
        self.source_sha256 = source_sha256
        self._previous_info: dict[str, Any] | None = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._previous_info = dict(info)
        return observation, info

    def _reward_context(self, action: Any, info: Mapping[str, Any], ended: bool):
        if self._previous_info is None:
            raise RuntimeError("reset() must be called before step()")
        action_value = np.asarray(action, dtype=np.float64)
        if action_value.shape != (1,) or not np.isfinite(action_value).all():
            raise ValueError("action must be a finite one-element array")
        base = self.env.unwrapped
        sensor = base.track.sense(
            float(info["position_m"]), float(base.config.sensor_range_m)
        )
        return RewardContext(
            position_m=info["position_m"],
            previous_position_m=self._previous_info["position_m"],
            delta_position_m=(
                float(info["position_m"]) - float(self._previous_info["position_m"])
            ),
            velocity_m_s=info["velocity_m_s"],
            previous_velocity_m_s=self._previous_info["velocity_m_s"],
            acceleration_m_s2=info["acceleration_m_s2"],
            previous_acceleration_m_s2=self._previous_info["acceleration_m_s2"],
            action=float(action_value[0]),
            speed_limit_m_s=sensor.current_limit_m_s,
            future_speed_limit_1_m_s=sensor.future_limits_m_s[0],
            future_speed_limit_2_m_s=sensor.future_limits_m_s[1],
            distance_to_future_limit_1_m=sensor.distances_m[0],
            distance_to_future_limit_2_m=sensor.distances_m[1],
            step_energy_kwh=info["step_energy_kwh"],
            cumulative_energy_kwh=info["total_energy_kwh"],
            elapsed_time_s=info["elapsed_time_s"],
            dt_s=base.config.dt_s,
            route_length_m=base.config.track_length_m,
            time_budget_s=self.task.max_time_s,
            completed=bool(info["episode_metrics"]["completed"]),
            episode_ended=ended,
        )

    def step(self, action):
        observation, historical_reward, terminated, truncated, raw_info = self.env.step(
            action
        )
        ended = bool(terminated or truncated)
        context = self._reward_context(action, raw_info, ended)
        output = self.compute_reward(context)
        if not isinstance(output, RewardOutput):
            raise TypeError("compute_reward must return RewardOutput")
        info = dict(raw_info)
        info.update(
            {
                "historical_reward": float(historical_reward),
                "llm_reward_candidate_id": self.candidate_id,
                "llm_reward_source_sha256": self.source_sha256,
                "llm_reward_components": dict(output.components),
            }
        )
        self._previous_info = dict(raw_info)
        return observation, output.reward, bool(terminated), bool(truncated), info
