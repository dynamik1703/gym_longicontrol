"""Benchmark-only Gymnasium Dict adapter for the canonical goal task."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .goal import (
    GOAL_SIZE,
    GoalScales,
    canonical_desired_goal,
    encode_achieved_goal,
    goal_success,
    goal_transition_reward,
)


class GoalConditionedTask(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Expose stored physical task state without changing public v1 envs."""

    def __init__(
        self,
        env: gym.Env,
        *,
        task: TaskSpecification,
        scales: GoalScales,
    ):
        gym.utils.RecordConstructorArgs.__init__(self, task=task, scales=scales)
        gym.Wrapper.__init__(self, env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("Goal adapter requires the public Box observation")
        base = env.unwrapped
        if base.config.track_length_m != scales.route_length_m:
            raise ValueError("Goal route scale differs from simulator route length")
        if base.vehicle.specs.velocity_limits[1] != scales.speed_scale_m_s:
            raise ValueError("Goal speed scale differs from vehicle limit")
        max_steps = getattr(env, "_max_episode_steps", None)
        if max_steps is None or max_steps * base.config.dt_s != scales.horizon_s:
            raise ValueError("Goal horizon differs from the finite TimeLimit")
        self.task = task
        self.scales = scales
        self._desired_goal = canonical_desired_goal(task, scales)
        self._previous_position_m = 0.0
        goal_space = spaces.Box(0.0, 1.0, shape=(GOAL_SIZE,), dtype=np.float64)
        self.observation_space = spaces.Dict(
            {
                "observation": env.observation_space,
                "achieved_goal": goal_space,
                "desired_goal": goal_space,
            }
        )

    def _achieved_goal(self, info) -> np.ndarray:
        return encode_achieved_goal(
            position_m=float(info["position_m"]),
            previous_position_m=self._previous_position_m,
            elapsed_time_s=float(info["elapsed_time_s"]),
            max_speed_violation_m_s=float(info["max_speed_violation_m_s"]),
            scales=self.scales,
        )

    def _observation(self, observation, info):
        return {
            "observation": np.asarray(observation, dtype=np.float64),
            "achieved_goal": self._achieved_goal(info),
            "desired_goal": self._desired_goal.copy(),
        }

    def reset(self, *, seed=None, options=None):
        observation, raw_info = self.env.reset(seed=seed, options=options)
        self._previous_position_m = float(raw_info["position_m"])
        info = dict(raw_info)
        info["goal_task_version"] = "first-arrival-v1"
        return self._observation(observation, info), info

    def step(self, action):
        previous_position = self._previous_position_m
        observation, historical_reward, terminated, truncated, raw_info = (
            self.env.step(action)
        )
        self._previous_position_m = previous_position
        goal_observation = self._observation(observation, raw_info)
        success = bool(
            goal_success(
                goal_observation["achieved_goal"],
                goal_observation["desired_goal"],
            )
        )
        reward = float(success)
        self._previous_position_m = float(raw_info["position_m"])

        if terminated or truncated:
            authoritative = is_feasible(
                EpisodeMetrics(**raw_info["episode_metrics"]), self.task
            )
            if success != authoritative:
                raise RuntimeError(
                    "Canonical goal success differs from the external evaluator"
                )

        info = dict(raw_info)
        info.update(
            {
                "historical_reward": float(historical_reward),
                "goal_success": success,
                "goal_task_version": "first-arrival-v1",
            }
        )
        return (
            goal_observation,
            reward,
            bool(terminated),
            bool(truncated),
            info,
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        """SB3-compatible pure reward API; ``info`` is deliberately unused."""

        del info
        return goal_transition_reward(achieved_goal, desired_goal)

    def compute_terminated(self, achieved_goal, desired_goal, info):
        """Return counterfactual first-arrival termination for goal tooling."""

        del info
        return goal_success(achieved_goal, desired_goal)
