"""Terminal-only binary reward using the authoritative physical evaluator."""

from __future__ import annotations

import gymnasium as gym

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible


def binary_success(metrics: EpisodeMetrics, task: TaskSpecification) -> bool:
    """Return the exact external benchmark feasibility decision."""

    return is_feasible(metrics, task)


class BinarySuccessReward(gym.Wrapper):
    """Replace historical reward with one terminal success bit."""

    def __init__(self, environment: gym.Env, *, task: TaskSpecification):
        super().__init__(environment)
        self.task = task

    def step(self, action):
        observation, historical_reward, terminated, truncated, raw_info = self.env.step(
            action
        )
        episode_finished = bool(terminated or truncated)
        success = False
        if episode_finished:
            success = binary_success(
                EpisodeMetrics(**raw_info["episode_metrics"]), self.task
            )
        reward = float(success)
        info = dict(raw_info)
        info.update(
            {
                "historical_reward": float(historical_reward),
                "binary_reward_success": success,
                "binary_reward_outcome_known": episode_finished,
                "binary_reward_formula_version": "terminal-success-v1",
            }
        )
        return observation, reward, bool(terminated), bool(truncated), info
