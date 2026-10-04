"""Physical CMDP signals and the benchmark-only Gymnasium wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import gymnasium as gym
import numpy as np

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification

COST_NAMES = ("speed_integral_m", "task_failure")


def objective_reward(step_energy_kwh: float, energy_scale_kwh: float) -> float:
    """Return normalized negative signed net energy, with no task shaping."""

    if not isfinite(step_energy_kwh):
        raise ValueError("step_energy_kwh must be finite")
    if not isfinite(energy_scale_kwh) or energy_scale_kwh <= 0:
        raise ValueError("energy_scale_kwh must be finite and positive")
    return -float(step_energy_kwh) / float(energy_scale_kwh)


def speed_cost(speed_excess_m_s: float, dt_s: float) -> float:
    """Right-endpoint integral contribution ``speed_excess_m_s * dt_s``."""

    if not isfinite(speed_excess_m_s) or speed_excess_m_s < 0:
        raise ValueError("speed_excess_m_s must be finite and nonnegative")
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    return float(speed_excess_m_s) * float(dt_s)


def terminal_task_cost(metrics: EpisodeMetrics, task: TaskSpecification) -> float:
    """Binary completion/deadline cost, intentionally independent of speed."""

    return float(not (metrics.completed and metrics.travel_time_s <= task.max_time_s))


@dataclass(frozen=True)
class CompletedTrainingEpisode:
    objective_return: float
    costs: tuple[float, float]
    metrics: EpisodeMetrics
    simulator_steps: int
    budget_truncated: bool


class ConstrainedTaskWrapper(gym.Wrapper):
    """Replace reward with energy and expose two ordered FSRL cost signals.

    The wrapped simulator still performs exactly one native transition per call.
    This wrapper neither changes observations/actions nor reads split or validation
    metadata. ``max_simulator_steps`` is a lifetime training budget; it truncates only
    the last training episode so the physical interaction count is exact.
    """

    def __init__(
        self,
        environment: gym.Env,
        *,
        task: TaskSpecification,
        energy_scale_kwh: float,
        initial_seed: int | None = None,
        max_simulator_steps: int | None = None,
    ):
        super().__init__(environment)
        objective_reward(0.0, energy_scale_kwh)
        if initial_seed is not None and (
            isinstance(initial_seed, bool) or not isinstance(initial_seed, int)
        ):
            raise ValueError("initial_seed must be an integer or None")
        if max_simulator_steps is not None and (
            isinstance(max_simulator_steps, bool)
            or not isinstance(max_simulator_steps, int)
            or max_simulator_steps <= 0
        ):
            raise ValueError("max_simulator_steps must be a positive integer or None")
        self.task = task
        self.energy_scale_kwh = float(energy_scale_kwh)
        self.initial_seed = initial_seed
        self.max_simulator_steps = max_simulator_steps
        self.simulator_steps = 0
        self.last_completed_episode: CompletedTrainingEpisode | None = None
        self._initial_seed_pending = initial_seed is not None
        self._last_elapsed_time_s = 0.0
        self._episode_objective_return = 0.0
        self._episode_costs = np.zeros(2, dtype=np.float64)
        self._episode_steps = 0

    def reset(self, *, seed=None, options=None):
        if seed is None and self._initial_seed_pending:
            seed = self.initial_seed
        self._initial_seed_pending = False
        observation, info = self.env.reset(seed=seed, options=options)
        self._last_elapsed_time_s = float(info["elapsed_time_s"])
        self._episode_objective_return = 0.0
        self._episode_costs = np.zeros(2, dtype=np.float64)
        self._episode_steps = 0
        return observation, info

    def step(self, action):
        if (
            self.max_simulator_steps is not None
            and self.simulator_steps >= self.max_simulator_steps
        ):
            raise gym.error.ResetNeeded("The lifetime simulator budget is exhausted")
        observation, _historical_reward, terminated, truncated, raw_info = (
            self.env.step(action)
        )
        info = dict(raw_info)
        elapsed_time_s = float(info["elapsed_time_s"])
        dt_s = elapsed_time_s - self._last_elapsed_time_s
        self._last_elapsed_time_s = elapsed_time_s
        reward = objective_reward(
            float(info["step_energy_kwh"]), self.energy_scale_kwh
        )
        step_costs = np.array(
            [speed_cost(float(info["speed_excess_m_s"]), dt_s), 0.0],
            dtype=np.float64,
        )
        self.simulator_steps += 1
        self._episode_steps += 1
        budget_truncated = bool(
            self.max_simulator_steps is not None
            and self.simulator_steps == self.max_simulator_steps
            and not (terminated or truncated)
        )
        truncated = bool(truncated or budget_truncated)
        if terminated or truncated:
            metrics = EpisodeMetrics(**info["episode_metrics"])
            step_costs[1] = terminal_task_cost(metrics, self.task)
        self._episode_objective_return += reward
        self._episode_costs += step_costs
        info.update(
            {
                "objective_reward": reward,
                "constraint_cost_names": COST_NAMES,
                "constraint_costs": {
                    name: float(value) for name, value in zip(COST_NAMES, step_costs)
                },
                "cost": step_costs.copy(),
                "training_budget_truncated": budget_truncated,
                "simulator_steps_total": self.simulator_steps,
            }
        )
        if terminated or truncated:
            self.last_completed_episode = CompletedTrainingEpisode(
                objective_return=self._episode_objective_return,
                costs=tuple(float(value) for value in self._episode_costs),
                metrics=EpisodeMetrics(**info["episode_metrics"]),
                simulator_steps=self._episode_steps,
                budget_truncated=budget_truncated,
            )
        return observation, reward, bool(terminated), truncated, info
