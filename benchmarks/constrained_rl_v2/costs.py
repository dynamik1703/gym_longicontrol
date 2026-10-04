"""Physical V2 CMDP signals and benchmark-only Gymnasium wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import gymnasium as gym
import numpy as np

from benchmarks.constrained_rl.costs import objective_reward, speed_cost
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification
from gym_longicontrol.domain.track import Track

COST_NAMES = ("speed_integral_m", "deadline_deficit_integral_s")


def discounted_elapsed_cost(duration_s: float, dt_s: float, gamma: float) -> float:
    """Analytical discounted value of constant ``dt_s`` physical-time costs."""

    for name, value in (("duration_s", duration_s), ("dt_s", dt_s), ("gamma", gamma)):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
    if duration_s < 0 or dt_s <= 0 or not 0 <= gamma <= 1:
        raise ValueError("Invalid duration, timestep, or discount")
    steps = round(duration_s / dt_s)
    if not np.isclose(steps * dt_s, duration_s, atol=1e-12):
        raise ValueError("duration_s must be an integer number of timesteps")
    if gamma == 1:
        return float(steps * dt_s)
    return float(dt_s * (1 - gamma**steps) / (1 - gamma))


def optimistic_remaining_time_s(
    track: Track,
    *,
    position_m: float,
    track_length_m: float,
) -> float:
    """Lower-bound remaining time at the piecewise-constant speed limits.

    Acceleration, braking, comfort, and energy are intentionally ignored. The result is
    therefore optimistic and is used only to detect states already behind any possible
    speed-limit-respecting deadline schedule.
    """

    if not isfinite(position_m) or position_m < 0:
        raise ValueError("position_m must be finite and nonnegative")
    if not isfinite(track_length_m) or track_length_m <= track.positions_m[-1]:
        raise ValueError("track_length_m must exceed the last speed-limit position")
    position = min(float(position_m), float(track_length_m))
    ends = np.r_[track.positions_m[1:], float(track_length_m)]
    remaining = 0.0
    for start, end, limit in zip(track.positions_m, ends, track.limits_m_s):
        if position < end:
            remaining += (float(end) - max(position, float(start))) / float(limit)
    return float(remaining)


def deadline_state(
    *,
    elapsed_time_s: float,
    position_m: float,
    track: Track,
    track_length_m: float,
    deadline_s: float,
) -> tuple[float, float, float]:
    """Return optimistic remaining time, deadline slack, and nonnegative deficit."""

    if not isfinite(elapsed_time_s) or elapsed_time_s < 0:
        raise ValueError("elapsed_time_s must be finite and nonnegative")
    if not isfinite(deadline_s) or deadline_s <= 0:
        raise ValueError("deadline_s must be finite and positive")
    remaining = optimistic_remaining_time_s(
        track, position_m=position_m, track_length_m=track_length_m
    )
    slack = float(deadline_s - elapsed_time_s - remaining)
    return remaining, slack, max(0.0, -slack)


def deadline_deficit_cost(
    *, deficit_s: float, dt_s: float, normalization_s: float
) -> float:
    """Right-endpoint area under fractional negative deadline slack, in seconds."""

    for name, value in (
        ("deficit_s", deficit_s),
        ("dt_s", dt_s),
        ("normalization_s", normalization_s),
    ):
        if not isfinite(value):
            raise ValueError(f"{name} must be finite")
    if deficit_s < 0 or dt_s <= 0 or normalization_s <= 0:
        raise ValueError("Deadline cost inputs are outside their physical domain")
    return float(dt_s * deficit_s / normalization_s)


@dataclass(frozen=True)
class CompletedTrainingEpisode:
    objective_return: float
    costs: tuple[float, float]
    metrics: EpisodeMetrics
    simulator_steps: int
    budget_truncated: bool
    final_position_m: float
    final_deadline_deficit_s: float
    maximum_deadline_deficit_s: float


class DenseDeadlineTaskWrapper(gym.Wrapper):
    """Use energy reward plus speed and dense physical deadline constraints."""

    def __init__(
        self,
        environment: gym.Env,
        *,
        task: TaskSpecification,
        energy_scale_kwh: float,
        deadline_normalization_s: float,
        initial_seed: int | None = None,
        max_simulator_steps: int | None = None,
    ):
        super().__init__(environment)
        objective_reward(0.0, energy_scale_kwh)
        if deadline_normalization_s != task.max_time_s:
            raise ValueError("Deadline normalization must equal task.max_time_s")
        if initial_seed is not None and (
            isinstance(initial_seed, bool) or not isinstance(initial_seed, int)
        ):
            raise ValueError("initial_seed must be an integer or None")
        if max_simulator_steps is not None and (
            isinstance(max_simulator_steps, bool)
            or not isinstance(max_simulator_steps, int)
            or max_simulator_steps <= 0
        ):
            raise ValueError("max_simulator_steps must be positive or None")
        self.task = task
        self.energy_scale_kwh = float(energy_scale_kwh)
        self.deadline_normalization_s = float(deadline_normalization_s)
        self.initial_seed = initial_seed
        self.max_simulator_steps = max_simulator_steps
        self.simulator_steps = 0
        self.last_completed_episode: CompletedTrainingEpisode | None = None
        self._initial_seed_pending = initial_seed is not None
        self._last_elapsed_time_s = 0.0
        self._episode_objective_return = 0.0
        self._episode_costs = np.zeros(2, dtype=np.float64)
        self._episode_steps = 0
        self._maximum_deadline_deficit_s = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is None and self._initial_seed_pending:
            seed = self.initial_seed
        self._initial_seed_pending = False
        observation, info = self.env.reset(seed=seed, options=options)
        self._last_elapsed_time_s = float(info["elapsed_time_s"])
        self._episode_objective_return = 0.0
        self._episode_costs = np.zeros(2, dtype=np.float64)
        self._episode_steps = 0
        self._maximum_deadline_deficit_s = 0.0
        return observation, info

    def _deadline_state(self, info):
        base = self.env.unwrapped
        if base.track is None:
            raise RuntimeError("The physical track is unavailable")
        return deadline_state(
            elapsed_time_s=float(info["elapsed_time_s"]),
            position_m=float(info["position_m"]),
            track=base.track,
            track_length_m=float(base.config.track_length_m),
            deadline_s=self.task.max_time_s,
        )

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
        remaining_s, slack_s, deficit_s = self._deadline_state(info)
        step_costs = np.array(
            [
                speed_cost(float(info["speed_excess_m_s"]), dt_s),
                deadline_deficit_cost(
                    deficit_s=deficit_s,
                    dt_s=dt_s,
                    normalization_s=self.deadline_normalization_s,
                ),
            ],
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
        self._episode_objective_return += reward
        self._episode_costs += step_costs
        self._maximum_deadline_deficit_s = max(
            self._maximum_deadline_deficit_s, deficit_s
        )
        info.update(
            {
                "objective_reward": reward,
                "constraint_cost_names": COST_NAMES,
                "constraint_costs": {
                    name: float(value) for name, value in zip(COST_NAMES, step_costs)
                },
                "cost": step_costs.copy(),
                "deadline_optimistic_remaining_time_s": remaining_s,
                "deadline_slack_s": slack_s,
                "deadline_deficit_s": deficit_s,
                "deadline_deficit_cost_s": float(step_costs[1]),
                "deadline_deficit_integral_s": float(self._episode_costs[1]),
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
                final_position_m=float(info["position_m"]),
                final_deadline_deficit_s=deficit_s,
                maximum_deadline_deficit_s=self._maximum_deadline_deficit_s,
            )
        return observation, reward, bool(terminated), truncated, info

