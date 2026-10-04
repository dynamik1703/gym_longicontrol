"""Benchmark-only requirement observation and parameterized V2 costs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from benchmarks.constrained_rl.costs import objective_reward, speed_cost
from benchmarks.constrained_rl_v2.costs import (
    COST_NAMES,
    deadline_deficit_cost,
    deadline_state,
    optimistic_remaining_time_s,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification


class BalancedMarginSampler:
    """Independent shuffled blocks containing every margin exactly once."""

    def __init__(self, margins_s: tuple[float, ...], *, seed: int):
        if not margins_s or len(set(margins_s)) != len(margins_s):
            raise ValueError("Margins must be non-empty and unique")
        if any(not isfinite(value) or value <= 0 for value in margins_s):
            raise ValueError("Margins must be finite and positive")
        self.margins_s = tuple(float(value) for value in margins_s)
        self._rng = np.random.default_rng(seed)
        self._pending: list[float] = []

    def sample(self) -> float:
        if not self._pending:
            self._pending = list(self._rng.permutation(self.margins_s))
        return float(self._pending.pop())


@dataclass(frozen=True)
class CompletedRequirementEpisode:
    objective_return: float
    costs: tuple[float, float]
    metrics: EpisodeMetrics
    simulator_steps: int
    budget_truncated: bool
    final_position_m: float
    requirement_margin_s: float
    t_min_start_s: float
    max_time_s: float
    final_deadline_deficit_s: float
    maximum_deadline_deficit_s: float


class RequirementConditionedTaskWrapper(gym.Wrapper):
    """Expose absolute deadline and elapsed time while retaining V2 costs."""

    def __init__(
        self,
        environment: gym.Env,
        *,
        margins_s: tuple[float, ...],
        sampler_seed: int,
        time_scale_s: float,
        maximum_t_min_start_s: float,
        energy_scale_kwh: float,
        max_speed_violation_m_s: float = 0.0,
        initial_seed: int | None = None,
        max_simulator_steps: int | None = None,
    ):
        super().__init__(environment)
        if not isinstance(environment.observation_space, spaces.Box):
            raise TypeError("Requirement wrapper requires a Box observation")
        if time_scale_s <= 0 or maximum_t_min_start_s <= 0:
            raise ValueError("Time normalization bounds must be positive")
        self.margin_sampler = BalancedMarginSampler(margins_s, seed=sampler_seed)
        self.margins_s = tuple(float(value) for value in margins_s)
        self.time_scale_s = float(time_scale_s)
        self.maximum_t_min_start_s = float(maximum_t_min_start_s)
        self.maximum_deadline_s = maximum_t_min_start_s + max(margins_s)
        self.energy_scale_kwh = float(energy_scale_kwh)
        self.max_speed_violation_m_s = float(max_speed_violation_m_s)
        self.initial_seed = initial_seed
        self.max_simulator_steps = max_simulator_steps
        self.simulator_steps = 0
        self.last_completed_episode: CompletedRequirementEpisode | None = None
        self.current_task: TaskSpecification | None = None
        self.requirement_margin_s: float | None = None
        self.t_min_start_s: float | None = None
        self._initial_seed_pending = initial_seed is not None
        self._last_elapsed_time_s = 0.0
        self._episode_objective_return = 0.0
        self._episode_costs = np.zeros(2, dtype=np.float64)
        self._episode_steps = 0
        self._maximum_deadline_deficit_s = 0.0
        low = np.concatenate([environment.observation_space.low, np.array([0.0, 0.0])])
        high = np.concatenate(
            [
                environment.observation_space.high,
                np.array([self.maximum_deadline_s / self.time_scale_s, 1.0]),
            ]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)

    def _augment(self, observation, elapsed_time_s: float):
        if self.current_task is None:
            raise RuntimeError("Requirement is unavailable before reset")
        return np.concatenate(
            [
                np.asarray(observation, dtype=np.float64),
                np.array(
                    [
                        self.current_task.max_time_s / self.time_scale_s,
                        elapsed_time_s / self.time_scale_s,
                    ]
                ),
            ]
        )

    def _requirement_info(self) -> dict[str, float]:
        if self.current_task is None or self.requirement_margin_s is None:
            raise RuntimeError("Requirement is unavailable before reset")
        return {
            "requirement_margin_s": self.requirement_margin_s,
            "requirement_max_time_s": self.current_task.max_time_s,
            "requirement_t_min_start_s": float(self.t_min_start_s),
            "requirement_time_scale_s": self.time_scale_s,
        }

    def reset(self, *, seed=None, options=None):
        if seed is None and self._initial_seed_pending:
            seed = self.initial_seed
        self._initial_seed_pending = False
        explicit_margin = None
        if options is not None:
            if set(options) != {"requirement_margin_s"}:
                raise ValueError("Only requirement_margin_s reset option is supported")
            explicit_margin = float(options["requirement_margin_s"])
            if not isfinite(explicit_margin) or explicit_margin <= 0:
                raise ValueError("Explicit requirement margin must be positive")
        observation, raw_info = self.env.reset(seed=seed)
        base = self.env.unwrapped
        t_min_start = optimistic_remaining_time_s(
            base.track,
            position_m=0.0,
            track_length_m=float(base.config.track_length_m),
        )
        if t_min_start > self.maximum_t_min_start_s + 1e-12:
            raise ValueError("Track exceeds the preregistered physical time bound")
        margin = (
            self.margin_sampler.sample() if explicit_margin is None else explicit_margin
        )
        max_time = t_min_start + margin
        if max_time > self.maximum_deadline_s + 1e-12:
            raise ValueError("Episode requirement exceeds observation-space bound")
        self.requirement_margin_s = margin
        self.t_min_start_s = t_min_start
        self.current_task = TaskSpecification(
            max_time_s=max_time,
            max_speed_violation_m_s=self.max_speed_violation_m_s,
        )
        self._last_elapsed_time_s = float(raw_info["elapsed_time_s"])
        self._episode_objective_return = 0.0
        self._episode_costs = np.zeros(2, dtype=np.float64)
        self._episode_steps = 0
        self._maximum_deadline_deficit_s = 0.0
        info = {**raw_info, **self._requirement_info()}
        return self._augment(observation, self._last_elapsed_time_s), info

    def step(self, action):
        if self.current_task is None:
            raise gym.error.ResetNeeded("Call reset before step")
        if (
            self.max_simulator_steps is not None
            and self.simulator_steps >= self.max_simulator_steps
        ):
            raise gym.error.ResetNeeded("The lifetime simulator budget is exhausted")
        observation, _historical_reward, terminated, truncated, raw_info = (
            self.env.step(action)
        )
        info = dict(raw_info)
        elapsed = float(info["elapsed_time_s"])
        dt_s = elapsed - self._last_elapsed_time_s
        self._last_elapsed_time_s = elapsed
        reward = objective_reward(float(info["step_energy_kwh"]), self.energy_scale_kwh)
        base = self.env.unwrapped
        remaining, slack, deficit = deadline_state(
            elapsed_time_s=elapsed,
            position_m=float(info["position_m"]),
            track=base.track,
            track_length_m=float(base.config.track_length_m),
            deadline_s=self.current_task.max_time_s,
        )
        step_costs = np.array(
            [
                speed_cost(float(info["speed_excess_m_s"]), dt_s),
                deadline_deficit_cost(
                    deficit_s=deficit,
                    dt_s=dt_s,
                    normalization_s=self.current_task.max_time_s,
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
            self._maximum_deadline_deficit_s, deficit
        )
        info.update(
            {
                **self._requirement_info(),
                "objective_reward": reward,
                "constraint_cost_names": COST_NAMES,
                "constraint_costs": dict(zip(COST_NAMES, map(float, step_costs))),
                "cost": step_costs.copy(),
                "deadline_optimistic_remaining_time_s": remaining,
                "deadline_slack_s": slack,
                "deadline_deficit_s": deficit,
                "deadline_deficit_cost_s": float(step_costs[1]),
                "deadline_deficit_integral_s": float(self._episode_costs[1]),
                "training_budget_truncated": budget_truncated,
                "simulator_steps_total": self.simulator_steps,
            }
        )
        if terminated or truncated:
            self.last_completed_episode = CompletedRequirementEpisode(
                objective_return=self._episode_objective_return,
                costs=tuple(float(value) for value in self._episode_costs),
                metrics=EpisodeMetrics(**info["episode_metrics"]),
                simulator_steps=self._episode_steps,
                budget_truncated=budget_truncated,
                final_position_m=float(info["position_m"]),
                requirement_margin_s=float(self.requirement_margin_s),
                t_min_start_s=float(self.t_min_start_s),
                max_time_s=self.current_task.max_time_s,
                final_deadline_deficit_s=deficit,
                maximum_deadline_deficit_s=self._maximum_deadline_deficit_s,
            )
        return (
            self._augment(observation, elapsed),
            reward,
            bool(terminated),
            truncated,
            info,
        )
