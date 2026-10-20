"""One shared signal path with a scalar-reward view."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from benchmarks.constrained_rl.costs import objective_reward, speed_cost
from benchmarks.constrained_rl_v2.costs import (
    DenseDeadlineTaskWrapper,
    deadline_deficit_cost,
)

from .scalarization import scalar_reward


@dataclass(frozen=True)
class TransitionSignals:
    objective: float
    speed_cost: float
    deadline_cost: float


def canonical_transition_signals(
    *,
    signed_step_energy_kwh: float,
    velocity_m_s: float,
    speed_limit_m_s: float,
    elapsed_time_s: float,
    optimistic_remaining_time_s: float,
    deadline_s: float = 140.0,
    dt_s: float = 0.1,
) -> TransitionSignals:
    """Pure fixture form of the exact wrapper equations."""

    deficit = max(0.0, elapsed_time_s + optimistic_remaining_time_s - deadline_s)
    return TransitionSignals(
        objective_reward(signed_step_energy_kwh, 0.25),
        speed_cost(max(0.0, velocity_m_s - speed_limit_m_s), dt_s),
        deadline_deficit_cost(deficit_s=deficit, dt_s=dt_s, normalization_s=deadline_s),
    )


def signals_from_info(objective: float, info: dict[str, Any]) -> TransitionSignals:
    costs = np.asarray(info["cost"], dtype=np.float64)
    if costs.shape != (2,):
        raise ValueError("Expected speed and deadline cost in canonical order")
    stored_objective = float(info["objective_reward"])
    if stored_objective != float(objective):
        raise RuntimeError("Objective signal diverged inside the shared wrapper")
    return TransitionSignals(stored_objective, float(costs[0]), float(costs[1]))


class SameInformationScalarWrapper(DenseDeadlineTaskWrapper):
    """Expose the canonical three signals as one fixed scalar reward.

    The superclass is the exact Constrained-V2 signal implementation. No terminal,
    progress, time, or completion term is added here.
    """

    def step(self, action):
        observation, objective, terminated, truncated, raw_info = super().step(action)
        info = dict(raw_info)
        signals = signals_from_info(objective, info)
        reward = scalar_reward(
            signals.objective, signals.speed_cost, signals.deadline_cost
        )
        info["same_information_signals"] = {
            "r_E": signals.objective,
            "c_speed": signals.speed_cost,
            "c_deadline": signals.deadline_cost,
        }
        info["scalar_reward"] = reward
        info["scalar_reward_has_hidden_terms"] = False
        return observation, reward, terminated, truncated, info
