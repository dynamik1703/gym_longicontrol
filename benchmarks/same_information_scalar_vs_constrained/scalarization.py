"""Reference-informed fixed scalarization and analytical helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .config import load_configuration


@dataclass(frozen=True)
class Scalarization:
    lambda_speed: float
    lambda_deadline: float
    normalizer: float
    w_reward: float
    w_speed: float
    w_deadline: float

    def reward(
        self, objective: float, speed_cost: float, deadline_cost: float
    ) -> float:
        values = (objective, speed_cost, deadline_cost)
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("Scalar reward inputs must be finite")
        if speed_cost < 0 or deadline_cost < 0:
            raise ValueError("Constraint costs must be nonnegative")
        return float(
            self.w_reward * objective
            - self.w_speed * speed_cost
            - self.w_deadline * deadline_cost
        )


def frozen_scalarization() -> Scalarization:
    raw = load_configuration().raw["scalarization"]
    speed = float(np.median(raw["lambda_speed_values"]))
    deadline = float(np.median(raw["lambda_deadline_values"]))
    normalizer = 1.0 + speed + deadline
    calculated = {
        "lambda_speed": speed,
        "lambda_deadline": deadline,
        "normalizer": normalizer,
        "w_reward": 1.0 / normalizer,
        "w_speed": speed / normalizer,
        "w_deadline": deadline / normalizer,
    }
    for name, value in calculated.items():
        frozen_name = {
            "lambda_speed": "lambda_speed_star",
            "lambda_deadline": "lambda_deadline_star",
        }.get(name, name)
        if not np.isclose(value, raw[frozen_name], rtol=0, atol=1e-15):
            raise RuntimeError(f"Frozen scalarization mismatch: {name}")
    return Scalarization(**calculated)


def scalar_reward(objective: float, speed_cost: float, deadline_cost: float) -> float:
    return frozen_scalarization().reward(objective, speed_cost, deadline_cost)


def n_step_scalar_return(
    rewards: tuple[float, ...], *, bootstrap: float, gamma: float, terminated: bool
) -> float:
    """Exact scalar n-step target used by the pinned FSRL return convention."""

    if not rewards:
        raise ValueError("At least one reward is required")
    result = sum((gamma**index) * value for index, value in enumerate(rewards))
    if not terminated:
        result += gamma ** len(rewards) * bootstrap
    return float(result)
