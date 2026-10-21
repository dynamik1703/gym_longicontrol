"""Frozen constant-price selector for the shared causal MPC."""

from __future__ import annotations

import numpy as np

from .time_price import lambda_to_normalized_action, normalized_action_to_lambda

# Frozen after the finite Development calibration recorded in
# ``mpc_development_results.json``.  The initial value is the centre of the
# logarithmic physical interval and is updated only before the study freeze.
FIXED_PRICE_ACTION = 1.0
FIXED_PRICE_KWH_PER_S = normalized_action_to_lambda(FIXED_PRICE_ACTION)


class FixedPrice:
    """Return one route-independent time price at every decision boundary."""

    def __init__(self, normalized_action: float = FIXED_PRICE_ACTION) -> None:
        value = float(normalized_action)
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("fixed normalized price must be finite in [-1, 1]")
        self.normalized_action = value
        self.price_kwh_per_s = normalized_action_to_lambda(value)

    @classmethod
    def from_price(cls, price_kwh_per_s: float) -> FixedPrice:
        return cls(lambda_to_normalized_action(price_kwh_per_s))

    def reset(self) -> None:
        return None

    def action(self, strategy_observation: np.ndarray) -> float:
        value = np.asarray(strategy_observation, dtype=np.float64)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError("strategy observation must be a finite 16-vector")
        return self.normalized_action
