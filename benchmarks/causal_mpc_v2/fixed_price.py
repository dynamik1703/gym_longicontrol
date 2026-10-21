"""Physical time-price mapping and constant selector for Causal MPC V2."""

from __future__ import annotations

import math

import numpy as np

LAMBDA_MIN_KWH_PER_S = 1.0e-5
LAMBDA_MAX_KWH_PER_S = 1.0e-2


def normalized_action_to_lambda(action: float) -> float:
    value = float(action)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("normalized price must be finite in [-1, 1]")
    if value == -1.0:
        return LAMBDA_MIN_KWH_PER_S
    if value == 1.0:
        return LAMBDA_MAX_KWH_PER_S
    fraction = 0.5 * (value + 1.0)
    return math.exp(
        math.log(LAMBDA_MIN_KWH_PER_S)
        + fraction
        * math.log(LAMBDA_MAX_KWH_PER_S / LAMBDA_MIN_KWH_PER_S)
    )


def lambda_to_normalized_action(price_kwh_per_s: float) -> float:
    value = float(price_kwh_per_s)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("time price must be finite and nonnegative")
    clipped = float(
        np.clip(value, LAMBDA_MIN_KWH_PER_S, LAMBDA_MAX_KWH_PER_S)
    )
    fraction = math.log(clipped / LAMBDA_MIN_KWH_PER_S) / math.log(
        LAMBDA_MAX_KWH_PER_S / LAMBDA_MIN_KWH_PER_S
    )
    return 2.0 * fraction - 1.0


class FixedPrice:
    def __init__(self, normalized_action: float) -> None:
        value = float(normalized_action)
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("fixed price must be finite in [-1, 1]")
        self.normalized_action = value
        self.price_kwh_per_s = normalized_action_to_lambda(value)

    def reset(self) -> None:
        return None

    def action(self, strategy_observation: np.ndarray) -> float:
        value = np.asarray(strategy_observation, dtype=np.float64)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError("strategy observation must be a finite 16-vector")
        return self.normalized_action
