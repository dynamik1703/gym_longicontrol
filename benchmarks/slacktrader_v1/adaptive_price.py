"""Classical deadline-reserve feedback for the shared causal MPC."""

from __future__ import annotations

import numpy as np

from benchmarks.hybrid_rl_design.viability import DEADLINE_S

from .time_price import normalized_action_to_lambda

HIGH_RESERVE_S = 24.0
LOW_RESERVE_S = 2.0


class AdaptivePrice:
    """Increase time price monotonically as causal deadline slack decreases."""

    def __init__(
        self,
        *,
        high_reserve_s: float = HIGH_RESERVE_S,
        low_reserve_s: float = LOW_RESERVE_S,
    ) -> None:
        if not 0.0 <= low_reserve_s < high_reserve_s:
            raise ValueError("reserve thresholds must satisfy 0 <= low < high")
        self.high_reserve_s = float(high_reserve_s)
        self.low_reserve_s = float(low_reserve_s)
        self.last_action = 1.0

    def reset(self) -> None:
        self.last_action = 1.0

    def action(self, strategy_observation: np.ndarray) -> float:
        value = np.asarray(strategy_observation, dtype=np.float64)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError("strategy observation must be a finite 16-vector")
        slack_s = float(value[14] * DEADLINE_S)
        fraction = (slack_s - self.low_reserve_s) / (
            self.high_reserve_s - self.low_reserve_s
        )
        self.last_action = float(np.clip(1.0 - 2.0 * fraction, -1.0, 1.0))
        return self.last_action

    @property
    def last_price_kwh_per_s(self) -> float:
        return normalized_action_to_lambda(self.last_action)
