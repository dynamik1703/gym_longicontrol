"""Frozen deterministic non-learning strategy for the future comparison."""

from __future__ import annotations

import math

from gym_longicontrol.domain.state import VehicleState

from .controller import (
    TARGET_MAX_M_S,
    TARGET_MIN_M_S,
    target_speed_to_normalized_action,
)
from .viability import DEADLINE_RESERVE_S, DEADLINE_S

TIME_AWARE_SPEED_MARGIN_KM_H = 5.0


class TimeAwareEnergyStrategy:
    """Choose the slowest 1 km/h target with simple deadline headroom.

    This baseline is deliberately non-learning and uses only controller memory.
    Its target is refreshed every ten native steps, exactly like the RL arm, and
    is passed through the identical shield.
    """

    decision_interval_steps = 10

    def __init__(self) -> None:
        self._held_action = 1.0

    def reset(self) -> None:
        self._held_action = 1.0

    def action(self, state: VehicleState, step: int, *, route_length_m: float) -> float:
        if step % self.decision_interval_steps:
            return self._held_action
        usable = max(0.1, DEADLINE_S - DEADLINE_RESERVE_S - state.elapsed_time_s)
        remaining = max(0.0, route_length_m - state.position_m)
        required_km_h = 3.6 * remaining / usable
        target_km_h = math.ceil(required_km_h + TIME_AWARE_SPEED_MARGIN_KM_H)
        target_m_s = min(
            TARGET_MAX_M_S,
            max(TARGET_MIN_M_S, target_km_h / 3.6),
        )
        self._held_action = target_speed_to_normalized_action(target_m_s)
        return self._held_action
