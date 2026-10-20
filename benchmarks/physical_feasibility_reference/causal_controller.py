"""Deterministic observation-limited physics reference."""

from __future__ import annotations

import numpy as np

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .oracle_controller import (
    VELOCITY_SAFETY_MARGIN_M_S,
    Transition,
    action_for_next_velocity,
    braking_safe,
)

# At 60 km/h the vehicle can brake to the generator's minimum 20 km/h limit
# well inside the 150 m preview.  This fixed cap protects against an unseen
# maximum legal 40 km/h downward change and was frozen before holdout.
CAUSAL_CRUISE_CAP_M_S = 60.0 / 3.6


def decode_observation(observation: np.ndarray) -> dict[str, object]:
    value = np.asarray(observation, dtype=np.float64)
    if value.shape != (8,) or not np.isfinite(value).all():
        raise ValueError("ordinary observation must be a finite 8-vector")
    return {
        "velocity_m_s": float(value[0] * 37.0),
        "acceleration_m_s2": float(value[1] * 6.0 - 3.0),
        "current_limit_m_s": float(value[2] * 37.0),
        "future_limits_m_s": (float(value[3] * 37.0), float(value[4] * 37.0)),
        "distances_m": (float(value[5] * 150.0), float(value[6] * 150.0)),
    }


class ObservationLimitedController:
    """Causal controller with disclosed deterministic internal memory.

    Its position, velocity, acceleration and elapsed time are propagated only
    from the reset state, its own previous actions and known exact dynamics.
    It never receives the environment state or full Track object.
    """

    information = (
        "ordinary 8D observation; exact known dynamics; fixed task parameters; "
        "internally propagated position/velocity/acceleration/elapsed time; "
        "remembered previously visible transitions"
    )

    def __init__(
        self,
        *,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
    ) -> None:
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        self.state = VehicleState()
        self._known_transitions: dict[float, float] = {}
        self._last_action: float | None = None

    def reset(self) -> None:
        self.state = VehicleState()
        self._known_transitions.clear()
        self._last_action = None

    def _ingest_visible(self, decoded: dict[str, object]) -> None:
        current = float(decoded["current_limit_m_s"])
        limits = tuple(decoded["future_limits_m_s"])
        distances = tuple(decoded["distances_m"])
        previous = current
        for limit, distance in zip(limits, distances):
            # A clamped placeholder repeats the latest visible limit at 150 m.
            visible = distance < self.config.sensor_range_m - 1e-9 or limit != previous
            if visible and limit != previous:
                absolute = round(self.state.position_m + distance, 9)
                if absolute > self.state.position_m:
                    self._known_transitions[absolute] = float(limit)
            previous = float(limit)
        self._known_transitions = {
            position: limit
            for position, limit in self._known_transitions.items()
            if position > self.state.position_m + 1e-9
        }

    def action(self, observation: np.ndarray) -> float:
        decoded = decode_observation(observation)
        if abs(float(decoded["velocity_m_s"]) - self.state.velocity_m_s) > 1e-9:
            raise RuntimeError("causal internal velocity diverged from observation")
        if (
            abs(float(decoded["acceleration_m_s2"]) - self.state.acceleration_m_s2)
            > 1e-9
        ):
            raise RuntimeError("causal internal acceleration diverged from observation")
        self._ingest_visible(decoded)
        current_limit = float(decoded["current_limit_m_s"])
        target_velocity = max(
            0.0,
            min(CAUSAL_CRUISE_CAP_M_S, current_limit)
            - VELOCITY_SAFETY_MARGIN_M_S,
        )
        candidate = action_for_next_velocity(
            self.state,
            target_velocity,
            vehicle=self.vehicle,
            config=self.config,
        )
        next_state, _, _ = advance(
            self.state, candidate, self.vehicle, self.config
        )
        transitions = tuple(
            Transition(position, limit)
            for position, limit in sorted(self._known_transitions.items())
        )
        # The decoded current limit remains valid unless a remembered transition
        # is crossed during this step.
        endpoint_limit = current_limit
        for transition in transitions:
            if transition.position_m <= next_state.position_m:
                endpoint_limit = transition.limit_m_s
        if (
            next_state.velocity_m_s
            <= endpoint_limit - VELOCITY_SAFETY_MARGIN_M_S
            and braking_safe(
                next_state,
                transitions,
                vehicle=self.vehicle,
                config=self.config,
            )
        ):
            self._last_action = candidate
            return candidate
        self._last_action = -1.0
        return -1.0

    def observe_executed_action(self, action: float) -> None:
        """Advance memory from the controller's own action, without env state."""

        if self._last_action is None or float(action) != self._last_action:
            raise RuntimeError("executed action differs from causal proposal")
        self.state, _, _ = advance(
            self.state, float(action), self.vehicle, self.config
        )
        self._last_action = None
