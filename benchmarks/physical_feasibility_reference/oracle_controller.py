"""Privileged full-route deterministic physics reference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.track import Track
from gym_longicontrol.domain.vehicle import VehicleModel

from .feasibility import braking_distance_m

# Frozen before holdout.  The evaluator still applies the exact zero tolerance.
VELOCITY_SAFETY_MARGIN_M_S = 1e-6
BRAKING_POSITION_MARGIN_M = 0.5


@dataclass(frozen=True)
class Transition:
    position_m: float
    limit_m_s: float


def speed_limit_at(track: Track, position_m: float) -> float:
    index = max(
        0, int(np.searchsorted(track.positions_m, position_m, side="right")) - 1
    )
    return float(track.limits_m_s[index])


def action_for_next_velocity(
    state: VehicleState,
    desired_next_velocity_m_s: float,
    *,
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> float:
    """Invert the production action interpolation for one desired next speed."""

    required = (desired_next_velocity_m_s - state.velocity_m_s) / config.dt_s
    coast = vehicle.acceleration_from_action(state.velocity_m_s, 0.0)
    if required >= coast:
        extreme = vehicle.acceleration_from_action(state.velocity_m_s, 1.0)
        denominator = extreme - coast
        action = 1.0 if denominator <= 0 else (required - coast) / denominator
    else:
        extreme = vehicle.acceleration_from_action(state.velocity_m_s, -1.0)
        denominator = coast - extreme
        action = -1.0 if denominator <= 0 else -(coast - required) / denominator
    return float(np.clip(action, -1.0, 1.0))


def braking_safe(
    next_state: VehicleState,
    transitions: tuple[Transition, ...],
    *,
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> bool:
    """Check that immediate maximum braking reaches every known lower limit."""

    for transition in transitions:
        remaining = transition.position_m - next_state.position_m
        if remaining <= 0 or next_state.velocity_m_s <= transition.limit_m_s:
            continue
        target = max(0.0, transition.limit_m_s - VELOCITY_SAFETY_MARGIN_M_S)
        required = braking_distance_m(
            next_state.velocity_m_s,
            target,
            vehicle=vehicle,
            config=config,
        )
        if required + BRAKING_POSITION_MARGIN_M > remaining:
            return False
    return True


class FullRoutePhysicsOracle:
    """Route-aware controller whose proposed actions run in the original env."""

    information = (
        "complete generated route, exact vehicle dynamics, current physical state, "
        "canonical deadline"
    )

    def __init__(
        self,
        track: Track,
        *,
        cruise_cap_m_s: float = 37.0,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
    ) -> None:
        self.track = track
        self.cruise_cap_m_s = float(cruise_cap_m_s)
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config

    def _future_transitions(self, position_m: float) -> tuple[Transition, ...]:
        return tuple(
            Transition(float(position), float(limit))
            for position, limit in zip(
                self.track.positions_m[1:], self.track.limits_m_s[1:]
            )
            if position > position_m
        )

    def action(self, state: VehicleState) -> float:
        current_limit = speed_limit_at(self.track, state.position_m)
        target_velocity = max(
            0.0,
            min(self.cruise_cap_m_s, current_limit)
            - VELOCITY_SAFETY_MARGIN_M_S,
        )
        candidate = action_for_next_velocity(
            state,
            target_velocity,
            vehicle=self.vehicle,
            config=self.config,
        )
        next_state, _, _ = advance(
            state, candidate, self.vehicle, self.config
        )
        endpoint_limit = speed_limit_at(self.track, next_state.position_m)
        endpoint_safe = (
            next_state.velocity_m_s
            <= endpoint_limit - VELOCITY_SAFETY_MARGIN_M_S
        )
        transitions = self._future_transitions(next_state.position_m)
        if endpoint_safe and braking_safe(
            next_state,
            transitions,
            vehicle=self.vehicle,
            config=self.config,
        ):
            return candidate
        return -1.0
