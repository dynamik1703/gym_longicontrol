"""Causal deadline-paced terminal continuation for the V2 spatial MPC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from benchmarks.hybrid_rl_design.viability import (
    DEADLINE_RESERVE_S,
    DEADLINE_S,
    MINIMUM_LIMIT_M_S,
)
from benchmarks.physical_feasibility_reference.causal_controller import (
    CAUSAL_CRUISE_CAP_M_S,
)
from benchmarks.physical_feasibility_reference.oracle_controller import (
    VELOCITY_SAFETY_MARGIN_M_S,
    Transition,
    action_for_next_velocity,
    braking_safe,
)
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

PACE_MARGIN_M_S = 5.0 / 3.6
MAX_CONNECTOR_NATIVE_STEPS = 150
MAX_COAST_CONNECTOR_NATIVE_STEPS = 50
LATENESS_PENALTY_KWH_PER_S = 1.0e-2


@dataclass(frozen=True)
class TerminalEstimate:
    energy_kwh: float
    time_s: float
    target_speed_m_s: float
    final_connection_position_m: float
    final_connection_velocity_m_s: float
    deadline_feasible_estimate: bool
    estimated_completion_time_s: float
    lateness_penalty_kwh: float
    physics_calls: int
    classification: str = "HEURISTIC_CONSTANT_SPEED_UNKNOWN_TAIL"

    @property
    def total_energy_kwh(self) -> float:
        return self.energy_kwh + self.lateness_penalty_kwh

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def limit_at(
    position_m: float,
    current_limit_m_s: float,
    transitions: tuple[Transition, ...],
) -> float:
    limit = float(current_limit_m_s)
    for transition in transitions:
        if transition.position_m <= position_m + 1.0e-12:
            limit = float(transition.limit_m_s)
    return limit


def deadline_paced_target(
    state: VehicleState,
    *,
    current_limit_m_s: float,
    config: SimulationConfig,
) -> float:
    remaining_distance = max(0.0, config.track_length_m - state.position_m)
    usable_time = max(
        config.dt_s,
        DEADLINE_S - DEADLINE_RESERVE_S - state.elapsed_time_s,
    )
    required_average = remaining_distance / usable_time
    target = max(MINIMUM_LIMIT_M_S, required_average + PACE_MARGIN_M_S)
    return max(
        config.dt_s * 0.0,
        min(
            target,
            CAUSAL_CRUISE_CAP_M_S,
            current_limit_m_s - VELOCITY_SAFETY_MARGIN_M_S,
        ),
    )


def _steady_energy_rate(
    velocity_m_s: float,
    *,
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> tuple[float, float, int]:
    target = max(1.0e-6, float(velocity_m_s))
    state = VehicleState(velocity_m_s=target)
    action = action_for_next_velocity(
        state, target, vehicle=vehicle, config=config
    )
    next_state, power_kw, _ = advance(state, action, vehicle, config)
    mean_velocity = max(
        1.0e-6, 0.5 * (state.velocity_m_s + next_state.velocity_m_s)
    )
    # Synthetic continuation never claims a regenerative tail credit.
    return max(0.0, power_kw) / 3600.0, mean_velocity, 1


def estimate_terminal_value(
    state: VehicleState,
    *,
    current_limit_m_s: float,
    transitions: tuple[Transition, ...],
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> TerminalEstimate:
    """Estimate a full causal continuation without requiring 60 km/h.

    Exact production dynamics connect the terminal velocity to a deadline-paced
    speed. The remaining distance uses production-model steady power. Signed
    regenerative connector energy is clipped at zero to prevent a horizon-edge
    braking reward. This is explicitly a heuristic, not a value-function bound.
    """

    remaining_route = max(0.0, config.track_length_m - state.position_m)
    if remaining_route <= 1.0e-12:
        return TerminalEstimate(
            energy_kwh=0.0,
            time_s=0.0,
            target_speed_m_s=0.0,
            final_connection_position_m=state.position_m,
            final_connection_velocity_m_s=state.velocity_m_s,
            deadline_feasible_estimate=state.elapsed_time_s <= DEADLINE_S,
            estimated_completion_time_s=state.elapsed_time_s,
            lateness_penalty_kwh=max(
                0.0, state.elapsed_time_s - DEADLINE_S
            )
            * LATENESS_PENALTY_KWH_PER_S,
            physics_calls=0,
        )

    terminal_limit = limit_at(state.position_m, current_limit_m_s, transitions)
    target = deadline_paced_target(
        state, current_limit_m_s=terminal_limit, config=config
    )
    cursor = state
    connector_energy = 0.0
    calls = 0
    while cursor.position_m < config.track_length_m - 1.0e-12:
        local_limit = limit_at(
            cursor.position_m, terminal_limit, transitions
        )
        local_target = max(
            0.0,
            min(target, local_limit - VELOCITY_SAFETY_MARGIN_M_S),
        )
        velocity_error = cursor.velocity_m_s - local_target
        if abs(velocity_error) <= 1.0e-3:
            break
        if velocity_error < 0.0:
            desired = min(
                local_target,
                cursor.velocity_m_s + 3.0 * config.dt_s,
            )
            action = action_for_next_velocity(
                cursor, desired, vehicle=vehicle, config=config
            )
        elif calls < MAX_COAST_CONNECTOR_NATIVE_STEPS:
            action = 0.0
        else:
            desired = max(
                local_target,
                cursor.velocity_m_s - 1.0 * config.dt_s,
            )
            action = action_for_next_velocity(
                cursor, desired, vehicle=vehicle, config=config
            )
        next_state, _, signed_energy = advance(cursor, action, vehicle, config)
        if not braking_safe(
            next_state, transitions, vehicle=vehicle, config=config
        ):
            next_state, _, signed_energy = advance(
                cursor, -1.0, vehicle, config
            )
        connector_energy += max(0.0, signed_energy)
        calls += 1
        if next_state.position_m <= cursor.position_m + 1.0e-12:
            break
        cursor = next_state
        if calls >= MAX_CONNECTOR_NATIVE_STEPS:
            break

    connector_time = cursor.elapsed_time_s - state.elapsed_time_s
    remaining = max(0.0, config.track_length_m - cursor.position_m)
    local_limit = limit_at(cursor.position_m, terminal_limit, transitions)
    cruise_speed = max(
        1.0e-6,
        min(target, local_limit - VELOCITY_SAFETY_MARGIN_M_S),
    )
    energy_rate_kwh_per_s, mean_velocity, steady_calls = _steady_energy_rate(
        cruise_speed, vehicle=vehicle, config=config
    )
    calls += steady_calls
    cruise_time = remaining / mean_velocity
    cruise_energy = energy_rate_kwh_per_s * cruise_time
    time_s = connector_time + cruise_time
    completion = state.elapsed_time_s + time_s
    lateness = max(0.0, completion - (DEADLINE_S - DEADLINE_RESERVE_S))
    penalty = lateness * LATENESS_PENALTY_KWH_PER_S
    values = (connector_energy, cruise_energy, time_s, completion, penalty)
    if not np.isfinite(values).all():
        raise RuntimeError("terminal heuristic produced a non-finite value")
    return TerminalEstimate(
        energy_kwh=float(connector_energy + cruise_energy),
        time_s=float(time_s),
        target_speed_m_s=float(target),
        final_connection_position_m=float(cursor.position_m),
        final_connection_velocity_m_s=float(cursor.velocity_m_s),
        deadline_feasible_estimate=completion <= DEADLINE_S - DEADLINE_RESERVE_S,
        estimated_completion_time_s=float(completion),
        lateness_penalty_kwh=float(penalty),
        physics_calls=calls,
    )
