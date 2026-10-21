"""Deterministic fixed-distance causal planner shared by all price arms."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

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

PLANNING_DISTANCE_M = 100.0
CONTROL_SEGMENT_NATIVE_STEPS = 10
CANDIDATE_SPEEDS_KM_H = (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0)
MAX_PLANNING_NATIVE_STEPS = 1_800
OBJECTIVE_TIE_TOLERANCE = 1.0e-12


@dataclass(frozen=True)
class CandidateTrajectory:
    target_speed_m_s: float
    actions: tuple[float, ...]
    travel_time_s: float
    horizon_energy_kwh: float
    terminal_energy_kwh: float
    objective_kwh: float
    final_position_m: float
    final_velocity_m_s: float
    feasible: bool
    rejection_reason: str | None
    physics_calls: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Plan:
    price_kwh_per_s: float
    start_position_m: float
    waypoint_position_m: float
    planning_distance_m: float
    selected: CandidateTrajectory
    candidates: tuple[CandidateTrajectory, ...]
    segment_actions: tuple[float, ...]
    physics_calls: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def limit_at(
    position_m: float,
    current_limit_m_s: float,
    transitions: tuple[Transition, ...],
) -> float:
    """Return the causally known limit at a propagated position."""

    limit = float(current_limit_m_s)
    for transition in transitions:
        if transition.position_m <= position_m + 1.0e-12:
            limit = float(transition.limit_m_s)
    return limit


def _terminal_continuation_energy(
    state: VehicleState,
    *,
    reference_speed_m_s: float,
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> tuple[float, int]:
    """Exact-dynamics energy-to-speed correction preventing slow terminal bias.

    The continuation is deliberately charged only when a candidate ends below
    the common reference speed.  Its propagated distance is not treated as
    observed route information and is not used for control or constraints.
    """

    if state.velocity_m_s >= reference_speed_m_s - 1.0e-9:
        return 0.0, 0
    cursor = state
    calls = 0
    start_energy = state.total_energy_kwh
    while cursor.velocity_m_s < reference_speed_m_s - 1.0e-9:
        desired = min(reference_speed_m_s, cursor.velocity_m_s + 3.0 * config.dt_s)
        action = action_for_next_velocity(
            cursor, desired, vehicle=vehicle, config=config
        )
        cursor, _, _ = advance(cursor, action, vehicle, config)
        calls += 1
        if calls > 2_000:
            raise RuntimeError("terminal continuation failed to converge")
    return max(0.0, cursor.total_energy_kwh - start_energy), calls


class FixedDistancePlanner:
    """Enumerate a frozen speed lattice to a spatial, never temporal, waypoint.

    Every rollout uses the production ``advance`` function.  The lattice is a
    deterministic one-dimensional optimizer: O(C * H), where C is the nine
    frozen candidate speeds and H is native steps required to cover 100 m.
    """

    def __init__(
        self,
        *,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
        planning_distance_m: float = PLANNING_DISTANCE_M,
    ) -> None:
        if not 0.0 < planning_distance_m <= config.sensor_range_m:
            raise ValueError("planning distance must be positive and within preview")
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        self.planning_distance_m = float(planning_distance_m)

    def _candidate(
        self,
        state: VehicleState,
        *,
        target_speed_m_s: float,
        price_kwh_per_s: float,
        waypoint_position_m: float,
        current_limit_m_s: float,
        transitions: tuple[Transition, ...],
    ) -> CandidateTrajectory:
        cursor = state
        actions: list[float] = []
        calls = 0
        reason: str | None = None
        while cursor.position_m < waypoint_position_m - 1.0e-12:
            limit = limit_at(cursor.position_m, current_limit_m_s, transitions)
            desired = max(
                0.0,
                min(target_speed_m_s, CAUSAL_CRUISE_CAP_M_S, limit)
                - VELOCITY_SAFETY_MARGIN_M_S,
            )
            action = action_for_next_velocity(
                cursor, desired, vehicle=self.vehicle, config=self.config
            )
            next_state, _, _ = advance(cursor, action, self.vehicle, self.config)
            calls += 1
            endpoint_limit = limit_at(
                next_state.position_m, current_limit_m_s, transitions
            )
            endpoint_safe = (
                next_state.velocity_m_s
                <= endpoint_limit - VELOCITY_SAFETY_MARGIN_M_S + 1.0e-12
            )
            known_safe = braking_safe(
                next_state,
                transitions,
                vehicle=self.vehicle,
                config=self.config,
            )
            if not endpoint_safe or not known_safe:
                action = -1.0
                next_state, _, _ = advance(
                    cursor, action, self.vehicle, self.config
                )
                calls += 1
                endpoint_limit = limit_at(
                    next_state.position_m, current_limit_m_s, transitions
                )
                if (
                    next_state.velocity_m_s
                    > endpoint_limit - VELOCITY_SAFETY_MARGIN_M_S + 1.0e-12
                    or not braking_safe(
                        next_state,
                        transitions,
                        vehicle=self.vehicle,
                        config=self.config,
                    )
                ):
                    reason = "NO_SAFE_NATIVE_ACTION"
                    break
            if next_state.position_m <= cursor.position_m + 1.0e-12:
                reason = "NO_POSITIVE_PROGRESS"
                break
            actions.append(float(action))
            cursor = next_state
            if len(actions) >= MAX_PLANNING_NATIVE_STEPS:
                reason = "PLANNING_STEP_LIMIT"
                break
        feasible = reason is None and cursor.position_m >= waypoint_position_m
        horizon_energy = cursor.total_energy_kwh - state.total_energy_kwh
        duration = cursor.elapsed_time_s - state.elapsed_time_s
        terminal_limit = limit_at(
            min(cursor.position_m, waypoint_position_m),
            current_limit_m_s,
            transitions,
        )
        reference = max(
            0.0,
            min(CAUSAL_CRUISE_CAP_M_S, terminal_limit)
            - VELOCITY_SAFETY_MARGIN_M_S,
        )
        terminal_energy, terminal_calls = _terminal_continuation_energy(
            cursor,
            reference_speed_m_s=reference,
            vehicle=self.vehicle,
            config=self.config,
        )
        calls += terminal_calls
        objective = horizon_energy + terminal_energy + price_kwh_per_s * duration
        if not feasible:
            objective = float("inf")
        return CandidateTrajectory(
            target_speed_m_s=float(target_speed_m_s),
            actions=tuple(actions),
            travel_time_s=float(duration),
            horizon_energy_kwh=float(horizon_energy),
            terminal_energy_kwh=float(terminal_energy),
            objective_kwh=float(objective),
            final_position_m=float(cursor.position_m),
            final_velocity_m_s=float(cursor.velocity_m_s),
            feasible=bool(feasible),
            rejection_reason=reason,
            physics_calls=calls,
        )

    def solve(
        self,
        state: VehicleState,
        *,
        current_limit_m_s: float,
        transitions: tuple[Transition, ...],
        price_kwh_per_s: float,
    ) -> Plan:
        price = float(price_kwh_per_s)
        if not np.isfinite(price) or price < 0.0:
            raise ValueError("time price must be finite and nonnegative")
        waypoint = min(
            self.config.track_length_m,
            state.position_m + self.planning_distance_m,
        )
        if waypoint <= state.position_m:
            raise ValueError("planner requires positive remaining distance")
        future = tuple(
            transition
            for transition in transitions
            if state.position_m < transition.position_m <= waypoint + 1.0e-9
        )
        candidates = tuple(
            self._candidate(
                state,
                target_speed_m_s=speed_km_h / 3.6,
                price_kwh_per_s=price,
                waypoint_position_m=waypoint,
                current_limit_m_s=float(current_limit_m_s),
                transitions=future,
            )
            for speed_km_h in CANDIDATE_SPEEDS_KM_H
        )
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            raise RuntimeError("fixed-distance planner found no feasible candidate")
        selected = min(
            feasible,
            key=lambda candidate: (
                round(candidate.objective_kwh / OBJECTIVE_TIE_TOLERANCE),
                candidate.travel_time_s,
                candidate.horizon_energy_kwh + candidate.terminal_energy_kwh,
                candidate.target_speed_m_s,
            ),
        )
        if not selected.actions:
            raise RuntimeError("selected candidate has no executable action")
        segment = selected.actions[:CONTROL_SEGMENT_NATIVE_STEPS]
        return Plan(
            price_kwh_per_s=price,
            start_position_m=float(state.position_m),
            waypoint_position_m=float(waypoint),
            planning_distance_m=float(waypoint - state.position_m),
            selected=selected,
            candidates=candidates,
            segment_actions=segment,
            physics_calls=sum(candidate.physics_calls for candidate in candidates),
        )
