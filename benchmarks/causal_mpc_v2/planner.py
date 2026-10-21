"""Deterministic multi-phase fixed-distance causal trajectory optimizer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from benchmarks.hybrid_rl_design.viability import MINIMUM_LIMIT_M_S
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

from .terminal_value import TerminalEstimate, estimate_terminal_value, limit_at

PLANNING_DISTANCE_M = 100.0
CONTROL_SEGMENT_NATIVE_STEPS = 10
MAX_PLANNING_NATIVE_STEPS = 400
OBJECTIVE_TIE_TOLERANCE_KWH = 1.0e-12
SPEED_GRID_KM_H = (20.0, 30.0, 40.0, 50.0, 60.0)


@dataclass(frozen=True)
class Profile:
    profile_id: str
    family: str
    target_speed_m_s: float
    traction_action: float = 1.0
    coast_start_fraction: float = 1.0
    brake_start_fraction: float = 1.0
    brake_action: float = 0.0
    terminal_speed_m_s: float = 0.0


@dataclass(frozen=True)
class CandidateTrajectory:
    profile_id: str
    family: str
    actions: tuple[float, ...]
    travel_time_s: float
    horizon_energy_kwh: float
    terminal_energy_kwh: float
    terminal_time_s: float
    lateness_penalty_kwh: float
    objective_kwh: float
    final_position_m: float
    final_velocity_m_s: float
    feasible: bool
    rejection_reason: str | None
    action_modes_used: tuple[str, ...]
    physics_calls: int
    terminal: TerminalEstimate

    @property
    def total_estimated_energy_kwh(self) -> float:
        return (
            self.horizon_energy_kwh
            + self.terminal_energy_kwh
            + self.lateness_penalty_kwh
        )

    @property
    def total_estimated_time_s(self) -> float:
        return self.travel_time_s + self.terminal_time_s

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


def profile_lattice() -> tuple[Profile, ...]:
    profiles: list[Profile] = []
    for speed in SPEED_GRID_KM_H:
        profiles.append(
            Profile(
                profile_id=f"cruise_{speed:02.0f}",
                family="regulated_cruise",
                target_speed_m_s=speed / 3.6,
            )
        )
    for speed in (30.0, 45.0, 60.0):
        for traction in (0.5, 1.0):
            profiles.append(
                Profile(
                    profile_id=f"pulse_{speed:02.0f}_{traction:.1f}",
                    family="pulse_coast",
                    target_speed_m_s=speed / 3.6,
                    traction_action=traction,
                    terminal_speed_m_s=(speed - 5.0) / 3.6,
                )
            )
    for speed in (40.0, 60.0):
        profiles.append(
            Profile(
                profile_id=f"accel_coast_{speed:02.0f}_0.5",
                family="traction_then_coast",
                target_speed_m_s=speed / 3.6,
                traction_action=1.0,
                coast_start_fraction=0.5,
            )
        )
    for brake in (-0.5, -1.0):
        profiles.append(
            Profile(
                profile_id=f"accel_coast_brake_50_{brake:.1f}",
                family="traction_coast_brake",
                target_speed_m_s=50.0 / 3.6,
                traction_action=1.0,
                coast_start_fraction=0.4,
                brake_start_fraction=0.82,
                brake_action=brake,
                terminal_speed_m_s=30.0 / 3.6,
            )
        )
    for speed in (35.0, 50.0):
        profiles.append(
            Profile(
                profile_id=f"coast_partial_{speed:02.0f}",
                family="coast_then_partial_traction",
                target_speed_m_s=speed / 3.6,
                traction_action=0.5,
                coast_start_fraction=0.3,
            )
        )
    return tuple(profiles)


PROFILES = profile_lattice()


def _mode(action: float) -> str:
    if action >= 0.95:
        return "positive_traction"
    if action > 1.0e-9:
        return "partial_traction"
    if action >= -1.0e-9:
        return "coasting"
    if action > -0.95:
        return "moderate_regenerative_braking"
    return "strong_braking"


class FixedDistanceProfilePlanner:
    """Enumerate a fixed lambda-independent lattice of multi-phase profiles."""

    def __init__(
        self,
        *,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
        planning_distance_m: float = PLANNING_DISTANCE_M,
        profiles: tuple[Profile, ...] = PROFILES,
    ) -> None:
        if not 0.0 < planning_distance_m <= config.sensor_range_m:
            raise ValueError("planning distance must be positive and within preview")
        if not profiles:
            raise ValueError("profile lattice must not be empty")
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        self.planning_distance_m = float(planning_distance_m)
        self.profiles = tuple(profiles)

    def _raw_action(
        self,
        profile: Profile,
        state: VehicleState,
        fraction: float,
    ) -> float:
        target = profile.target_speed_m_s
        if profile.family == "regulated_cruise":
            return action_for_next_velocity(
                state, target, vehicle=self.vehicle, config=self.config
            )
        if profile.family == "pulse_coast":
            if state.velocity_m_s <= profile.terminal_speed_m_s:
                return profile.traction_action
            if state.velocity_m_s >= target:
                return 0.0
            return profile.traction_action
        if profile.family == "traction_then_coast":
            if fraction >= profile.coast_start_fraction:
                if state.velocity_m_s <= MINIMUM_LIMIT_M_S:
                    return action_for_next_velocity(
                        state,
                        MINIMUM_LIMIT_M_S,
                        vehicle=self.vehicle,
                        config=self.config,
                    )
                return 0.0
            if state.velocity_m_s < target:
                return profile.traction_action
            return 0.0
        if profile.family == "traction_coast_brake":
            if fraction >= profile.brake_start_fraction:
                return (
                    profile.brake_action
                    if state.velocity_m_s > profile.terminal_speed_m_s
                    else action_for_next_velocity(
                        state,
                        profile.terminal_speed_m_s,
                        vehicle=self.vehicle,
                        config=self.config,
                    )
                )
            if fraction >= profile.coast_start_fraction:
                return 0.0
            if state.velocity_m_s < target:
                return profile.traction_action
            return 0.0
        if profile.family == "coast_then_partial_traction":
            if fraction < profile.coast_start_fraction:
                return 0.0
            if state.velocity_m_s < target:
                return profile.traction_action
            return action_for_next_velocity(
                state, target, vehicle=self.vehicle, config=self.config
            )
        raise ValueError(f"unknown profile family: {profile.family}")

    def _candidate(
        self,
        state: VehicleState,
        *,
        profile: Profile,
        price_kwh_per_s: float,
        waypoint_position_m: float,
        current_limit_m_s: float,
        transitions: tuple[Transition, ...],
    ) -> CandidateTrajectory:
        cursor = state
        actions: list[float] = []
        modes: set[str] = set()
        calls = 0
        reason: str | None = None
        planning_distance = waypoint_position_m - state.position_m
        while cursor.position_m < waypoint_position_m - 1.0e-12:
            fraction = np.clip(
                (cursor.position_m - state.position_m) / planning_distance,
                0.0,
                1.0,
            )
            local_limit = limit_at(
                cursor.position_m, current_limit_m_s, transitions
            )
            speed_cap = max(
                0.0,
                min(CAUSAL_CRUISE_CAP_M_S, local_limit)
                - VELOCITY_SAFETY_MARGIN_M_S,
            )
            action = float(self._raw_action(profile, cursor, float(fraction)))
            tentative, _, _ = advance(
                cursor, action, self.vehicle, self.config
            )
            calls += 1
            if tentative.velocity_m_s > speed_cap + 1.0e-12:
                action = action_for_next_velocity(
                    cursor,
                    speed_cap,
                    vehicle=self.vehicle,
                    config=self.config,
                )
                tentative, _, _ = advance(
                    cursor, action, self.vehicle, self.config
                )
                calls += 1
            endpoint_limit = limit_at(
                tentative.position_m, current_limit_m_s, transitions
            )
            endpoint_safe = (
                tentative.velocity_m_s
                <= endpoint_limit - VELOCITY_SAFETY_MARGIN_M_S + 1.0e-12
            )
            known_safe = braking_safe(
                tentative,
                transitions,
                vehicle=self.vehicle,
                config=self.config,
            )
            if not endpoint_safe or not known_safe:
                action = -1.0
                tentative, _, _ = advance(
                    cursor, action, self.vehicle, self.config
                )
                calls += 1
                endpoint_limit = limit_at(
                    tentative.position_m, current_limit_m_s, transitions
                )
                if (
                    tentative.velocity_m_s
                    > endpoint_limit - VELOCITY_SAFETY_MARGIN_M_S + 1.0e-12
                    or not braking_safe(
                        tentative,
                        transitions,
                        vehicle=self.vehicle,
                        config=self.config,
                    )
                ):
                    reason = "NO_SAFE_NATIVE_ACTION"
                    break
            if tentative.position_m <= cursor.position_m + 1.0e-12:
                reason = "NO_POSITIVE_PROGRESS"
                break
            if len(actions) >= 10 and tentative.velocity_m_s < 0.25 and action <= 0:
                reason = "NO_USEFUL_PROGRESS"
                break
            actions.append(float(action))
            modes.add(_mode(float(action)))
            cursor = tentative
            if len(actions) >= MAX_PLANNING_NATIVE_STEPS:
                reason = "PLANNING_STEP_LIMIT"
                break
        feasible = reason is None and cursor.position_m >= waypoint_position_m
        horizon_energy = cursor.total_energy_kwh - state.total_energy_kwh
        duration = cursor.elapsed_time_s - state.elapsed_time_s
        terminal = estimate_terminal_value(
            cursor,
            current_limit_m_s=limit_at(
                cursor.position_m, current_limit_m_s, transitions
            ),
            transitions=tuple(
                transition
                for transition in transitions
                if transition.position_m > cursor.position_m
            ),
            vehicle=self.vehicle,
            config=self.config,
        )
        calls += terminal.physics_calls
        energy = horizon_energy + terminal.total_energy_kwh
        total_time = duration + terminal.time_s
        objective = energy + price_kwh_per_s * total_time
        if not feasible:
            objective = float("inf")
        return CandidateTrajectory(
            profile_id=profile.profile_id,
            family=profile.family,
            actions=tuple(actions),
            travel_time_s=float(duration),
            horizon_energy_kwh=float(horizon_energy),
            terminal_energy_kwh=float(terminal.energy_kwh),
            terminal_time_s=float(terminal.time_s),
            lateness_penalty_kwh=float(terminal.lateness_penalty_kwh),
            objective_kwh=float(objective),
            final_position_m=float(cursor.position_m),
            final_velocity_m_s=float(cursor.velocity_m_s),
            feasible=bool(feasible),
            rejection_reason=reason,
            action_modes_used=tuple(sorted(modes)),
            physics_calls=calls,
            terminal=terminal,
        )

    def candidate_set(
        self,
        state: VehicleState,
        *,
        current_limit_m_s: float,
        transitions: tuple[Transition, ...],
        price_kwh_per_s: float,
    ) -> tuple[CandidateTrajectory, ...]:
        waypoint = min(
            self.config.track_length_m,
            state.position_m + self.planning_distance_m,
        )
        future = tuple(
            transition
            for transition in transitions
            if state.position_m < transition.position_m
        )
        return tuple(
            self._candidate(
                state,
                profile=profile,
                price_kwh_per_s=price_kwh_per_s,
                waypoint_position_m=waypoint,
                current_limit_m_s=current_limit_m_s,
                transitions=future,
            )
            for profile in self.profiles
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
        candidates = self.candidate_set(
            state,
            current_limit_m_s=current_limit_m_s,
            transitions=transitions,
            price_kwh_per_s=price,
        )
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            raise RuntimeError("profile planner found no feasible candidate")
        selected = min(
            feasible,
            key=lambda candidate: (
                round(
                    candidate.objective_kwh / OBJECTIVE_TIE_TOLERANCE_KWH
                ),
                candidate.total_estimated_time_s,
                candidate.total_estimated_energy_kwh,
                candidate.profile_id,
            ),
        )
        if not selected.actions:
            raise RuntimeError("selected candidate has no executable action")
        return Plan(
            price_kwh_per_s=price,
            start_position_m=float(state.position_m),
            waypoint_position_m=float(waypoint),
            planning_distance_m=float(waypoint - state.position_m),
            selected=selected,
            candidates=candidates,
            segment_actions=selected.actions[:CONTROL_SEGMENT_NATIVE_STEPS],
            physics_calls=sum(candidate.physics_calls for candidate in candidates),
        )
