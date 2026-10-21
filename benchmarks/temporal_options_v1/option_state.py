"""Persistent state and phase semantics for physical driving options."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from benchmarks.causal_mpc_v2.planner import MAX_PLANNING_NATIVE_STEPS, Profile
from gym_longicontrol.domain.state import VehicleState


class TerminationReason(StrEnum):
    NONE = "NONE"
    WAYPOINT_REACHED = "WAYPOINT_REACHED"
    STRATEGIC_SWITCH = "STRATEGIC_SWITCH"
    SAFETY_INTERRUPTION = "SAFETY_INTERRUPTION"
    LIFETIME_TIMEOUT = "LIFETIME_TIMEOUT"
    EPISODE_TERMINATION = "EPISODE_TERMINATION"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"


class OptionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    TERMINATED = "TERMINATED"


@dataclass(frozen=True)
class HighLevelAction:
    """A distinct START or CONTINUE decision.

    START of the active profile intentionally remains different from CONTINUE.
    """

    kind: str
    profile_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"CONTINUE", "START"}:
            raise ValueError("high-level action must be CONTINUE or START")
        if (self.kind == "START") != (self.profile_id is not None):
            raise ValueError("START requires profile_id; CONTINUE forbids it")

    @property
    def action_id(self) -> str:
        return "CONTINUE" if self.kind == "CONTINUE" else f"START({self.profile_id})"


def phase_for(profile: Profile, state: VehicleState, progress: float) -> str:
    """Return the physical phase implied by current state and persistent progress."""

    fraction = float(np.clip(progress, 0.0, 1.0))
    if profile.family == "regulated_cruise":
        return "regulated_feedback"
    if profile.family == "pulse_coast":
        return (
            "pulse_traction"
            if state.velocity_m_s <= profile.terminal_speed_m_s
            or state.velocity_m_s < profile.target_speed_m_s
            else "coast"
        )
    if profile.family == "traction_then_coast":
        return "coast" if fraction >= profile.coast_start_fraction else "traction"
    if profile.family == "traction_coast_brake":
        if fraction >= profile.brake_start_fraction:
            return "brake"
        if fraction >= profile.coast_start_fraction:
            return "coast"
        return "traction"
    if profile.family == "coast_then_partial_traction":
        return (
            "partial_traction" if fraction >= profile.coast_start_fraction else "coast"
        )
    raise ValueError(f"unknown profile family: {profile.family}")


@dataclass
class OptionState:
    instance_id: int
    profile_id: str
    family: str
    start_position_m: float
    start_time_s: float
    target_waypoint_m: float
    original_length_m: float
    progress: float = 0.0
    phase: str = "inactive"
    native_steps_elapsed: int = 0
    last_executed_action: float = 0.0
    status: str = OptionStatus.ACTIVE
    termination_reason: str = TerminationReason.NONE
    termination_detail: str | None = None
    end_position_m: float | None = None
    end_time_s: float | None = None

    @classmethod
    def start(
        cls,
        *,
        instance_id: int,
        profile: Profile,
        state: VehicleState,
        route_length_m: float,
        planning_distance_m: float = 100.0,
    ) -> OptionState:
        waypoint = min(float(route_length_m), state.position_m + planning_distance_m)
        length = waypoint - state.position_m
        if length <= 0.0:
            raise ValueError("option requires positive remaining distance")
        option = cls(
            instance_id=int(instance_id),
            profile_id=profile.profile_id,
            family=profile.family,
            start_position_m=float(state.position_m),
            start_time_s=float(state.elapsed_time_s),
            target_waypoint_m=float(waypoint),
            original_length_m=float(length),
        )
        option.update_physical_state(profile, state, count_step=False)
        return option

    def update_physical_state(
        self,
        profile: Profile,
        state: VehicleState,
        *,
        executed_action: float | None = None,
        count_step: bool,
    ) -> None:
        if self.status != OptionStatus.ACTIVE:
            raise RuntimeError("cannot advance a terminated option")
        self.progress = float(
            np.clip(
                (state.position_m - self.start_position_m) / self.original_length_m,
                0.0,
                1.0,
            )
        )
        self.phase = phase_for(profile, state, self.progress)
        if count_step:
            self.native_steps_elapsed += 1
        if executed_action is not None:
            self.last_executed_action = float(executed_action)

    def terminate(
        self,
        reason: TerminationReason | str,
        state: VehicleState,
        *,
        detail: str | None = None,
    ) -> None:
        if self.status == OptionStatus.TERMINATED:
            raise RuntimeError("option was already terminated")
        self.status = OptionStatus.TERMINATED
        self.termination_reason = str(reason)
        self.termination_detail = detail
        self.end_position_m = float(state.position_m)
        self.end_time_s = float(state.elapsed_time_s)

    @property
    def remaining_distance_m(self) -> float:
        return max(
            0.0,
            self.target_waypoint_m
            - (self.start_position_m + self.progress * self.original_length_m),
        )

    @property
    def duration_s(self) -> float | None:
        if self.end_time_s is None:
            return None
        return self.end_time_s - self.start_time_s

    @property
    def distance_m(self) -> float | None:
        if self.end_position_m is None:
            return None
        return self.end_position_m - self.start_position_m

    @property
    def lifetime_exhausted(self) -> bool:
        return self.native_steps_elapsed >= MAX_PLANNING_NATIVE_STEPS

    def copy(self) -> OptionState:
        return type(self).from_dict(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OptionState:
        return cls(**value)
