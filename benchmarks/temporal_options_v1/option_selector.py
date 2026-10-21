"""Matched scoring of START and CONTINUE under persistent option semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.causal_mpc_v2.planner import (
    MAX_PLANNING_NATIVE_STEPS,
    OBJECTIVE_TIE_TOLERANCE_KWH,
    PROFILES,
    Profile,
)
from benchmarks.causal_mpc_v2.terminal_value import (
    TerminalEstimate,
    estimate_terminal_value,
    limit_at,
)
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .feedback_executor import NativeFeedbackExecutor, frozen_fallback_action
from .option_state import HighLevelAction, OptionState


@dataclass(frozen=True)
class OptionCandidate:
    high_level_action: str
    profile_id: str
    family: str
    instance: OptionState
    actions: tuple[float, ...]
    phases: tuple[str, ...]
    future_option_energy_kwh: float
    future_option_time_s: float
    terminal_energy_kwh: float
    terminal_time_s: float
    lateness_penalty_kwh: float
    objective_kwh: float
    final_position_m: float
    final_velocity_m_s: float
    outcome: str
    interruption_reason: str | None
    feasible: bool
    physics_calls: int
    terminal: TerminalEstimate

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptionSelection:
    price_kwh_per_s: float
    selected: OptionCandidate
    candidates: tuple[OptionCandidate, ...]
    physics_calls: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class StatefulOptionSelector:
    """Score actual remaining suffixes and freshly anchored START candidates."""

    def __init__(
        self,
        *,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
        profiles: tuple[Profile, ...] = PROFILES,
    ) -> None:
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        self.profiles = tuple(profiles)
        self.profile_by_id = {profile.profile_id: profile for profile in profiles}
        if len(self.profile_by_id) != len(self.profiles):
            raise ValueError("profile identifiers must be unique")
        self.executor = NativeFeedbackExecutor(vehicle=self.vehicle, config=self.config)

    def _candidate(
        self,
        state: VehicleState,
        *,
        action: HighLevelAction,
        active_option: OptionState | None,
        instance_id: int,
        current_limit_m_s: float,
        transitions: tuple[Transition, ...],
        price_kwh_per_s: float,
    ) -> OptionCandidate:
        if action.kind == "CONTINUE":
            if active_option is None:
                raise ValueError("CONTINUE requires an active option")
            option = active_option.copy()
            profile = self.profile_by_id[option.profile_id]
        else:
            assert action.profile_id is not None
            profile = self.profile_by_id[action.profile_id]
            option = OptionState.start(
                instance_id=instance_id,
                profile=profile,
                state=state,
                route_length_m=self.config.track_length_m,
            )
        cursor = state
        actions: list[float] = []
        phases: list[str] = []
        calls = 0
        outcome = "WAYPOINT_REACHED"
        interruption: str | None = None
        remaining_steps = MAX_PLANNING_NATIVE_STEPS - option.native_steps_elapsed
        if remaining_steps <= 0:
            outcome = "LIFETIME_TIMEOUT"
        for _ in range(max(0, remaining_steps)):
            if cursor.position_m >= option.target_waypoint_m - 1.0e-12:
                break
            local_limit = limit_at(cursor.position_m, current_limit_m_s, transitions)
            fallback = frozen_fallback_action(
                cursor,
                current_limit_m_s=local_limit,
                transitions=transitions,
                vehicle=self.vehicle,
                config=self.config,
            )
            option.update_physical_state(profile, cursor, count_step=False)
            decision = self.executor.decide(
                option,
                profile,
                cursor,
                current_limit_m_s=local_limit,
                transitions=transitions,
                fallback_action=fallback,
            )
            calls += decision.physics_prediction_calls
            next_state, _, _ = advance(
                cursor, decision.executed_action, self.vehicle, self.config
            )
            calls += 1
            actions.append(decision.executed_action)
            phases.append(option.phase)
            option.update_physical_state(
                profile,
                next_state,
                executed_action=decision.executed_action,
                count_step=True,
            )
            cursor = next_state
            if not decision.accepted:
                outcome = "SAFETY_INTERRUPTION"
                interruption = decision.intervention_reason
                break
            if option.native_steps_elapsed >= MAX_PLANNING_NATIVE_STEPS:
                outcome = "LIFETIME_TIMEOUT"
                break
        else:
            if cursor.position_m < option.target_waypoint_m - 1.0e-12:
                outcome = "LIFETIME_TIMEOUT"
        if cursor.position_m >= option.target_waypoint_m - 1.0e-12:
            outcome = "WAYPOINT_REACHED"
        duration = cursor.elapsed_time_s - state.elapsed_time_s
        energy = cursor.total_energy_kwh - state.total_energy_kwh
        current_at_terminal = limit_at(
            cursor.position_m, current_limit_m_s, transitions
        )
        future = tuple(t for t in transitions if t.position_m > cursor.position_m)
        terminal = estimate_terminal_value(
            cursor,
            current_limit_m_s=current_at_terminal,
            transitions=future,
            vehicle=self.vehicle,
            config=self.config,
        )
        calls += terminal.physics_calls
        objective = (
            energy
            + terminal.energy_kwh
            + terminal.lateness_penalty_kwh
            + price_kwh_per_s * (duration + terminal.time_s)
        )
        feasible = bool(actions) and all(
            value == value and abs(value) != float("inf")
            for value in (energy, duration, objective)
        )
        if not feasible:
            objective = float("inf")
        return OptionCandidate(
            high_level_action=action.action_id,
            profile_id=profile.profile_id,
            family=profile.family,
            instance=option,
            actions=tuple(actions),
            phases=tuple(phases),
            future_option_energy_kwh=float(energy),
            future_option_time_s=float(duration),
            terminal_energy_kwh=float(terminal.energy_kwh),
            terminal_time_s=float(terminal.time_s),
            lateness_penalty_kwh=float(terminal.lateness_penalty_kwh),
            objective_kwh=float(objective),
            final_position_m=float(cursor.position_m),
            final_velocity_m_s=float(cursor.velocity_m_s),
            outcome=outcome,
            interruption_reason=interruption,
            feasible=feasible,
            physics_calls=calls,
            terminal=terminal,
        )

    def solve(
        self,
        state: VehicleState,
        *,
        active_option: OptionState | None,
        allow_continue: bool,
        next_instance_id: int,
        current_limit_m_s: float,
        transitions: tuple[Transition, ...],
        price_kwh_per_s: float,
    ) -> OptionSelection:
        actions: list[HighLevelAction] = []
        if allow_continue and active_option is not None:
            actions.append(HighLevelAction("CONTINUE"))
        actions.extend(
            HighLevelAction("START", profile.profile_id) for profile in self.profiles
        )
        candidates = tuple(
            self._candidate(
                state,
                action=action,
                active_option=active_option,
                instance_id=next_instance_id,
                current_limit_m_s=current_limit_m_s,
                transitions=transitions,
                price_kwh_per_s=float(price_kwh_per_s),
            )
            for action in actions
        )
        feasible = [candidate for candidate in candidates if candidate.feasible]
        if not feasible:
            raise RuntimeError("option selector found no feasible candidate")
        best_value = min(candidate.objective_kwh for candidate in feasible)
        tied = [
            candidate
            for candidate in feasible
            if abs(candidate.objective_kwh - best_value) <= OBJECTIVE_TIE_TOLERANCE_KWH
        ]
        continuation = next(
            (
                candidate
                for candidate in tied
                if candidate.high_level_action == "CONTINUE"
            ),
            None,
        )
        selected = continuation or min(
            tied,
            key=lambda candidate: (
                candidate.future_option_time_s + candidate.terminal_time_s,
                candidate.future_option_energy_kwh
                + candidate.terminal_energy_kwh
                + candidate.lateness_penalty_kwh,
                candidate.high_level_action,
            ),
        )
        return OptionSelection(
            price_kwh_per_s=float(price_kwh_per_s),
            selected=selected,
            candidates=candidates,
            physics_calls=sum(candidate.physics_calls for candidate in candidates),
        )
