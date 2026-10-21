"""Closed-loop RESET, CONTINUE and COMMIT option controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from benchmarks.causal_mpc_v2.controller import PriceMPCController, _ExternalFallback
from benchmarks.causal_mpc_v2.fixed_price import normalized_action_to_lambda
from benchmarks.causal_mpc_v2.planner import PROFILES, Profile, _mode
from benchmarks.hybrid_rl_design.viability import (
    DEADLINE_S,
    pessimistic_completion_bound,
)
from benchmarks.physical_feasibility_reference.causal_controller import (
    decode_observation,
)
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .feedback_executor import NativeFeedbackExecutor
from .observation_adapter import OptionObservation, augment_observation
from .option_selector import OptionSelection, StatefulOptionSelector
from .option_state import OptionState, TerminationReason


class OptionArm(StrEnum):
    RESET_FEEDBACK = "RESET_FEEDBACK"
    CONTINUE_FEEDBACK = "CONTINUE_FEEDBACK"
    COMMIT_FEEDBACK = "COMMIT_FEEDBACK"


@dataclass(frozen=True)
class TemporalDecision:
    native_step: int
    decision_boundary: bool
    high_level_action: str | None
    option_instance_id: int | None
    profile_id: str | None
    family: str | None
    phase: str
    progress: float
    fallback_action: float
    proposed_native_action: float
    executed_action: float
    accepted: bool
    intervention_reason: str
    option_termination: str | None
    physics_prediction_calls: int
    replanned: bool
    action_mode: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TemporalOptionController:
    """Preserve anchors while retaining the frozen causal fallback and Shield."""

    def __init__(
        self,
        arm: OptionArm | str,
        *,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
        profiles: tuple[Profile, ...] = PROFILES,
    ) -> None:
        self.arm = OptionArm(arm)
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        self.reference = _ExternalFallback(vehicle=self.vehicle, config=self.config)
        self.profiles = tuple(profiles)
        self.profile_by_id = {profile.profile_id: profile for profile in profiles}
        self.executor = NativeFeedbackExecutor(vehicle=self.vehicle, config=self.config)
        self.selector = StatefulOptionSelector(
            vehicle=self.vehicle, config=self.config, profiles=self.profiles
        )
        self.reset()

    @property
    def state(self) -> VehicleState:
        return self.reference.state

    def reset(self) -> None:
        self.reference.reset()
        self.active_option: OptionState | None = None
        self.option_history: list[OptionState] = []
        self.previous_termination = TerminationReason.NONE.value
        self.last_decision: TemporalDecision | None = None
        self.last_selection: OptionSelection | None = None
        self.last_high_level_action: str | None = None
        self.current_price_action = 1.0
        self.current_price_kwh_per_s = normalized_action_to_lambda(1.0)
        self.last_executed_action = 0.0
        self._pending_observation: np.ndarray | None = None
        self._pending_fallback: float | None = None
        self._fallback_until_boundary = False
        self._next_instance_id = 1
        self.native_step = 0

    def _transitions(self) -> tuple[Transition, ...]:
        return tuple(
            Transition(position, limit)
            for position, limit in sorted(self.reference._known_transitions.items())
        )

    def _base_strategy_observation(
        self, observation: np.ndarray, fallback_action: float
    ) -> np.ndarray:
        ordinary = np.asarray(observation, dtype=np.float64)
        viability = pessimistic_completion_bound(
            self.state, vehicle=self.vehicle, config=self.config
        )
        remaining_distance = max(
            0.0, self.config.track_length_m - self.state.position_m
        )
        remaining_time = max(0.0, DEADLINE_S - self.state.elapsed_time_s)
        slack = DEADLINE_S - viability.completion_bound_s
        memory = np.array(
            [
                self.state.position_m / self.config.track_length_m,
                self.state.elapsed_time_s / DEADLINE_S,
                remaining_distance / self.config.track_length_m,
                remaining_time / DEADLINE_S,
                self.last_executed_action,
                self.current_price_action,
                np.clip(slack / DEADLINE_S, -1.0, 1.0),
                float(fallback_action),
            ],
            dtype=np.float64,
        )
        return np.concatenate((ordinary, memory))

    def prepare_strategy_step(self, observation: np.ndarray) -> np.ndarray:
        if self._pending_fallback is not None:
            raise RuntimeError("previous prepared strategy step was not consumed")
        ordinary = np.asarray(observation, dtype=np.float64)
        fallback = float(self.reference.action(ordinary))
        self._pending_observation = ordinary.copy()
        self._pending_fallback = fallback
        return self._base_strategy_observation(ordinary, fallback)

    def rl_observation(
        self, base_strategy_observation: np.ndarray
    ) -> OptionObservation:
        return augment_observation(
            base_strategy_observation,
            active_option=self.active_option,
            previous_termination=self.previous_termination,
        )

    def _fallback(self, ordinary: np.ndarray) -> tuple[float, bool]:
        boundary = self._pending_fallback is not None
        if boundary:
            if self._pending_observation is None or not np.array_equal(
                ordinary, self._pending_observation
            ):
                raise RuntimeError("prepared observation does not match action")
            fallback = float(self._pending_fallback)
            self._pending_fallback = None
            self._pending_observation = None
            return fallback, True
        return float(self.reference.action(ordinary)), False

    def _finish_active(
        self,
        reason: TerminationReason,
        *,
        detail: str | None = None,
    ) -> OptionState | None:
        if self.active_option is None:
            return None
        ended = self.active_option
        ended.terminate(reason, self.state, detail=detail)
        self.option_history.append(ended.copy())
        self.previous_termination = reason.value
        self.active_option = None
        return ended

    def _start(self, profile_id: str) -> None:
        profile = self.profile_by_id[profile_id]
        self.active_option = OptionState.start(
            instance_id=self._next_instance_id,
            profile=profile,
            state=self.state,
            route_length_m=self.config.track_length_m,
        )
        self._next_instance_id += 1

    def _select(self, current_limit_m_s: float) -> tuple[str, int]:
        if self.arm == OptionArm.COMMIT_FEEDBACK and self.active_option is not None:
            self.last_selection = None
            return "CONTINUE", 0
        allow_continue = self.arm == OptionArm.CONTINUE_FEEDBACK
        selection = self.selector.solve(
            self.state,
            active_option=self.active_option,
            allow_continue=allow_continue,
            next_instance_id=self._next_instance_id,
            current_limit_m_s=current_limit_m_s,
            transitions=self._transitions(),
            price_kwh_per_s=self.current_price_kwh_per_s,
        )
        self.last_selection = selection
        action = selection.selected.high_level_action
        if action == "CONTINUE":
            if self.active_option is None:
                raise RuntimeError("selector continued without an active option")
        else:
            profile_id = action.removeprefix("START(").removesuffix(")")
            if self.active_option is not None:
                self._finish_active(TerminationReason.STRATEGIC_SWITCH)
            self._start(profile_id)
        return action, selection.physics_calls

    def action(self, observation: np.ndarray, strategy_price_action: float) -> float:
        ordinary = np.asarray(observation, dtype=np.float64)
        fallback, boundary = self._fallback(ordinary)
        decoded = decode_observation(ordinary)
        current_limit = float(decoded["current_limit_m_s"])
        planner_calls = 0
        high_level: str | None = None
        replanned = False
        termination: str | None = None
        if boundary:
            self.current_price_action = float(strategy_price_action)
            self.current_price_kwh_per_s = normalized_action_to_lambda(
                self.current_price_action
            )
            self._fallback_until_boundary = False
            try:
                high_level, planner_calls = self._select(current_limit)
                self.last_high_level_action = high_level
                replanned = self.last_selection is not None
            except RuntimeError:
                high_level = None
                self._fallback_until_boundary = True
        if self.active_option is None or self._fallback_until_boundary:
            reason = "FALLBACK_UNTIL_STRATEGIC_BOUNDARY"
            decision = None
            executed = fallback
            accepted = True
            phase = "fallback"
            progress = 0.0
            proposal = fallback
            instance_id = None
            profile_id = family = None
            calls = planner_calls
        else:
            profile = self.profile_by_id[self.active_option.profile_id]
            self.active_option.update_physical_state(
                profile, self.state, count_step=False
            )
            decision = self.executor.decide(
                self.active_option,
                profile,
                self.state,
                current_limit_m_s=current_limit,
                transitions=self._transitions(),
                fallback_action=fallback,
            )
            executed = decision.executed_action
            accepted = decision.accepted
            phase = self.active_option.phase
            progress = self.active_option.progress
            proposal = decision.speed_capped_action
            instance_id = self.active_option.instance_id
            profile_id = profile.profile_id
            family = profile.family
            reason = decision.intervention_reason
            calls = planner_calls + decision.physics_prediction_calls
            if not accepted:
                ended = self._finish_active(
                    TerminationReason.SAFETY_INTERRUPTION, detail=reason
                )
                termination = None if ended is None else ended.termination_reason
                self._fallback_until_boundary = True
        self.last_decision = TemporalDecision(
            native_step=self.native_step,
            decision_boundary=boundary,
            high_level_action=high_level,
            option_instance_id=instance_id,
            profile_id=profile_id,
            family=family,
            phase=phase,
            progress=float(progress),
            fallback_action=fallback,
            proposed_native_action=float(proposal),
            executed_action=float(executed),
            accepted=accepted,
            intervention_reason=reason,
            option_termination=termination,
            physics_prediction_calls=calls,
            replanned=replanned,
            action_mode=_mode(float(executed)),
        )
        return float(executed)

    def observe_executed_action(
        self, action: float, *, episode_ended: bool = False
    ) -> None:
        value = float(action)
        if self.last_decision is None or value != self.last_decision.executed_action:
            raise RuntimeError("environment action differs from option decision")
        self.reference.observe_external_action(value)
        self.last_executed_action = value
        self.native_step += 1
        if self.active_option is not None:
            profile = self.profile_by_id[self.active_option.profile_id]
            self.active_option.update_physical_state(
                profile,
                self.state,
                executed_action=value,
                count_step=True,
            )
            if episode_ended:
                self._finish_active(TerminationReason.EPISODE_TERMINATION)
            elif (
                self.state.position_m >= self.active_option.target_waypoint_m - 1.0e-12
            ):
                self._finish_active(TerminationReason.WAYPOINT_REACHED)
                self._fallback_until_boundary = True
            elif self.active_option.lifetime_exhausted:
                self._finish_active(TerminationReason.LIFETIME_TIMEOUT)
                self._fallback_until_boundary = True

    def snapshot(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "reference_state": asdict(self.reference.state),
            "known_transitions": dict(self.reference._known_transitions),
            "reference_last_action": self.reference._last_action,
            "active_option": (
                None if self.active_option is None else self.active_option.as_dict()
            ),
            "option_history": [option.as_dict() for option in self.option_history],
            "previous_termination": self.previous_termination,
            "last_decision": (
                None if self.last_decision is None else self.last_decision.as_dict()
            ),
            "last_high_level_action": self.last_high_level_action,
            "current_price_action": self.current_price_action,
            "current_price_kwh_per_s": self.current_price_kwh_per_s,
            "last_executed_action": self.last_executed_action,
            "fallback_until_boundary": self._fallback_until_boundary,
            "next_instance_id": self._next_instance_id,
            "native_step": self.native_step,
            "pending_observation": (
                None
                if self._pending_observation is None
                else self._pending_observation.tolist()
            ),
            "pending_fallback": self._pending_fallback,
        }

    def restore(self, payload: dict[str, Any]) -> None:
        if payload["arm"] != self.arm.value:
            raise ValueError("checkpoint arm mismatch")
        self.reference.state = VehicleState(**payload["reference_state"])
        self.reference._known_transitions = {
            float(key): float(value)
            for key, value in payload["known_transitions"].items()
        }
        self.reference._last_action = payload["reference_last_action"]
        self.active_option = (
            None
            if payload["active_option"] is None
            else OptionState.from_dict(payload["active_option"])
        )
        self.option_history = [
            OptionState.from_dict(value) for value in payload["option_history"]
        ]
        self.previous_termination = payload["previous_termination"]
        self.last_decision = (
            None
            if payload["last_decision"] is None
            else TemporalDecision(**payload["last_decision"])
        )
        self.last_selection = None
        self.last_high_level_action = payload["last_high_level_action"]
        self.current_price_action = float(payload["current_price_action"])
        self.current_price_kwh_per_s = float(payload["current_price_kwh_per_s"])
        self.last_executed_action = float(payload["last_executed_action"])
        self._fallback_until_boundary = bool(payload["fallback_until_boundary"])
        self._next_instance_id = int(payload["next_instance_id"])
        self.native_step = int(payload["native_step"])
        self._pending_observation = (
            None
            if payload["pending_observation"] is None
            else np.asarray(payload["pending_observation"], dtype=np.float64)
        )
        self._pending_fallback = payload["pending_fallback"]


def build_controller(arm: str) -> PriceMPCController | TemporalOptionController:
    """Explicit factory keeping the historical arm byte-for-byte inherited."""

    if arm == "FROZEN_MPC_V2":
        return PriceMPCController()
    return TemporalOptionController(arm)
