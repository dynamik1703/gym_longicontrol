"""CONTINUE_FEEDBACK adapter retaining every native safety input and result."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from benchmarks.causal_mpc_v2.fixed_price import normalized_action_to_lambda
from benchmarks.causal_mpc_v2.planner import _mode
from benchmarks.hybrid_rl_design.controller import (
    _endpoint_limit,
    _unknown_preview_safe,
)
from benchmarks.hybrid_rl_design.viability import pessimistic_completion_bound
from benchmarks.physical_feasibility_reference.causal_controller import (
    decode_observation,
)
from benchmarks.physical_feasibility_reference.oracle_controller import (
    VELOCITY_SAFETY_MARGIN_M_S,
    braking_safe,
)
from benchmarks.temporal_options_v1.controller import (
    OptionArm,
    TemporalOptionController,
)
from benchmarks.temporal_options_v1.option_selector import OptionSelection
from benchmarks.temporal_options_v1.option_state import TerminationReason
from gym_longicontrol.domain.dynamics import advance


@dataclass(frozen=True)
class InstrumentedDecision:
    native_step: int
    decision_boundary: bool
    high_level_action: str | None
    natural_high_level_action: str | None
    action_injected: bool
    option_instance_id: int | None
    profile_id: str | None
    family: str | None
    phase: str
    progress: float
    fallback_action: float
    proposed_native_action: float
    executed_action: float
    accepted: bool
    proposal_source: str
    intervention_reason: str
    option_termination: str | None
    physics_prediction_calls: int
    replanned: bool
    action_mode: str
    completion_bound_s: float
    deadline_viable: bool
    endpoint_speed_safe: bool
    known_transition_braking_safe: bool
    unknown_preview_safe: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class InstrumentedContinueController(TemporalOptionController):
    """Frozen controller plus one-shot high-level injection and full telemetry."""

    def __init__(self) -> None:
        super().__init__(OptionArm.CONTINUE_FEEDBACK)
        self._forced_high_level_action: str | None = None
        self._force_pending = False
        self._last_natural_high_level_action: str | None = None
        self._last_action_injected = False

    def reset(self) -> None:
        super().reset()
        self._forced_high_level_action = None
        self._force_pending = False
        self._last_natural_high_level_action = None
        self._last_action_injected = False

    def force_next_high_level_action(self, action_id: str) -> None:
        if self._force_pending:
            raise RuntimeError("a forced high-level action is already pending")
        self._forced_high_level_action = str(action_id)
        self._force_pending = True

    def _select(self, current_limit_m_s: float) -> tuple[str, int]:
        selection = self.selector.solve(
            self.state,
            active_option=self.active_option,
            allow_continue=True,
            next_instance_id=self._next_instance_id,
            current_limit_m_s=current_limit_m_s,
            transitions=self._transitions(),
            price_kwh_per_s=self.current_price_kwh_per_s,
        )
        natural = selection.selected.high_level_action
        selected = selection.selected
        injected = False
        if self._force_pending:
            requested = self._forced_high_level_action
            matches = [
                candidate
                for candidate in selection.candidates
                if candidate.high_level_action == requested and candidate.feasible
            ]
            if len(matches) != 1:
                raise ValueError("injected action is not a unique valid candidate")
            selected = matches[0]
            injected = requested != natural
            self._force_pending = False
            self._forced_high_level_action = None
        self._last_natural_high_level_action = natural
        self._last_action_injected = injected
        self.last_selection = OptionSelection(
            price_kwh_per_s=selection.price_kwh_per_s,
            selected=selected,
            candidates=selection.candidates,
            physics_calls=selection.physics_calls,
        )
        action = selected.high_level_action
        if action == "CONTINUE":
            if self.active_option is None:
                raise RuntimeError("selector continued without an active option")
        else:
            profile_id = action.removeprefix("START(").removesuffix(")")
            if self.active_option is not None:
                self._finish_active(TerminationReason.STRATEGIC_SWITCH)
            self._start(profile_id)
        return action, selection.physics_calls

    def _fallback_safety(
        self,
        fallback: float,
        current_limit: float,
    ) -> tuple[float, bool, bool, bool, bool, int]:
        next_state, _, _ = advance(self.state, fallback, self.vehicle, self.config)
        endpoint = _endpoint_limit(current_limit, next_state, self._transitions())
        endpoint_safe = (
            next_state.velocity_m_s <= endpoint - VELOCITY_SAFETY_MARGIN_M_S + 1.0e-12
        )
        known_safe = braking_safe(
            next_state,
            self._transitions(),
            vehicle=self.vehicle,
            config=self.config,
        )
        preview_safe, preview_calls = _unknown_preview_safe(
            next_state, vehicle=self.vehicle, config=self.config
        )
        viability = pessimistic_completion_bound(
            next_state, vehicle=self.vehicle, config=self.config
        )
        return (
            float(viability.completion_bound_s),
            bool(viability.viable),
            bool(endpoint_safe),
            bool(known_safe),
            bool(preview_safe),
            1 + preview_calls + viability.physics_prediction_calls,
        )

    def action(self, observation: np.ndarray, strategy_price_action: float) -> float:
        ordinary = np.asarray(observation, dtype=np.float64)
        fallback, boundary = self._fallback(ordinary)
        decoded = decode_observation(ordinary)
        current_limit = float(decoded["current_limit_m_s"])
        planner_calls = 0
        high_level: str | None = None
        replanned = False
        termination: str | None = None
        natural: str | None = None
        injected = False
        if boundary:
            self.current_price_action = float(strategy_price_action)
            self.current_price_kwh_per_s = normalized_action_to_lambda(
                self.current_price_action
            )
            self._fallback_until_boundary = False
            try:
                high_level, planner_calls = self._select(current_limit)
                natural = self._last_natural_high_level_action
                injected = self._last_action_injected
                self.last_high_level_action = high_level
                replanned = self.last_selection is not None
            except RuntimeError:
                high_level = None
                self._fallback_until_boundary = True
        if self.active_option is None or self._fallback_until_boundary:
            (
                completion_bound,
                deadline_viable,
                endpoint_safe,
                known_safe,
                preview_safe,
                safety_calls,
            ) = self._fallback_safety(fallback, current_limit)
            reason = "FALLBACK_UNTIL_STRATEGIC_BOUNDARY"
            executed = proposal = fallback
            accepted = True
            phase = "fallback"
            progress = 0.0
            instance_id = None
            profile_id = family = None
            calls = planner_calls + safety_calls
            source = "FALLBACK_ONLY"
        else:
            profile = self.profile_by_id[self.active_option.profile_id]
            self.active_option.update_physical_state(
                profile, self.state, count_step=False
            )
            native = self.executor.decide(
                self.active_option,
                profile,
                self.state,
                current_limit_m_s=current_limit,
                transitions=self._transitions(),
                fallback_action=fallback,
            )
            executed = native.executed_action
            accepted = native.accepted
            phase = self.active_option.phase
            progress = self.active_option.progress
            proposal = native.speed_capped_action
            instance_id = self.active_option.instance_id
            profile_id = profile.profile_id
            family = profile.family
            reason = native.intervention_reason
            calls = planner_calls + native.physics_prediction_calls
            completion_bound = native.completion_bound_s
            deadline_viable = native.deadline_viable
            endpoint_safe = native.endpoint_speed_safe
            known_safe = native.known_transition_braking_safe
            preview_safe = native.unknown_preview_safe
            source = "OPTION"
            if not accepted:
                ended = self._finish_active(
                    TerminationReason.SAFETY_INTERRUPTION, detail=reason
                )
                termination = None if ended is None else ended.termination_reason
                self._fallback_until_boundary = True
        self.last_decision = InstrumentedDecision(
            native_step=self.native_step,
            decision_boundary=boundary,
            high_level_action=high_level,
            natural_high_level_action=natural,
            action_injected=injected,
            option_instance_id=instance_id,
            profile_id=profile_id,
            family=family,
            phase=phase,
            progress=float(progress),
            fallback_action=float(fallback),
            proposed_native_action=float(proposal),
            executed_action=float(executed),
            accepted=accepted,
            proposal_source=source,
            intervention_reason=reason,
            option_termination=termination,
            physics_prediction_calls=calls,
            replanned=replanned,
            action_mode=_mode(float(executed)),
            completion_bound_s=float(completion_bound),
            deadline_viable=bool(deadline_viable),
            endpoint_speed_safe=bool(endpoint_safe),
            known_transition_braking_safe=bool(known_safe),
            unknown_preview_safe=bool(preview_safe),
        )
        return float(executed)

    def snapshot(self) -> dict[str, Any]:
        payload = super().snapshot()
        payload["last_decision"] = (
            None if self.last_decision is None else self.last_decision.as_dict()
        )
        payload.update(
            {
                "instrumentation_schema_version": 1,
                "forced_high_level_action": self._forced_high_level_action,
                "force_pending": self._force_pending,
                "last_natural_high_level_action": self._last_natural_high_level_action,
                "last_action_injected": self._last_action_injected,
            }
        )
        return payload

    def restore(self, payload: dict[str, Any]) -> None:
        if payload.get("instrumentation_schema_version") != 1:
            raise ValueError("unsupported instrumented controller snapshot")
        saved = dict(payload)
        last = saved["last_decision"]
        saved["last_decision"] = None
        super().restore(saved)
        self.last_decision = None if last is None else InstrumentedDecision(**last)
        self._forced_high_level_action = payload["forced_high_level_action"]
        self._force_pending = bool(payload["force_pending"])
        self._last_natural_high_level_action = payload["last_natural_high_level_action"]
        self._last_action_injected = bool(payload["last_action_injected"])
