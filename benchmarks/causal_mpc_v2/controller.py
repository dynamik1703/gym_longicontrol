"""Shared causal V2 profile-MPC controller with the frozen physical Shield."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from benchmarks.hybrid_rl_design.controller import (
    _endpoint_limit,
    _unknown_preview_safe,
)
from benchmarks.hybrid_rl_design.viability import (
    DEADLINE_S,
    pessimistic_completion_bound,
)
from benchmarks.physical_feasibility_reference.causal_controller import (
    ObservationLimitedController,
    decode_observation,
)
from benchmarks.physical_feasibility_reference.oracle_controller import (
    VELOCITY_SAFETY_MARGIN_M_S,
    Transition,
    braking_safe,
)
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .fixed_price import normalized_action_to_lambda
from .planner import FixedDistanceProfilePlanner, Plan

STRATEGY_OBSERVATION_DIMENSIONS = 16


@dataclass(frozen=True)
class PriceShieldDecision:
    fallback_action: float
    proposed_native_action: float
    executed_action: float
    normalized_price_action: float
    price_kwh_per_s: float
    planner_profile_id: str | None
    planner_family: str | None
    planner_terminal_energy_kwh: float | None
    planner_terminal_time_s: float | None
    accepted: bool
    overridden: bool
    intervention_reason: str
    endpoint_speed_safe: bool
    known_transition_braking_safe: bool
    unknown_preview_safe: bool
    deadline_viable: bool
    completion_bound_s: float | None
    native_action_influence: float
    physics_prediction_calls: int
    replanned: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ExternalFallback(ObservationLimitedController):
    def observe_external_action(self, action: float) -> None:
        self._last_action = float(action)
        super().observe_executed_action(float(action))


class PriceMPCController:
    """Convert a causal scalar price into shielded exact-dynamics controls."""

    information = (
        "ordinary 8D observation; scalar normalized time price; exact known "
        "dynamics; propagated controller state/time; remembered visible signs; "
        "fixed deadline and stochastic-generator bounds"
    )

    def __init__(
        self,
        *,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
        planner: FixedDistanceProfilePlanner | None = None,
    ) -> None:
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        self.reference = _ExternalFallback(vehicle=self.vehicle, config=self.config)
        self.planner = planner or FixedDistanceProfilePlanner(
            vehicle=self.vehicle, config=self.config
        )
        self.last_decision: PriceShieldDecision | None = None
        self.last_plan: Plan | None = None
        self.current_price_action = 1.0
        self.current_price_kwh_per_s = normalized_action_to_lambda(1.0)
        self.last_executed_action = 0.0
        self._segment_actions: list[float] = []
        self._planned_transition_signature: tuple[tuple[float, float], ...] = ()
        self._pending_observation: np.ndarray | None = None
        self._pending_fallback: float | None = None
        self._fallback_until_boundary = False

    @property
    def state(self) -> VehicleState:
        return self.reference.state

    def reset(self) -> None:
        self.reference.reset()
        self.last_decision = None
        self.last_plan = None
        self.current_price_action = 1.0
        self.current_price_kwh_per_s = normalized_action_to_lambda(1.0)
        self.last_executed_action = 0.0
        self._segment_actions.clear()
        self._planned_transition_signature = ()
        self._pending_observation = None
        self._pending_fallback = None
        self._fallback_until_boundary = False

    def _transitions(self) -> tuple[Transition, ...]:
        return tuple(
            Transition(position, limit)
            for position, limit in sorted(self.reference._known_transitions.items())
        )

    def _transition_signature(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (transition.position_m, transition.limit_m_s)
            for transition in self._transitions()
        )

    def strategy_observation(
        self, observation: np.ndarray, fallback_action: float
    ) -> np.ndarray:
        ordinary = np.asarray(observation, dtype=np.float64)
        if ordinary.shape != (8,) or not np.isfinite(ordinary).all():
            raise ValueError("ordinary observation must be a finite 8-vector")
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
        result = np.concatenate((ordinary, memory))
        if result.shape != (STRATEGY_OBSERVATION_DIMENSIONS,):
            raise RuntimeError("strategy observation contract changed")
        return result

    def prepare_strategy_step(self, observation: np.ndarray) -> np.ndarray:
        if self._pending_fallback is not None:
            raise RuntimeError("previous prepared strategy step was not consumed")
        ordinary = np.asarray(observation, dtype=np.float64)
        fallback = float(self.reference.action(ordinary))
        self._pending_observation = ordinary.copy()
        self._pending_fallback = fallback
        return self.strategy_observation(ordinary, fallback)

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
        else:
            fallback = float(self.reference.action(ordinary))
        return fallback, boundary

    def _replan(self, current_limit_m_s: float) -> None:
        self.last_plan = self.planner.solve(
            self.state,
            current_limit_m_s=current_limit_m_s,
            transitions=self._transitions(),
            price_kwh_per_s=self.current_price_kwh_per_s,
        )
        self._segment_actions = list(self.last_plan.segment_actions)
        self._planned_transition_signature = self._transition_signature()

    def action(self, observation: np.ndarray, strategy_action: float) -> float:
        ordinary = np.asarray(observation, dtype=np.float64)
        fallback, decision_boundary = self._fallback(ordinary)
        decoded = decode_observation(ordinary)
        current_limit = float(decoded["current_limit_m_s"])
        normalized_price = float(strategy_action)
        price = normalized_action_to_lambda(normalized_price)
        if decision_boundary:
            self.current_price_action = normalized_price
            self.current_price_kwh_per_s = price
            self._segment_actions.clear()
            self._fallback_until_boundary = False

        signature_changed = (
            self._planned_transition_signature != self._transition_signature()
        )
        replanned = bool(
            decision_boundary
            or signature_changed
            or (not self._segment_actions and not self._fallback_until_boundary)
        )
        planner_error: str | None = None
        if replanned:
            try:
                self._replan(current_limit)
            except RuntimeError as error:
                planner_error = str(error)
                self.last_plan = None
                self._segment_actions.clear()

        proposal = (
            float(self._segment_actions.pop(0))
            if self._segment_actions
            else fallback
        )
        next_state, _, _ = advance(
            self.state, proposal, self.vehicle, self.config
        )
        transitions = self._transitions()
        endpoint_limit = _endpoint_limit(current_limit, next_state, transitions)
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
        preview_safe, preview_calls = _unknown_preview_safe(
            next_state, vehicle=self.vehicle, config=self.config
        )
        viability = pessimistic_completion_bound(
            next_state, vehicle=self.vehicle, config=self.config
        )
        identical_to_fallback = abs(proposal - fallback) <= 1.0e-15
        deadline_safe = viability.viable or identical_to_fallback
        accepted = all((endpoint_safe, known_safe, preview_safe, deadline_safe))
        if planner_error is not None:
            reason = "PLANNER_INFEASIBLE_FALLBACK"
        elif not endpoint_safe:
            reason = "RIGHT_ENDPOINT_SPEED"
        elif not known_safe:
            reason = "KNOWN_TRANSITION_BRAKING"
        elif not preview_safe:
            reason = "UNKNOWN_PREVIEW_BRAKING"
        elif not deadline_safe:
            reason = "DEADLINE_RECOVERABILITY"
        elif identical_to_fallback:
            reason = "ACCEPTED_IDENTICAL_TO_FALLBACK"
        else:
            reason = "ACCEPTED"
        executed = proposal if accepted else fallback
        if not accepted:
            self._segment_actions.clear()
            self._fallback_until_boundary = True
        selected = None if self.last_plan is None else self.last_plan.selected
        planner_calls = 0 if self.last_plan is None else self.last_plan.physics_calls
        self.last_decision = PriceShieldDecision(
            fallback_action=fallback,
            proposed_native_action=proposal,
            executed_action=float(executed),
            normalized_price_action=normalized_price,
            price_kwh_per_s=price,
            planner_profile_id=None if selected is None else selected.profile_id,
            planner_family=None if selected is None else selected.family,
            planner_terminal_energy_kwh=(
                None if selected is None else selected.terminal_energy_kwh
            ),
            planner_terminal_time_s=(
                None if selected is None else selected.terminal_time_s
            ),
            accepted=accepted,
            overridden=not accepted,
            intervention_reason=reason,
            endpoint_speed_safe=endpoint_safe,
            known_transition_braking_safe=known_safe,
            unknown_preview_safe=preview_safe,
            deadline_viable=viability.viable,
            completion_bound_s=viability.completion_bound_s,
            native_action_influence=abs(float(executed) - fallback),
            physics_prediction_calls=(
                1
                + preview_calls
                + viability.physics_prediction_calls
                + (planner_calls if replanned else 0)
            ),
            replanned=replanned,
        )
        return float(executed)

    def observe_executed_action(self, action: float) -> None:
        value = float(action)
        if self.last_decision is None or value != self.last_decision.executed_action:
            raise RuntimeError("environment action differs from shield decision")
        if self._pending_fallback is not None:
            raise RuntimeError("cannot observe with an unconsumed strategy step")
        self.reference.observe_external_action(value)
        self.last_executed_action = value
