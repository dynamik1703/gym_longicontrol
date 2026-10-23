"""Feedback-corrected residual wrapper around frozen AdaptivePrice MPC V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.causal_mpc_v2.controller import PriceMPCController
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
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig
from gym_longicontrol.domain.vehicle import VehicleModel

from .chunk_decoder import (
    NATIVE_STEPS_PER_SECOND,
    ResidualChunk,
    combined_proposal,
    method_spec,
    validate_actor_action,
)

ADAPTIVE_HIGH_RESERVE_S = 24.0
ADAPTIVE_LOW_RESERVE_S = 2.0


@dataclass(frozen=True)
class ResidualSafetyDecision:
    native_step: int
    chunk_id: int
    chunk_phase: int
    actor_action: tuple[float, ...]
    requested_residual: float
    mpc_fallback_action: float
    mpc_proposed_action: float
    mpc_executed_action: float
    combined_proposal: float
    executed_action: float
    residual_accepted: bool
    residual_attempted: bool
    residual_influence: float
    intervention_deviation: float
    endpoint_speed_safe: bool
    known_transition_braking_safe: bool
    unknown_preview_safe: bool
    deadline_viable: bool
    completion_bound_s: float | None
    cancellation_reason: str | None
    chunk_cancelled: bool
    newly_visible_transition: bool
    mpc_fallback_active: bool
    current_speed_limit_m_s: float
    visible_speed_limits_m_s: tuple[float, float]
    visible_transition_distances_m: tuple[float, float]
    predicted_position_m: float
    predicted_velocity_m_s: float
    predicted_acceleration_m_s2: float
    physics_prediction_calls: int
    mpc_decision: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FeedbackResidualController:
    """Add safe learned corrections while retaining frozen MPC feedback.

    ``prepare_native_step`` is called once before every native action. It
    triggers MPC replanning at the original one-second boundaries and returns
    the actor observation only at the method's scheduled decision boundary.
    """

    information = (
        "ordinary causal 8D observation; frozen 16D MPC strategy observation; "
        "joint residual actor action; exact known dynamics; no route or future state"
    )

    def __init__(
        self,
        method: str,
        *,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
    ) -> None:
        self.spec = method_spec(method)
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        self.mpc = PriceMPCController(vehicle=self.vehicle, config=self.config)
        self.selector = AdaptivePrice(
            high_reserve_s=ADAPTIVE_HIGH_RESERVE_S,
            low_reserve_s=ADAPTIVE_LOW_RESERVE_S,
        )
        self.native_step = 0
        self.chunk_count = 0
        self.active_chunk: ResidualChunk | None = None
        self.held_price: float | None = None
        self.last_decision: ResidualSafetyDecision | None = None
        self._prepared_observation: np.ndarray | None = None
        self._prepared_actor_boundary = False
        self._signature_before_native: tuple[tuple[float, float], ...] = ()
        self._actor_observation: np.ndarray | None = None

    @property
    def state(self):
        return self.mpc.state

    @property
    def actor_interval_native_steps(self) -> int:
        return self.spec.actor_interval_native_steps

    def _transition_signature(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (transition.position_m, transition.limit_m_s)
            for transition in self.mpc._transitions()
        )

    def reset(self) -> None:
        self.mpc.reset()
        self.selector.reset()
        self.native_step = 0
        self.chunk_count = 0
        self.active_chunk = None
        self.held_price = None
        self.last_decision = None
        self._prepared_observation = None
        self._prepared_actor_boundary = False
        self._signature_before_native = ()
        self._actor_observation = None

    def prepare_native_step(self, observation: np.ndarray) -> np.ndarray | None:
        if self._prepared_observation is not None:
            raise RuntimeError("previous prepared native step was not consumed")
        ordinary = np.asarray(observation, dtype=np.float64)
        if ordinary.shape != (8,) or not np.isfinite(ordinary).all():
            raise ValueError("ordinary observation must be a finite 8-vector")
        self._signature_before_native = self._transition_signature()
        self._prepared_actor_boundary = (
            self.native_step % self.actor_interval_native_steps == 0
        )
        self._actor_observation = None
        if self.native_step % NATIVE_STEPS_PER_SECOND == 0:
            strategy_observation = self.mpc.prepare_strategy_step(ordinary)
            self.held_price = float(self.selector.action(strategy_observation))
            if self._prepared_actor_boundary:
                self._actor_observation = strategy_observation.copy()
        if self._prepared_actor_boundary and self._actor_observation is None:
            raise RuntimeError("actor and MPC boundaries must coincide")
        self._prepared_observation = ordinary.copy()
        return (
            None if self._actor_observation is None else self._actor_observation.copy()
        )

    def begin_chunk(self, actor_action: np.ndarray) -> None:
        if not self._prepared_actor_boundary:
            raise RuntimeError("actor action is only allowed at a scheduled boundary")
        if self.active_chunk is not None:
            expected_end = (
                self.active_chunk.start_native_step
                + self.active_chunk.duration_native_steps
            )
            if self.native_step != expected_end:
                raise RuntimeError("prior chunk did not reach its scheduled boundary")
        action = validate_actor_action(actor_action, self.spec.action_dim)
        self.active_chunk = ResidualChunk(
            action=action,
            start_native_step=self.native_step,
            chunk_id=self.chunk_count,
        )
        self.chunk_count += 1

    def _mpc_fallback_active(self) -> bool:
        decision = self.mpc.last_decision
        if decision is None:
            raise RuntimeError("frozen MPC decision is missing")
        return bool(not decision.accepted or decision.planner_profile_id is None)

    def action(self, observation: np.ndarray) -> float:
        ordinary = np.asarray(observation, dtype=np.float64)
        if self._prepared_observation is None or not np.array_equal(
            ordinary, self._prepared_observation
        ):
            raise RuntimeError("native action does not match prepared observation")
        if self.active_chunk is None:
            raise RuntimeError("no residual chunk is active")
        if self.held_price is None:
            raise RuntimeError("frozen AdaptivePrice decision is missing")
        phase = self.active_chunk.phase(self.native_step)
        requested_residual = self.active_chunk.correction(self.native_step)
        mpc_action = float(self.mpc.action(ordinary, self.held_price))
        mpc_decision = self.mpc.last_decision
        if mpc_decision is None:
            raise RuntimeError("frozen MPC did not record its decision")
        signature_after_mpc = self._transition_signature()
        newly_visible = signature_after_mpc != self._signature_before_native
        if newly_visible and not self._prepared_actor_boundary:
            self.active_chunk.cancel(
                native_step=self.native_step,
                reason="NEWLY_VISIBLE_SPEED_LIMIT_TRANSITION",
            )

        proposal = combined_proposal(mpc_action, requested_residual)
        mpc_fallback = self._mpc_fallback_active()
        if mpc_fallback:
            self.active_chunk.cancel(
                native_step=self.native_step,
                reason=f"MPC_FALLBACK:{mpc_decision.intervention_reason}",
            )

        decoded = decode_observation(ordinary)
        current_limit = float(decoded["current_limit_m_s"])
        next_state, _, _ = advance(self.state, proposal, self.vehicle, self.config)
        transitions = self.mpc._transitions()
        endpoint_limit = _endpoint_limit(current_limit, next_state, transitions)
        endpoint_safe = bool(
            next_state.velocity_m_s
            <= endpoint_limit - VELOCITY_SAFETY_MARGIN_M_S + 1.0e-12
        )
        known_safe = bool(
            braking_safe(
                next_state,
                transitions,
                vehicle=self.vehicle,
                config=self.config,
            )
        )
        preview_safe, preview_calls = _unknown_preview_safe(
            next_state, vehicle=self.vehicle, config=self.config
        )
        viability = pessimistic_completion_bound(
            next_state, vehicle=self.vehicle, config=self.config
        )
        identical_to_mpc = abs(proposal - mpc_action) <= 1.0e-15
        deadline_safe = bool(viability.viable or identical_to_mpc)
        gates_pass = bool(
            endpoint_safe and known_safe and preview_safe and deadline_safe
        )
        residual_attempted = bool(not self.active_chunk.cancelled and not mpc_fallback)
        residual_accepted = bool(residual_attempted and gates_pass)
        cancellation_reason = self.active_chunk.cancellation_reason
        if residual_attempted and not gates_pass:
            if not endpoint_safe:
                cancellation_reason = "RESIDUAL_RIGHT_ENDPOINT_SPEED"
            elif not known_safe:
                cancellation_reason = "RESIDUAL_KNOWN_TRANSITION_BRAKING"
            elif not preview_safe:
                cancellation_reason = "RESIDUAL_UNKNOWN_PREVIEW_BRAKING"
            else:
                cancellation_reason = "RESIDUAL_DEADLINE_RECOVERABILITY"
            self.active_chunk.cancel(
                native_step=self.native_step, reason=cancellation_reason
            )
        executed = proposal if residual_accepted else mpc_action
        influence = abs(executed - mpc_action)
        if influence > 1.0e-15:
            self.active_chunk.residual_influence_native_steps += 1
        self.last_decision = ResidualSafetyDecision(
            native_step=self.native_step,
            chunk_id=self.active_chunk.chunk_id,
            chunk_phase=phase,
            actor_action=tuple(float(value) for value in self.active_chunk.action),
            requested_residual=requested_residual,
            mpc_fallback_action=float(mpc_decision.fallback_action),
            mpc_proposed_action=float(mpc_decision.proposed_native_action),
            mpc_executed_action=mpc_action,
            combined_proposal=proposal,
            executed_action=float(executed),
            residual_accepted=residual_accepted,
            residual_attempted=residual_attempted,
            residual_influence=influence,
            intervention_deviation=abs(proposal - executed),
            endpoint_speed_safe=endpoint_safe,
            known_transition_braking_safe=known_safe,
            unknown_preview_safe=preview_safe,
            deadline_viable=viability.viable,
            completion_bound_s=viability.completion_bound_s,
            cancellation_reason=self.active_chunk.cancellation_reason,
            chunk_cancelled=self.active_chunk.cancelled,
            newly_visible_transition=newly_visible,
            mpc_fallback_active=mpc_fallback,
            current_speed_limit_m_s=current_limit,
            visible_speed_limits_m_s=tuple(
                float(value) for value in decoded["future_limits_m_s"]
            ),
            visible_transition_distances_m=tuple(
                float(value) for value in decoded["distances_m"]
            ),
            predicted_position_m=next_state.position_m,
            predicted_velocity_m_s=next_state.velocity_m_s,
            predicted_acceleration_m_s2=next_state.acceleration_m_s2,
            physics_prediction_calls=(
                1 + preview_calls + viability.physics_prediction_calls
            ),
            mpc_decision=mpc_decision.as_dict(),
        )
        self._prepared_observation = None
        self._prepared_actor_boundary = False
        self._actor_observation = None
        return float(executed)

    def observe_executed_action(self, action: float) -> None:
        value = float(action)
        if self.last_decision is None or value != self.last_decision.executed_action:
            raise RuntimeError("environment action differs from residual decision")
        if value == self.last_decision.mpc_executed_action:
            self.mpc.observe_executed_action(value)
        else:
            if self.mpc._pending_fallback is not None:
                raise RuntimeError("cannot propagate with an unconsumed MPC step")
            self.mpc.reference.observe_external_action(value)
            self.mpc.last_executed_action = value
        self.native_step += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": self.spec.name,
            "native_step": self.native_step,
            "chunk_count": self.chunk_count,
            "active_chunk": (
                None if self.active_chunk is None else self.active_chunk.state_dict()
            ),
            "held_price": self.held_price,
            "last_decision": (
                None if self.last_decision is None else self.last_decision.as_dict()
            ),
            "prepared_observation": (
                None
                if self._prepared_observation is None
                else self._prepared_observation.copy()
            ),
            "prepared_actor_boundary": self._prepared_actor_boundary,
            "signature_before_native": self._signature_before_native,
            "actor_observation": (
                None
                if self._actor_observation is None
                else self._actor_observation.copy()
            ),
        }
