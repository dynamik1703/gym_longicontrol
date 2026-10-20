"""Target-speed Hybrid controller with an unchanged physical fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from benchmarks.physical_feasibility_reference.causal_controller import (
    CAUSAL_CRUISE_CAP_M_S,
    ObservationLimitedController,
    decode_observation,
)
from benchmarks.physical_feasibility_reference.oracle_controller import (
    BRAKING_POSITION_MARGIN_M,
    VELOCITY_SAFETY_MARGIN_M_S,
    Transition,
    action_for_next_velocity,
    braking_safe,
)
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .viability import (
    DEADLINE_S,
    MINIMUM_LIMIT_M_S,
    pessimistic_completion_bound,
)

TARGET_MIN_M_S = MINIMUM_LIMIT_M_S
TARGET_MAX_M_S = CAUSAL_CRUISE_CAP_M_S
STRATEGY_OBSERVATION_DIMENSIONS = 16


@dataclass(frozen=True)
class ShieldDecision:
    fallback_action: float
    proposed_native_action: float
    executed_action: float
    proposed_target_speed_m_s: float
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class _ExternalExecutionAdapter(ObservationLimitedController):
    """Advance frozen fallback memory after an accepted external proposal.

    Decision generation remains the inherited frozen controller. The adapter
    only supplies the action actually applied to its existing propagation path.
    """

    def observe_external_action(self, action: float) -> None:
        self._last_action = float(action)
        super().observe_executed_action(float(action))


def normalized_action_to_target_speed(action: float) -> float:
    value = float(action)
    if not np.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError("strategy action must be finite and in [-1, 1]")
    return TARGET_MIN_M_S + (value + 1.0) * 0.5 * (
        TARGET_MAX_M_S - TARGET_MIN_M_S
    )


def target_speed_to_normalized_action(target_m_s: float) -> float:
    target = float(np.clip(target_m_s, TARGET_MIN_M_S, TARGET_MAX_M_S))
    return 2.0 * (target - TARGET_MIN_M_S) / (
        TARGET_MAX_M_S - TARGET_MIN_M_S
    ) - 1.0


def _endpoint_limit(
    current_limit_m_s: float,
    next_state: VehicleState,
    transitions: tuple[Transition, ...],
) -> float:
    limit = float(current_limit_m_s)
    for transition in transitions:
        if transition.position_m <= next_state.position_m:
            limit = transition.limit_m_s
    return limit


def _unknown_preview_safe(
    next_state: VehicleState,
    *,
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> tuple[bool, int]:
    state = next_state
    calls = 0
    while state.velocity_m_s > MINIMUM_LIMIT_M_S - VELOCITY_SAFETY_MARGIN_M_S:
        state, _, _ = advance(state, -1.0, vehicle, config)
        calls += 1
        if calls > 10_000:
            raise RuntimeError("unknown-preview braking check failed to converge")
    distance = state.position_m - next_state.position_m
    safe = distance + BRAKING_POSITION_MARGIN_M <= config.sensor_range_m
    return safe, calls


class HybridTargetSpeedController:
    """Causal target-speed proposal plus speed/braking/deadline shield.

    The only runtime inputs are the ordinary 8D observation and, when enabled,
    a scalar strategy action. No environment object, Track, simulator state,
    future route, Oracle action, or future realized state is accepted.
    """

    information = (
        "ordinary 8D observation; scalar target-speed proposal; known exact "
        "dynamics; controller-propagated state/time; remembered visible signs; "
        "fixed deadline and generator bounds"
    )

    def __init__(
        self,
        *,
        strategy_enabled: bool,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
    ) -> None:
        self.strategy_enabled = bool(strategy_enabled)
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        if self.strategy_enabled:
            self.reference: ObservationLimitedController = _ExternalExecutionAdapter(
                vehicle=self.vehicle, config=self.config
            )
        else:
            self.reference = ObservationLimitedController(
                vehicle=self.vehicle, config=self.config
            )
        self.last_decision: ShieldDecision | None = None
        self.last_target_speed_m_s = TARGET_MAX_M_S
        self.last_executed_action = 0.0
        self._pending_observation: np.ndarray | None = None
        self._pending_fallback: float | None = None

    @property
    def state(self) -> VehicleState:
        return self.reference.state

    def reset(self) -> None:
        self.reference.reset()
        self.last_decision = None
        self.last_target_speed_m_s = TARGET_MAX_M_S
        self.last_executed_action = 0.0
        self._pending_observation = None
        self._pending_fallback = None

    def _transitions(self) -> tuple[Transition, ...]:
        known = self.reference._known_transitions
        return tuple(
            Transition(position, limit) for position, limit in sorted(known.items())
        )

    def strategy_observation(
        self, observation: np.ndarray, fallback_action: float
    ) -> np.ndarray:
        ordinary = np.asarray(observation, dtype=np.float64)
        if ordinary.shape != (8,):
            raise ValueError("ordinary observation must have shape (8,)")
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
                target_speed_to_normalized_action(self.last_target_speed_m_s),
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
        """Ingest the ordinary observation and expose the 16D strategy input.

        The subsequent ``action`` call must receive the same observation. This
        two-stage interface lets the learned strategy observe the physical
        fallback proposal without giving it an environment or simulator state.
        """

        if not self.strategy_enabled:
            raise RuntimeError("strategy step requested while influence is disabled")
        if self._pending_fallback is not None:
            raise RuntimeError("previous prepared strategy step was not consumed")
        ordinary = np.asarray(observation, dtype=np.float64)
        fallback = float(self.reference.action(ordinary))
        self._pending_observation = ordinary.copy()
        self._pending_fallback = fallback
        return self.strategy_observation(ordinary, fallback)

    def action(
        self, observation: np.ndarray, strategy_action: float | None = None
    ) -> float:
        ordinary = np.asarray(observation, dtype=np.float64)
        if self._pending_fallback is not None:
            if self._pending_observation is None or not np.array_equal(
                ordinary, self._pending_observation
            ):
                raise RuntimeError(
                    "prepared strategy observation does not match action"
                )
            fallback = self._pending_fallback
            self._pending_fallback = None
            self._pending_observation = None
        else:
            fallback = float(self.reference.action(ordinary))
        if not self.strategy_enabled:
            if strategy_action is not None:
                raise ValueError("strategy action supplied while influence is disabled")
            self.last_decision = ShieldDecision(
                fallback_action=fallback,
                proposed_native_action=fallback,
                executed_action=fallback,
                proposed_target_speed_m_s=TARGET_MAX_M_S,
                accepted=True,
                overridden=False,
                intervention_reason="STRATEGY_DISABLED_FALLBACK_PARITY",
                endpoint_speed_safe=True,
                known_transition_braking_safe=True,
                unknown_preview_safe=True,
                deadline_viable=True,
                completion_bound_s=None,
                native_action_influence=0.0,
                physics_prediction_calls=0,
            )
            return fallback
        if strategy_action is None:
            raise ValueError("enabled Hybrid controller requires a strategy action")

        target = normalized_action_to_target_speed(strategy_action)
        decoded = decode_observation(ordinary)
        current_limit = float(decoded["current_limit_m_s"])
        desired = max(
            0.0,
            min(target, current_limit, CAUSAL_CRUISE_CAP_M_S)
            - VELOCITY_SAFETY_MARGIN_M_S,
        )
        proposal = action_for_next_velocity(
            self.state,
            desired,
            vehicle=self.vehicle,
            config=self.config,
        )
        next_state, _, _ = advance(
            self.state, proposal, self.vehicle, self.config
        )
        transitions = self._transitions()
        endpoint_limit = _endpoint_limit(current_limit, next_state, transitions)
        endpoint_safe = (
            next_state.velocity_m_s
            <= endpoint_limit - VELOCITY_SAFETY_MARGIN_M_S
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
        identical_to_fallback = proposal == fallback
        deadline_safe = viability.viable or identical_to_fallback
        checks = (endpoint_safe, known_safe, preview_safe, deadline_safe)
        accepted = all(checks)
        if not endpoint_safe:
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
        influence = abs(executed - fallback)
        self.last_target_speed_m_s = target
        self.last_decision = ShieldDecision(
            fallback_action=fallback,
            proposed_native_action=proposal,
            executed_action=executed,
            proposed_target_speed_m_s=target,
            accepted=accepted,
            overridden=not accepted,
            intervention_reason=reason,
            endpoint_speed_safe=endpoint_safe,
            known_transition_braking_safe=known_safe,
            unknown_preview_safe=preview_safe,
            deadline_viable=viability.viable,
            completion_bound_s=viability.completion_bound_s,
            native_action_influence=influence,
            physics_prediction_calls=(
                2 + preview_calls + viability.physics_prediction_calls
            ),
        )
        return float(executed)

    def observe_executed_action(self, action: float) -> None:
        value = float(action)
        if self.last_decision is None or value != self.last_decision.executed_action:
            raise RuntimeError("environment action differs from shield decision")
        if self._pending_fallback is not None:
            raise RuntimeError("cannot observe action with an unconsumed strategy step")
        if self.strategy_enabled:
            adapter = self.reference
            assert isinstance(adapter, _ExternalExecutionAdapter)
            adapter.observe_external_action(value)
        else:
            self.reference.observe_executed_action(value)
        self.last_executed_action = value
