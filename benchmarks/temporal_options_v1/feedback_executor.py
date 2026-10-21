"""Native-feedback executor with the inherited physical authority and Shield."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from benchmarks.causal_mpc_v2.planner import FixedDistanceProfilePlanner, Profile
from benchmarks.causal_mpc_v2.terminal_value import limit_at
from benchmarks.hybrid_rl_design.controller import (
    _endpoint_limit,
    _unknown_preview_safe,
)
from benchmarks.hybrid_rl_design.viability import pessimistic_completion_bound
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

from .option_state import OptionState


@dataclass(frozen=True)
class NativeFeedbackDecision:
    option_instance_id: int
    profile_id: str
    phase: str
    progress: float
    fallback_action: float
    raw_profile_action: float
    speed_capped_action: float
    executed_action: float
    accepted: bool
    intervention_reason: str
    endpoint_speed_safe: bool
    known_transition_braking_safe: bool
    unknown_preview_safe: bool
    deadline_viable: bool
    completion_bound_s: float
    physics_prediction_calls: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def frozen_fallback_action(
    state: VehicleState,
    *,
    current_limit_m_s: float,
    transitions: tuple[Transition, ...],
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> float:
    """Reproduce the physical fallback proposal from an explicit causal state."""

    target = max(
        0.0,
        min(CAUSAL_CRUISE_CAP_M_S, current_limit_m_s) - VELOCITY_SAFETY_MARGIN_M_S,
    )
    action = action_for_next_velocity(state, target, vehicle=vehicle, config=config)
    next_state, _, _ = advance(state, action, vehicle, config)
    endpoint = _endpoint_limit(current_limit_m_s, next_state, transitions)
    if (
        next_state.velocity_m_s <= endpoint - VELOCITY_SAFETY_MARGIN_M_S
        and braking_safe(next_state, transitions, vehicle=vehicle, config=config)
    ):
        return float(action)
    return -1.0


class NativeFeedbackExecutor:
    """Recompute one option action from current physical state every 0.1 s."""

    def __init__(
        self,
        *,
        vehicle: VehicleModel | None = None,
        config: SimulationConfig = SimulationConfig(),
        planner: FixedDistanceProfilePlanner | None = None,
    ) -> None:
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.config = config
        self.profile_law = planner or FixedDistanceProfilePlanner(
            vehicle=self.vehicle, config=self.config
        )

    def decide(
        self,
        option: OptionState,
        profile: Profile,
        state: VehicleState,
        *,
        current_limit_m_s: float,
        transitions: tuple[Transition, ...],
        fallback_action: float,
    ) -> NativeFeedbackDecision:
        if option.profile_id != profile.profile_id:
            raise ValueError("option/profile mismatch")
        progress = float(
            max(
                0.0,
                min(
                    1.0,
                    (state.position_m - option.start_position_m)
                    / option.original_length_m,
                ),
            )
        )
        option.progress = progress
        raw = float(self.profile_law._raw_action(profile, state, progress))
        local_limit = limit_at(state.position_m, current_limit_m_s, transitions)
        speed_cap = max(
            0.0,
            min(CAUSAL_CRUISE_CAP_M_S, local_limit) - VELOCITY_SAFETY_MARGIN_M_S,
        )
        proposal = raw
        next_state, _, _ = advance(state, proposal, self.vehicle, self.config)
        calls = 1
        if next_state.velocity_m_s > speed_cap + 1.0e-12:
            proposal = action_for_next_velocity(
                state, speed_cap, vehicle=self.vehicle, config=self.config
            )
            next_state, _, _ = advance(state, proposal, self.vehicle, self.config)
            calls += 1
        endpoint = _endpoint_limit(current_limit_m_s, next_state, transitions)
        endpoint_safe = (
            next_state.velocity_m_s <= endpoint - VELOCITY_SAFETY_MARGIN_M_S + 1.0e-12
        )
        known_safe = braking_safe(
            next_state, transitions, vehicle=self.vehicle, config=self.config
        )
        preview_safe, preview_calls = _unknown_preview_safe(
            next_state, vehicle=self.vehicle, config=self.config
        )
        viability = pessimistic_completion_bound(
            next_state, vehicle=self.vehicle, config=self.config
        )
        identical = abs(proposal - fallback_action) <= 1.0e-15
        deadline_safe = viability.viable or identical
        accepted = all((endpoint_safe, known_safe, preview_safe, deadline_safe))
        if not endpoint_safe:
            reason = "RIGHT_ENDPOINT_SPEED"
        elif not known_safe:
            reason = "KNOWN_TRANSITION_BRAKING"
        elif not preview_safe:
            reason = "UNKNOWN_PREVIEW_BRAKING"
        elif not deadline_safe:
            reason = "DEADLINE_RECOVERABILITY"
        elif identical:
            reason = "ACCEPTED_IDENTICAL_TO_FALLBACK"
        else:
            reason = "ACCEPTED"
        return NativeFeedbackDecision(
            option_instance_id=option.instance_id,
            profile_id=profile.profile_id,
            phase=option.phase,
            progress=progress,
            fallback_action=float(fallback_action),
            raw_profile_action=raw,
            speed_capped_action=float(proposal),
            executed_action=float(proposal if accepted else fallback_action),
            accepted=accepted,
            intervention_reason=reason,
            endpoint_speed_safe=endpoint_safe,
            known_transition_braking_safe=known_safe,
            unknown_preview_safe=preview_safe,
            deadline_viable=viability.viable,
            completion_bound_s=viability.completion_bound_s,
            physics_prediction_calls=(
                calls + preview_calls + viability.physics_prediction_calls
            ),
        )
