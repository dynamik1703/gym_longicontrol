"""Matched native proposals through immutable inherited safety predicates."""

from __future__ import annotations

from dataclasses import asdict

from benchmarks.hybrid_rl_design.controller import (
    _endpoint_limit,
    _unknown_preview_safe,
)
from benchmarks.hybrid_rl_design.viability import pessimistic_completion_bound
from benchmarks.physical_feasibility_reference.oracle_controller import (
    VELOCITY_SAFETY_MARGIN_M_S,
    braking_safe,
)
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.vehicle import VehicleModel

from .backup_policy import CONFIG


class MeteredVehicle(VehicleModel):
    """Instrumentation only; every acceleration and power value is inherited."""

    def __init__(self):
        super().__init__()
        self.predictions = 0

    def power_kw(self, velocity_m_s, acceleration_m_s2):
        self.predictions += 1
        return super().power_kw(velocity_m_s, acceleration_m_s2)


def inherited_test(state, action, current_limit, transitions, fallback, vehicle):
    start = vehicle.predictions
    successor, _, _ = advance(state, action, vehicle, CONFIG)
    endpoint = successor.velocity_m_s <= (
        _endpoint_limit(current_limit, successor, transitions)
        - VELOCITY_SAFETY_MARGIN_M_S
    )
    known = braking_safe(successor, transitions, vehicle=vehicle, config=CONFIG)
    unknown, _ = _unknown_preview_safe(successor, vehicle=vehicle, config=CONFIG)
    deadline = pessimistic_completion_bound(successor, vehicle=vehicle, config=CONFIG)
    accepted = (
        endpoint and known and unknown and (deadline.viable or action == fallback)
    )
    return {
        "accepted": bool(accepted),
        "right_endpoint": bool(endpoint),
        "known_braking": bool(known),
        "unknown_preview": bool(unknown),
        "deadline_viable": deadline.viable,
        "completion_bound_s": deadline.completion_bound_s,
        "deadline_diagnostic_applicable": True,
        "fallback_identical_exception": action == fallback,
        "predicted_next_state": asdict(successor),
        "physics_predictions": vehicle.predictions - start,
    }
