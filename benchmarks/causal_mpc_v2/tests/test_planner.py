from __future__ import annotations

import math

from benchmarks.causal_mpc_v2.planner import (
    PROFILES,
    FixedDistanceProfilePlanner,
)
from benchmarks.causal_mpc_v2.terminal_value import estimate_terminal_value
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel


def _solve(price: float):
    return FixedDistanceProfilePlanner().solve(
        VehicleState(velocity_m_s=10.0),
        current_limit_m_s=60.0 / 3.6,
        transitions=(),
        price_kwh_per_s=price,
    )


def test_profile_lattice_has_all_required_physical_modes() -> None:
    assert {profile.family for profile in PROFILES} == {
        "regulated_cruise",
        "pulse_coast",
        "traction_then_coast",
        "traction_coast_brake",
        "coast_then_partial_traction",
    }
    declared = {
        profile.traction_action for profile in PROFILES
    } | {profile.brake_action for profile in PROFILES}
    assert {-1.0, -0.5, 0.0, 0.5, 1.0} <= declared


def test_lambda_changes_only_selection_and_duration_is_monotone() -> None:
    low = _solve(1.0e-5)
    high = _solve(1.0e-2)
    assert [candidate.profile_id for candidate in low.candidates] == [
        candidate.profile_id for candidate in high.candidates
    ]
    for left, right in zip(low.candidates, high.candidates):
        assert left.actions == right.actions
        assert left.horizon_energy_kwh == right.horizon_energy_kwh
        assert left.terminal_energy_kwh == right.terminal_energy_kwh
        assert left.travel_time_s == right.travel_time_s
        assert left.terminal_time_s == right.terminal_time_s
    assert (
        high.selected.total_estimated_time_s
        <= low.selected.total_estimated_time_s + 1.0e-9
    )
    assert high.selected.actions[0] != low.selected.actions[0]


def test_selected_trajectory_replays_exact_production_physics() -> None:
    start = VehicleState(velocity_m_s=10.0)
    plan = _solve(1.0e-3)
    cursor = start
    for action in plan.selected.actions:
        cursor, _, _ = advance(
            cursor, action, VehicleModel(), SimulationConfig()
        )
    assert cursor.position_m == plan.selected.final_position_m
    assert cursor.velocity_m_s == plan.selected.final_velocity_m_s
    assert (
        cursor.total_energy_kwh - start.total_energy_kwh
        == plan.selected.horizon_energy_kwh
    )
    assert cursor.position_m >= plan.waypoint_position_m
    assert plan.segment_actions
    assert all(-1.0 <= action <= 1.0 for action in plan.segment_actions)


def test_known_transition_right_endpoint_is_respected() -> None:
    limit = 20.0 / 3.6
    plan = FixedDistanceProfilePlanner().solve(
        VehicleState(velocity_m_s=8.0),
        current_limit_m_s=60.0 / 3.6,
        transitions=(Transition(30.0, limit),),
        price_kwh_per_s=1.0e-3,
    )
    cursor = VehicleState(velocity_m_s=8.0)
    for action in plan.selected.actions:
        cursor, _, _ = advance(cursor, action, VehicleModel(), SimulationConfig())
        if cursor.position_m >= 30.0:
            assert cursor.velocity_m_s <= limit


def test_braking_profiles_execute_moderate_and_strong_modes() -> None:
    plan = FixedDistanceProfilePlanner().solve(
        VehicleState(velocity_m_s=15.0),
        current_limit_m_s=60.0 / 3.6,
        transitions=(),
        price_kwh_per_s=1.0e-3,
    )
    braking = {
        candidate.profile_id: candidate
        for candidate in plan.candidates
        if candidate.family == "traction_coast_brake" and candidate.feasible
    }
    assert "moderate_regenerative_braking" in braking[
        "accel_coast_brake_50_-0.5"
    ].action_modes_used
    assert "strong_braking" in braking[
        "accel_coast_brake_50_-1.0"
    ].action_modes_used


def test_terminal_value_is_finite_paced_and_not_regen_rewarded() -> None:
    model = VehicleModel()
    config = SimulationConfig()
    slow = estimate_terminal_value(
        VehicleState(position_m=800.0, velocity_m_s=2.0, elapsed_time_s=90.0),
        current_limit_m_s=60.0 / 3.6,
        transitions=(),
        vehicle=model,
        config=config,
    )
    fast = estimate_terminal_value(
        VehicleState(position_m=800.0, velocity_m_s=15.0, elapsed_time_s=90.0),
        current_limit_m_s=60.0 / 3.6,
        transitions=(),
        vehicle=model,
        config=config,
    )
    assert 20.0 / 3.6 <= slow.target_speed_m_s < 60.0 / 3.6
    assert slow.energy_kwh > 0.0
    assert fast.energy_kwh >= 0.0
    assert all(
        math.isfinite(value)
        for value in (
            slow.energy_kwh,
            slow.time_s,
            slow.estimated_completion_time_s,
            slow.lateness_penalty_kwh,
        )
    )


def test_solver_is_deterministic() -> None:
    first = _solve(1.0e-3)
    second = _solve(1.0e-3)
    assert first.as_dict() == second.as_dict()

