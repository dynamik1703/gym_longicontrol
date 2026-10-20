from __future__ import annotations

from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.physical_feasibility_reference import runner
from benchmarks.physical_feasibility_reference.causal_controller import (
    ObservationLimitedController,
)
from benchmarks.physical_feasibility_reference.diagnostics import (
    build_split_ledger,
    physical_fixtures,
    preflight_benchmark,
    verify_runtime_row,
)
from benchmarks.physical_feasibility_reference.evaluation import (
    replay_actions,
    rollout,
    solve_oracle,
)
from benchmarks.physical_feasibility_reference.feasibility import (
    CANONICAL_TASK,
    FeasibilityClassification,
    classify_outcome,
    maximum_braking_result,
    physical_audit,
)
from benchmarks.physical_feasibility_reference.oracle_controller import (
    FullRoutePhysicsOracle,
    speed_limit_at,
)
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.domain.vehicle import VehicleModel
from gym_longicontrol.envs.longicontrol import LongiControlEnv


def test_exact_physics_braking_parity_and_required_finding():
    result = maximum_braking_result(100 / 3.6, 60 / 3.6)
    vehicle = VehicleModel()
    state = VehicleState(velocity_m_s=100 / 3.6)
    for _ in range(result.steps):
        state, _, _ = advance(state, -1.0, vehicle, SimulationConfig())
    assert result.steps == 60
    assert result.elapsed_time_s == pytest.approx(6.0)
    assert result.distance_m == pytest.approx(135.0764556315701)
    assert asdict(result)["distance_m"] == state.position_m
    assert result.final_velocity_m_s == state.velocity_m_s


def test_acceleration_and_braking_envelope_is_velocity_dependent():
    envelope = physical_audit()["acceleration_envelope"]
    maximum = [row["maximum_acceleration_m_s2"] for row in envelope]
    braking = [row["maximum_braking_m_s2"] for row in envelope]
    assert len(envelope) == 38
    assert maximum[0] != maximum[-2]
    assert braking[1] != braking[-2]
    assert physical_audit()["braking_100_to_60"]["steps"] == 60


def test_speed_limit_boundary_is_right_continuous_and_nextafter_exact():
    track = physical_fixtures()["maximum_legal_downward_change"]
    boundary = 500.0
    assert speed_limit_at(track, np.nextafter(boundary, -np.inf)) == pytest.approx(
        100 / 3.6
    )
    assert speed_limit_at(track, boundary) == pytest.approx(60 / 3.6)
    assert speed_limit_at(track, np.nextafter(boundary, np.inf)) == pytest.approx(
        60 / 3.6
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "constant_20_km_h",
        "constant_100_km_h",
        "upward_changes",
        "maximum_legal_downward_change",
        "consecutive_downward_changes",
        "short_braking_window",
        "route_endpoint_at_speed_boundary",
    ],
)
def test_oracle_completes_physical_fixtures_without_speed_violation(fixture_name):
    track = physical_fixtures()[fixture_name]
    base = LongiControlEnv(track_generator=lambda _rng: track)
    environment = gym.wrappers.TimeLimit(base, max_episode_steps=1800)
    try:
        _, info = environment.reset(seed=7)
        controller = FullRoutePhysicsOracle(track)
        ended = False
        while not ended:
            action = controller.action(environment.unwrapped.state)
            _, _, terminated, truncated, info = environment.step([action])
            ended = terminated or truncated
        metrics = EpisodeMetrics(**info["episode_metrics"])
        if fixture_name == "constant_20_km_h":
            assert not metrics.completed
        else:
            assert metrics.completed
        assert metrics.max_speed_violation_m_s == 0.0
    finally:
        environment.close()


def test_external_time_limit_remains_authoritative_for_incomplete_route():
    track = physical_fixtures()["forced_standstill"]
    environment = gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=lambda _rng: track), max_episode_steps=2
    )
    try:
        environment.reset(seed=1)
        environment.step([-1.0])
        _, _, terminated, truncated, info = environment.step([-1.0])
        metrics = EpisodeMetrics(**info["episode_metrics"])
        assert not terminated and truncated
        assert not metrics.completed
        assert not is_feasible(metrics, CANONICAL_TASK)
    finally:
        environment.close()


def test_external_feasibility_authority_uses_exact_zero_tolerance():
    exact = EpisodeMetrics(True, 140.0, 999.0, 0, 0.0, 0.0)
    above = EpisodeMetrics(True, 140.0, -999.0, 1, np.nextafter(0.0, 1.0), 0.0)
    assert is_feasible(exact, CANONICAL_TASK)
    assert not is_feasible(above, CANONICAL_TASK)
    assert classify_outcome(exact)[0] == FeasibilityClassification.FEASIBLE_WITNESS
    assert classify_outcome(None)[0] == FeasibilityClassification.UNRESOLVED
    assert (
        classify_outcome(
            None,
            TaskSpecification(0.01),
            unconstrained_time_lower_bound_s=1.0,
        )[0]
        == FeasibilityClassification.PROVEN_INFEASIBLE
    )


def test_oracle_witness_independent_replay_parity():
    result = solve_oracle(9_800_000, refinement_iterations=2)
    replay = replay_actions(9_800_000, result["actions"])
    assert result["feasible_witness_found"]
    assert result["numerical_parity"]
    assert replay == result["replay"]
    assert replay["trace_sha256"] == result["trace_sha256"]


def test_observation_limited_controller_has_no_track_or_oracle_input():
    controller = ObservationLimitedController()
    assert not hasattr(controller, "track")
    result = rollout(9_800_001, "causal")
    assert result["canonical_success"]
    assert result["speed_violation_count"] == 0
    assert "ordinary 8D observation" in result["information_available"]


def test_route_clean_split_is_deterministic_and_fully_disjoint():
    first = build_split_ledger()
    second = build_split_ledger()
    assert first == second
    assert first["route_clean_verified"]
    assert first["paper"]["content_fingerprints_constructed"] == 0
    assert len(first["evaluation_sets"]["development"]["accepted_tasks"]) == 18
    assert len(first["evaluation_sets"]["holdout"]["accepted_tasks"]) == 36
    for split in first["evaluation_sets"].values():
        for row in split["accepted_tasks"]:
            assert verify_runtime_row(row)


def test_paper_fails_closed_before_content_construction():
    with pytest.raises(PermissionError, match="PAPER_TEST_SEALED"):
        verify_runtime_row(
            {
                "numeric_seed": 4000,
                "route_sha256": "not-constructed",
                "full_task_sha256": "not-constructed",
            }
        )


def test_holdout_one_open_only_gate(monkeypatch, tmp_path):
    existing = tmp_path / "holdout_results.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner, "HOLDOUT", existing)
    with pytest.raises(RuntimeError, match="ONE_OPEN_ONLY"):
        runner.run_holdout()


def test_resource_accounting_gate_on_measured_trajectory():
    result = preflight_benchmark((9_800_000,))
    assert result["mean_time_per_simulated_trajectory_s"] > 0
    assert result["projected_total_physical_reference_execution_s"] > 0
    assert result["execution_gate_passed"]
    assert result["artifact_gate_passed"]
    assert result["checkpoint_overhead"].startswith("none")
