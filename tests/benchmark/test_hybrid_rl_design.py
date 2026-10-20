from __future__ import annotations

import ast
import inspect

import gymnasium as gym
import pytest

from benchmarks.hybrid_rl_design import controller as hybrid_controller
from benchmarks.hybrid_rl_design.controller import (
    HybridTargetSpeedController,
    normalized_action_to_target_speed,
    target_speed_to_normalized_action,
)
from benchmarks.hybrid_rl_design.preflight import (
    fallback_parity_preflight,
    shield_preflight,
)
from benchmarks.hybrid_rl_design.prepare import STATUS, verify
from benchmarks.hybrid_rl_design.split import build_split_ledger, verify_runtime_row
from benchmarks.hybrid_rl_design.viability import pessimistic_completion_bound
from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.envs.longicontrol import LongiControlEnv


def test_target_speed_action_mapping_is_bounded_and_round_trips():
    assert normalized_action_to_target_speed(-1.0) == pytest.approx(20.0 / 3.6)
    assert normalized_action_to_target_speed(1.0) == pytest.approx(60.0 / 3.6)
    for action in (-1.0, -0.5, 0.0, 0.5, 1.0):
        target = normalized_action_to_target_speed(action)
        assert target_speed_to_normalized_action(target) == pytest.approx(action)
    with pytest.raises(ValueError):
        normalized_action_to_target_speed(1.01)


def test_deadline_viability_is_falsifiable_and_state_dependent():
    initial = pessimistic_completion_bound(VehicleState())
    recoverable = pessimistic_completion_bound(
        VehicleState(position_m=500.0, velocity_m_s=20.0 / 3.6, elapsed_time_s=40.0)
    )
    late = pessimistic_completion_bound(
        VehicleState(position_m=100.0, velocity_m_s=20.0 / 3.6, elapsed_time_s=100.0)
    )
    assert not initial.viable
    assert recoverable.viable
    assert not late.viable
    assert recoverable.completion_bound_s < late.completion_bound_s


def test_disabled_hybrid_has_exact_frozen_fallback_parity():
    result = fallback_parity_preflight()
    assert result["fixture_count"] == 6
    assert result["all_exact"]
    assert all(row["all_matching"] for row in result["fixtures"])


def test_shield_fixtures_cover_endpoint_deadline_and_policy_influence():
    result = shield_preflight()
    assert result["endpoint_safety_passed"]
    assert result["deadline_fixture_passed"]
    assert result["feasibility_fixture_passed"]
    assert not result["viability_falsifications"]
    assert not result["SHIELD_DOMINATES_POLICY"]


def test_early_slow_proposal_is_rejected_for_deadline_recoverability():
    track = physical_fixtures()["constant_100_km_h"]
    environment = gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=lambda _rng: track), max_episode_steps=1_800
    )
    controller = HybridTargetSpeedController(strategy_enabled=True)
    try:
        observation, _ = environment.reset(seed=1)
        controller.reset()
        for _ in range(30):
            fallback_action = controller.action(observation, 1.0)
            observation, _, terminated, truncated, _ = environment.step(
                [fallback_action]
            )
            assert not terminated and not truncated
            controller.observe_executed_action(fallback_action)
        action = controller.action(observation, -1.0)
        decision = controller.last_decision
        assert decision is not None
        assert decision.overridden
        assert decision.intervention_reason == "DEADLINE_RECOVERABILITY"
        assert action == decision.fallback_action
    finally:
        environment.close()


def test_controller_api_and_ast_exclude_privileged_runtime_inputs():
    signature = inspect.signature(HybridTargetSpeedController.action)
    assert tuple(signature.parameters) == ("self", "observation", "strategy_action")
    tree = ast.parse(inspect.getsource(hybrid_controller))
    forbidden_attributes = {"track", "unwrapped"}
    observed = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not (observed & forbidden_attributes)


def test_two_stage_strategy_interface_exposes_only_causal_16d_input():
    track = physical_fixtures()["constant_100_km_h"]
    environment = gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=lambda _rng: track), max_episode_steps=1_800
    )
    controller = HybridTargetSpeedController(strategy_enabled=True)
    try:
        observation, _ = environment.reset(seed=2)
        controller.reset()
        strategy_observation = controller.prepare_strategy_step(observation)
        assert strategy_observation.shape == (16,)
        action = controller.action(observation, 1.0)
        observation, _, _, _, _ = environment.step([action])
        controller.observe_executed_action(action)
        assert observation.shape == (8,)
    finally:
        environment.close()


def test_hybrid_splits_are_route_clean_and_paper_fails_closed():
    ledger = build_split_ledger()
    assert ledger["route_clean_verified"]
    assert ledger["training"]["total_reachable_rows"] == 2595
    assert len(ledger["evaluation_sets"]["diagnostic"]["accepted_tasks"]) == 18
    assert len(ledger["evaluation_sets"]["validation"]["accepted_tasks"]) == 36
    for split in ledger["evaluation_sets"].values():
        for row in split["accepted_tasks"]:
            assert verify_runtime_row(row)
    with pytest.raises(PermissionError, match="PAPER_TEST_SEALED"):
        verify_runtime_row(
            {
                "numeric_seed": 4000,
                "route_sha256": "not-constructed",
                "full_task_sha256": "not-constructed",
            }
        )


def test_materialized_preparation_is_ready_but_did_not_authorize_or_train():
    if not STATUS.exists():
        pytest.skip("Hybrid preparation has not been materialized")
    status = verify()
    assert status["ready_for_future_separately_authorized_training"]
    assert not status["training_authorized"]
    assert not status["training_started"]
    assert status["real_training_transitions"] == 0
    assert status["rl_gradient_updates"] == 0
    assert status["hybrid_final_validation"]["state"] == "FROZEN_UNOPENED"
    assert status["hybrid_final_validation"]["performance_episode_count"] == 0
