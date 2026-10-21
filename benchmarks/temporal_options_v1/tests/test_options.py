from __future__ import annotations

import pickle

import gymnasium as gym
import numpy as np
import pytest

import gym_longicontrol  # noqa: F401
from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.causal_mpc_v2.controller import PriceMPCController
from benchmarks.causal_mpc_v2.planner import PROFILES
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import VehicleState

from ..checkpointing import capture_checkpoint, restore_checkpoint
from ..controller import OptionArm, TemporalOptionController, build_controller
from ..feedback_executor import NativeFeedbackExecutor, frozen_fallback_action
from ..integrity import require_not_paper
from ..observation_adapter import (
    ACTION_IDS,
    OPTION_OBSERVATION_DIMENSION,
    augment_observation,
    discount_for_duration,
)
from ..option_selector import StatefulOptionSelector
from ..option_state import HighLevelAction, OptionState, phase_for
from ..semantics_audit import build_audit


def profile(profile_id: str):
    return next(value for value in PROFILES if value.profile_id == profile_id)


def ordinary_observation(state: VehicleState, limit_m_s: float = 100.0 / 3.6):
    return np.array(
        [
            state.velocity_m_s / 37.0,
            (state.acceleration_m_s2 + 3.0) / 6.0,
            limit_m_s / 37.0,
            limit_m_s / 37.0,
            limit_m_s / 37.0,
            1.0,
            1.0,
            0.2,
        ]
    )


def test_inherited_lattice_has_seventeen_untuned_profiles():
    assert len(PROFILES) == 17
    assert len({value.profile_id for value in PROFILES}) == 17


def test_anchor_goal_and_length_persist_across_updates():
    selected = profile("accel_coast_brake_50_-0.5")
    initial = VehicleState(position_m=100.0, elapsed_time_s=5.0)
    option = OptionState.start(
        instance_id=1,
        profile=selected,
        state=initial,
        route_length_m=1000.0,
    )
    option.update_physical_state(
        selected,
        VehicleState(position_m=150.0, elapsed_time_s=9.0),
        count_step=False,
    )
    assert option.start_position_m == 100.0
    assert option.target_waypoint_m == 200.0
    assert option.original_length_m == 100.0
    assert option.progress == pytest.approx(0.5)


def test_start_same_profile_is_distinct_from_continue():
    selected = profile("cruise_50")
    state = VehicleState(position_m=100.0)
    first = OptionState.start(
        instance_id=1, profile=selected, state=state, route_length_m=1000.0
    )
    continued = first.copy()
    restarted = OptionState.start(
        instance_id=2,
        profile=selected,
        state=VehicleState(position_m=110.0, elapsed_time_s=1.0),
        route_length_m=1000.0,
    )
    assert continued.instance_id == first.instance_id
    assert continued.start_position_m == first.start_position_m
    assert restarted.instance_id != first.instance_id
    assert restarted.start_position_m != first.start_position_m
    assert (
        HighLevelAction("CONTINUE").action_id
        != HighLevelAction("START", selected.profile_id).action_id
    )


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.39, "traction"), (0.4, "coast"), (0.81, "coast"), (0.82, "brake")],
)
def test_distance_phase_boundaries(fraction: float, expected: str):
    assert (
        phase_for(
            profile("accel_coast_brake_50_-0.5"),
            VehicleState(velocity_m_s=12.0),
            fraction,
        )
        == expected
    )


def test_semantics_audit_demonstrates_restart_postponement_and_persistence():
    audit = build_audit()
    affected = audit["profiles"]["accel_coast_brake_50_-0.5"]
    assert affected["restart"]["phase_counts"].get("brake", 0) == 0
    assert affected["restart"]["phase_counts"].get("coast", 0) == 0
    assert affected["persistent"]["phase_counts"].get("coast", 0) > 0
    assert affected["persistent"]["phase_counts"].get("brake", 0) > 0
    assert audit["native_fixture_steps"] < 30_000


def test_native_zero_uses_production_coasting_curve():
    executor = NativeFeedbackExecutor()
    state = VehicleState(velocity_m_s=15.0)
    next_state, power, _ = advance(state, 0.0, executor.vehicle, executor.config)
    assert next_state.acceleration_m_s2 == pytest.approx(
        executor.vehicle.acceleration_from_action(15.0, 0.0)
    )
    assert np.isfinite(power)
    assert next_state.velocity_m_s < state.velocity_m_s


def test_newly_visible_lower_limit_can_interrupt_active_coast():
    executor = NativeFeedbackExecutor()
    selected = profile("accel_coast_60_0.5")
    state = VehicleState(
        position_m=300.0,
        velocity_m_s=16.0,
        elapsed_time_s=10.0,
    )
    option = OptionState.start(
        instance_id=1, profile=selected, state=state, route_length_m=1000.0
    )
    option.progress = 0.6
    option.phase = "coast"
    transitions = (Transition(305.0, 20.0 / 3.6),)
    fallback = frozen_fallback_action(
        state,
        current_limit_m_s=100.0 / 3.6,
        transitions=transitions,
        vehicle=executor.vehicle,
        config=executor.config,
    )
    decision = executor.decide(
        option,
        selected,
        state,
        current_limit_m_s=100.0 / 3.6,
        transitions=transitions,
        fallback_action=fallback,
    )
    assert not decision.accepted
    assert decision.executed_action == fallback
    assert decision.intervention_reason in {
        "KNOWN_TRANSITION_BRAKING",
        "RIGHT_ENDPOINT_SPEED",
    }


def test_deadline_intervention_uses_frozen_fallback():
    executor = NativeFeedbackExecutor()
    selected = profile("accel_coast_40_0.5")
    state = VehicleState(position_m=100.0, velocity_m_s=10.0, elapsed_time_s=130.0)
    option = OptionState.start(
        instance_id=1, profile=selected, state=state, route_length_m=1000.0
    )
    fallback = frozen_fallback_action(
        state,
        current_limit_m_s=100.0 / 3.6,
        transitions=(),
        vehicle=executor.vehicle,
        config=executor.config,
    )
    decision = executor.decide(
        option,
        selected,
        state,
        current_limit_m_s=100.0 / 3.6,
        transitions=(),
        fallback_action=fallback,
    )
    assert not decision.deadline_viable
    if decision.speed_capped_action != fallback:
        assert not decision.accepted
        assert decision.executed_action == fallback


def test_final_route_option_uses_shortened_length():
    selected = profile("cruise_40")
    option = OptionState.start(
        instance_id=1,
        profile=selected,
        state=VehicleState(position_m=950.0),
        route_length_m=1000.0,
    )
    assert option.target_waypoint_m == 1000.0
    assert option.original_length_m == 50.0


def test_reset_clears_all_stale_option_state():
    controller = TemporalOptionController(OptionArm.CONTINUE_FEEDBACK)
    controller._start("cruise_50")
    controller.previous_termination = "SAFETY_INTERRUPTION"
    controller.reset()
    assert controller.active_option is None
    assert controller.option_history == []
    assert controller.native_step == 0
    assert controller.previous_termination == "NONE"


def test_augmented_observation_is_70d_and_masks_continue():
    inactive = augment_observation(
        np.zeros(16), active_option=None, previous_termination="NONE"
    )
    assert inactive.vector.shape == (OPTION_OBSERVATION_DIMENSION,)
    assert OPTION_OBSERVATION_DIMENSION == 70
    assert len(ACTION_IDS) == 18
    assert inactive.valid_action_mask[0] == 0.0
    option = OptionState.start(
        instance_id=1,
        profile=profile("cruise_50"),
        state=VehicleState(),
        route_length_m=1000.0,
    )
    active = augment_observation(
        np.zeros(16), active_option=option, previous_termination="NONE"
    )
    assert active.valid_action_mask[0] == 1.0


def test_identical_first_action_does_not_merge_memory_distinct_actions():
    selector = StatefulOptionSelector(profiles=(profile("cruise_50"),))
    state = VehicleState(position_m=300.0, velocity_m_s=10.0, elapsed_time_s=10.0)
    active = OptionState.start(
        instance_id=1,
        profile=profile("cruise_50"),
        state=VehicleState(position_m=290.0, velocity_m_s=10.0, elapsed_time_s=9.0),
        route_length_m=1000.0,
    )
    selection = selector.solve(
        state,
        active_option=active,
        allow_continue=True,
        next_instance_id=2,
        current_limit_m_s=100.0 / 3.6,
        transitions=(),
        price_kwh_per_s=1.0e-3,
    )
    continued, started = selection.candidates
    assert continued.actions[0] == pytest.approx(started.actions[0])
    assert continued.instance.start_position_m != started.instance.start_position_m
    assert continued.instance.target_waypoint_m != started.instance.target_waypoint_m


def test_continue_scoring_excludes_sunk_energy_and_retains_anchor():
    selector = StatefulOptionSelector(profiles=(profile("cruise_50"),))
    state = VehicleState(
        position_m=320.0,
        velocity_m_s=10.0,
        elapsed_time_s=12.0,
        total_energy_kwh=9.0,
    )
    active = OptionState.start(
        instance_id=1,
        profile=profile("cruise_50"),
        state=VehicleState(position_m=300.0, velocity_m_s=10.0, elapsed_time_s=10.0),
        route_length_m=1000.0,
    )
    candidate = selector.solve(
        state,
        active_option=active,
        allow_continue=True,
        next_instance_id=2,
        current_limit_m_s=100.0 / 3.6,
        transitions=(),
        price_kwh_per_s=1.0e-3,
    ).candidates[0]
    assert candidate.instance.start_position_m == 300.0
    assert abs(candidate.future_option_energy_kwh) < 1.0


def test_planner_and_executor_first_step_parity_without_hidden_information():
    selected = profile("accel_coast_brake_50_-0.5")
    selector = StatefulOptionSelector(profiles=(selected,))
    state = VehicleState(position_m=340.0, velocity_m_s=12.0, elapsed_time_s=10.0)
    option = OptionState.start(
        instance_id=1,
        profile=selected,
        state=VehicleState(position_m=300.0, velocity_m_s=12.0, elapsed_time_s=6.0),
        route_length_m=1000.0,
    )
    candidate = selector.solve(
        state,
        active_option=option,
        allow_continue=True,
        next_instance_id=2,
        current_limit_m_s=100.0 / 3.6,
        transitions=(),
        price_kwh_per_s=1.0e-3,
    ).candidates[0]
    fallback = frozen_fallback_action(
        state,
        current_limit_m_s=100.0 / 3.6,
        transitions=(),
        vehicle=selector.vehicle,
        config=selector.config,
    )
    execution_option = option.copy()
    execution_option.update_physical_state(selected, state, count_step=False)
    decision = selector.executor.decide(
        execution_option,
        selected,
        state,
        current_limit_m_s=100.0 / 3.6,
        transitions=(),
        fallback_action=fallback,
    )
    assert candidate.actions[0] == pytest.approx(decision.executed_action)
    assert candidate.phases[0] == decision.phase


def test_lifetime_timeout_is_distinct_and_holds_fallback():
    controller = TemporalOptionController(
        OptionArm.CONTINUE_FEEDBACK, profiles=(profile("cruise_50"),)
    )
    controller.reference.state = VehicleState(
        position_m=300.0, velocity_m_s=10.0, elapsed_time_s=10.0
    )
    controller._start("cruise_50")
    controller.active_option.native_steps_elapsed = 399
    observation = ordinary_observation(controller.state)
    action = controller.action(observation, 0.0)
    controller.observe_executed_action(action)
    assert controller.active_option is None
    assert controller.option_history[-1].termination_reason == "LIFETIME_TIMEOUT"
    assert controller._fallback_until_boundary


def test_completion_between_boundaries_holds_fallback_until_next_boundary():
    controller = TemporalOptionController(
        OptionArm.CONTINUE_FEEDBACK, profiles=(profile("cruise_50"),)
    )
    controller.reference.state = VehicleState(
        position_m=399.5, velocity_m_s=10.0, elapsed_time_s=10.0
    )
    controller.active_option = OptionState.start(
        instance_id=1,
        profile=profile("cruise_50"),
        state=VehicleState(position_m=300.0, velocity_m_s=10.0),
        route_length_m=1000.0,
    )
    observation = ordinary_observation(controller.state)
    action = controller.action(observation, 0.0)
    controller.observe_executed_action(action)
    assert controller.active_option is None
    assert controller.option_history[-1].termination_reason == "WAYPOINT_REACHED"
    next_observation = ordinary_observation(controller.state)
    fallback_action = controller.action(next_observation, 0.0)
    assert controller.last_decision.phase == "fallback"
    assert fallback_action == controller.last_decision.fallback_action


@pytest.mark.parametrize("position_m", [339.99, 340.01])
def test_snapshot_restore_exact_immediately_around_phase_boundary(position_m: float):
    selected = profile("accel_coast_brake_50_-0.5")
    controller = TemporalOptionController(
        OptionArm.CONTINUE_FEEDBACK, profiles=(selected,)
    )
    controller.reference.state = VehicleState(
        position_m=position_m, velocity_m_s=12.0, elapsed_time_s=10.0
    )
    controller.active_option = OptionState.start(
        instance_id=1,
        profile=selected,
        state=VehicleState(position_m=300.0, velocity_m_s=12.0),
        route_length_m=1000.0,
    )
    controller.active_option.update_physical_state(
        selected, controller.state, count_step=False
    )
    clone = TemporalOptionController(OptionArm.CONTINUE_FEEDBACK, profiles=(selected,))
    clone.restore(controller.snapshot())
    observation = ordinary_observation(controller.state)
    clone_observation = ordinary_observation(clone.state)
    action = controller.action(observation, 0.0)
    clone_action = clone.action(clone_observation, 0.0)
    assert action == clone_action
    controller.observe_executed_action(action)
    clone.observe_executed_action(clone_action)
    assert controller.snapshot() == clone.snapshot()


def test_snapshot_restore_exact_during_safety_interruption():
    selected = profile("accel_coast_60_0.5")
    controller = TemporalOptionController(
        OptionArm.CONTINUE_FEEDBACK, profiles=(selected,)
    )
    controller.reference.state = VehicleState(
        position_m=300.0, velocity_m_s=16.0, elapsed_time_s=10.0
    )
    controller.reference._known_transitions = {305.0: 20.0 / 3.6}
    controller.active_option = OptionState.start(
        instance_id=1,
        profile=selected,
        state=VehicleState(position_m=240.0, velocity_m_s=16.0),
        route_length_m=1000.0,
    )
    clone = TemporalOptionController(OptionArm.CONTINUE_FEEDBACK, profiles=(selected,))
    clone.restore(controller.snapshot())
    observation = ordinary_observation(controller.state)
    action = controller.action(observation, 0.0)
    clone_action = clone.action(observation.copy(), 0.0)
    assert action == clone_action
    assert controller.last_decision.intervention_reason in {
        "KNOWN_TRANSITION_BRAKING",
        "RIGHT_ENDPOINT_SPEED",
    }
    controller.observe_executed_action(action)
    clone.observe_executed_action(clone_action)
    assert controller.snapshot() == clone.snapshot()
    assert controller.option_history[-1].termination_reason == "SAFETY_INTERRUPTION"


def test_time_consistent_discount_uses_actual_native_duration():
    assert discount_for_duration(0.99, 7) == pytest.approx(0.99**7)
    assert discount_for_duration(0.99, 7) != pytest.approx(0.99**10)


def test_explicit_legacy_factory_is_original_controller():
    assert isinstance(build_controller("FROZEN_MPC_V2"), PriceMPCController)


def test_paper_is_rejected_before_construction():
    with pytest.raises(PermissionError, match="SEALED_UNINSPECTED"):
        require_not_paper(4000)


def test_environment_and_controller_checkpoint_restore_exact_next_actions():
    env = gym.make("StochasticTrack-v1", max_episode_steps=1800)
    observation, _ = env.reset(seed=10_700_000)
    controller = TemporalOptionController(OptionArm.CONTINUE_FEEDBACK)
    price = AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
    held = None
    for step in range(13):
        if step % 10 == 0:
            held = price.action(controller.prepare_strategy_step(observation))
        action = controller.action(observation, held)
        observation, _, terminated, truncated, _ = env.step([action])
        controller.observe_executed_action(
            action, episode_ended=terminated or truncated
        )
    payload = capture_checkpoint(
        environment=env,
        controller=controller,
        observation=observation,
        held_strategy_action=held,
        decision_clock=13,
        partial_transition={"reward": 1.25},
        counters={"native_steps": 13},
    )
    clone = TemporalOptionController(OptionArm.CONTINUE_FEEDBACK)
    restored_env, restored_observation, restored_held, clock, partial, counters = (
        restore_checkpoint(pickle.loads(pickle.dumps(payload)), clone)
    )
    assert clock == 13 and partial == {"reward": 1.25}
    assert counters == {"native_steps": 13}
    original_actions = []
    restored_actions = []
    for step in range(13, 20):
        if step % 10 == 0:
            held = price.action(controller.prepare_strategy_step(observation))
            restored_held = price.action(
                clone.prepare_strategy_step(restored_observation)
            )
        original = controller.action(observation, held)
        restored = clone.action(restored_observation, restored_held)
        original_actions.append(original)
        restored_actions.append(restored)
        observation, _, t1, x1, info1 = env.step([original])
        restored_observation, _, t2, x2, info2 = restored_env.step([restored])
        controller.observe_executed_action(original, episode_ended=t1 or x1)
        clone.observe_executed_action(restored, episode_ended=t2 or x2)
        assert info1["position_m"] == pytest.approx(info2["position_m"], abs=0.0)
        assert info1["total_energy_kwh"] == pytest.approx(
            info2["total_energy_kwh"], abs=0.0
        )
    assert original_actions == restored_actions
    assert controller.snapshot() == clone.snapshot()
    env.close()
    restored_env.close()
