from __future__ import annotations

import inspect
import math

import gymnasium as gym
import numpy as np

from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from benchmarks.slacktrader_v1.controller import PriceMPCController
from benchmarks.slacktrader_v1.planner import FixedDistancePlanner
from benchmarks.slacktrader_v1.time_price import (
    LAMBDA_MAX_KWH_PER_S,
    LAMBDA_MIN_KWH_PER_S,
    lambda_to_normalized_action,
    normalized_action_to_lambda,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.domain.track import Track
from gym_longicontrol.envs.longicontrol import LongiControlEnv


def test_log_price_mapping_has_physical_units_and_is_monotonic() -> None:
    actions = np.linspace(-1.0, 1.0, 21)
    prices = [normalized_action_to_lambda(value) for value in actions]
    assert prices[0] == LAMBDA_MIN_KWH_PER_S
    assert math.isclose(prices[-1], LAMBDA_MAX_KWH_PER_S, rel_tol=1e-15)
    assert all(left < right for left, right in zip(prices, prices[1:]))
    for action, price in zip(actions, prices):
        assert math.isclose(lambda_to_normalized_action(price), action, abs_tol=1e-12)


def test_fixed_distance_objective_has_positive_progress_and_price_authority() -> None:
    planner = FixedDistancePlanner()
    state = VehicleState(
        position_m=400.0,
        velocity_m_s=13.888,
        acceleration_m_s2=0.0,
        elapsed_time_s=20.0,
    )
    transitions = (Transition(500.0, 13.888),)
    plans = [
        planner.solve(
            state,
            current_limit_m_s=16.666,
            transitions=transitions,
            price_kwh_per_s=normalized_action_to_lambda(action),
        )
        for action in (-1.0, 0.0, 1.0)
    ]
    durations = [plan.selected.travel_time_s for plan in plans]
    assert all(plan.selected.feasible for plan in plans)
    assert all(
        plan.selected.final_position_m >= plan.waypoint_position_m for plan in plans
    )
    assert all(plan.planning_distance_m == 100.0 for plan in plans)
    assert durations[0] >= durations[1] >= durations[2]
    assert len({round(value, 9) for value in durations}) >= 2
    assert plans[0].segment_actions[0] != plans[-1].segment_actions[0]


def test_terminal_value_charges_low_terminal_speed_and_cannot_stop_early() -> None:
    planner = FixedDistancePlanner()
    plan = planner.solve(
        VehicleState(),
        current_limit_m_s=60.0 / 3.6,
        transitions=(),
        price_kwh_per_s=LAMBDA_MIN_KWH_PER_S,
    )
    low = plan.candidates[0]
    assert low.terminal_energy_kwh > 0.0
    assert low.final_position_m >= plan.waypoint_position_m
    assert low.travel_time_s > 0.0
    assert low.actions


def test_controller_api_has_no_environment_track_or_seed_input() -> None:
    signature = inspect.signature(PriceMPCController.action)
    assert tuple(signature.parameters) == ("self", "observation", "strategy_action")
    source = inspect.getsource(PriceMPCController)
    assert "environment.unwrapped.track" not in source
    assert "reset_seed" not in source


def test_right_endpoint_known_braking_and_exact_zero_speed_tolerance() -> None:
    config = SimulationConfig(track_length_m=300.0)
    track = Track(
        np.array([0.0, 200.0]),
        np.array([60.0, 20.0]) / 3.6,
    )
    environment = gym.wrappers.TimeLimit(
        LongiControlEnv(
            config=config,
            track_generator=lambda _rng: track,
        ),
        max_episode_steps=1_800,
    )
    controller = PriceMPCController(config=config)
    try:
        observation, info = environment.reset(seed=123)
        controller.reset()
        step = 0
        held = 1.0
        saw_new_information_replan = False
        ended = False
        while not ended:
            if step % 10 == 0:
                controller.prepare_strategy_step(observation)
            action = controller.action(observation, held)
            saw_new_information_replan |= bool(
                controller.last_decision is not None
                and controller.last_decision.replanned
                and step % 10 != 0
            )
            observation, _, terminated, truncated, info = environment.step([action])
            controller.observe_executed_action(action)
            step += 1
            ended = bool(terminated or truncated)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        assert metrics.max_speed_violation_m_s <= 0.0
        assert metrics.completed
        assert saw_new_information_replan
        assert is_feasible(metrics, TaskSpecification(140.0, 0.0))
    finally:
        environment.close()
