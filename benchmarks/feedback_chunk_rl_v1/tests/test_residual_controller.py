from __future__ import annotations

from dataclasses import asdict

import gymnasium as gym
import numpy as np

from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.causal_mpc_v2.controller import PriceMPCController
from benchmarks.feedback_chunk_rl_v1.residual_controller import (
    FeedbackResidualController,
)
from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.track import Track
from gym_longicontrol.envs.longicontrol import LongiControlEnv


def _environment(track: Track) -> gym.Env:
    return gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=lambda _rng: track), max_episode_steps=1_800
    )


def _trace(info: dict) -> tuple[float, ...]:
    return (
        info["position_m"],
        info["velocity_m_s"],
        info["acceleration_m_s2"],
        info["total_energy_kwh"],
        info["speed_limit_m_s"],
        info["elapsed_time_s"],
    )


def _baseline_rollout(
    track: Track,
) -> tuple[list[float], list[tuple[float, ...]], list[dict], dict]:
    env = _environment(track)
    controller = PriceMPCController()
    selector = AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
    actions: list[float] = []
    traces: list[tuple[float, ...]] = []
    decisions: list[dict] = []
    try:
        observation, _ = env.reset(seed=20261023)
        controller.reset()
        selector.reset()
        held = 1.0
        ended = False
        step = 0
        while not ended:
            if step % 10 == 0:
                held = selector.action(controller.prepare_strategy_step(observation))
            action = controller.action(observation, held)
            observation, _, terminated, truncated, info = env.step([action])
            controller.observe_executed_action(action)
            actions.append(action)
            traces.append(_trace(info))
            decisions.append(controller.last_decision.as_dict())
            ended = bool(terminated or truncated)
            step += 1
        return (
            actions,
            traces,
            decisions,
            asdict(EpisodeMetrics(**info["episode_metrics"])),
        )
    finally:
        env.close()


def _zero_residual_rollout(
    track: Track,
) -> tuple[list[float], list[tuple[float, ...]], list[dict], dict]:
    env = _environment(track)
    controller = FeedbackResidualController("CHUNK3_RESIDUAL_SAC")
    actions: list[float] = []
    traces: list[tuple[float, ...]] = []
    decisions: list[dict] = []
    try:
        observation, _ = env.reset(seed=20261023)
        controller.reset()
        ended = False
        while not ended:
            actor_observation = controller.prepare_native_step(observation)
            if actor_observation is not None:
                assert actor_observation.shape == (16,)
                controller.begin_chunk(np.zeros(3))
            action = controller.action(observation)
            observation, _, terminated, truncated, info = env.step([action])
            controller.observe_executed_action(action)
            actions.append(action)
            traces.append(_trace(info))
            decisions.append(controller.last_decision.mpc_decision)
            ended = bool(terminated or truncated)
        return (
            actions,
            traces,
            decisions,
            asdict(EpisodeMetrics(**info["episode_metrics"])),
        )
    finally:
        env.close()


def test_zero_residual_exactly_matches_adaptive_price_mpc() -> None:
    fixtures = physical_fixtures()
    for name in (
        "constant_100_km_h",
        "maximum_legal_downward_change",
        "consecutive_downward_changes",
        "route_endpoint_at_speed_boundary",
    ):
        baseline = _baseline_rollout(fixtures[name])
        residual = _zero_residual_rollout(fixtures[name])
        assert residual == baseline, name


def test_chunk_components_are_requested_in_their_own_seconds() -> None:
    km_h = 1.0 / 3.6
    track = Track(
        np.array([0.0, 1000.0]),
        np.array([100.0, 100.0]) * km_h,
    )
    env = _environment(track)
    controller = FeedbackResidualController("CHUNK3_RESIDUAL_SAC")
    requested: list[tuple[int, float]] = []
    influenced: list[tuple[int, float, float]] = []
    try:
        observation, _ = env.reset(seed=8)
        controller.reset()
        for _ in range(500):
            actor_observation = controller.prepare_native_step(observation)
            if actor_observation is not None:
                controller.begin_chunk(np.array([-0.2, -0.4, -0.6]))
            action = controller.action(observation)
            requested.append(
                (
                    controller.last_decision.chunk_phase,
                    controller.last_decision.requested_residual,
                )
            )
            if controller.last_decision.residual_influence > 1.0e-15:
                influenced.append(
                    (
                        controller.last_decision.chunk_phase,
                        controller.last_decision.requested_residual,
                        action - controller.last_decision.mpc_executed_action,
                    )
                )
            observation, _, terminated, truncated, _ = env.step([action])
            controller.observe_executed_action(action)
            if terminated or truncated:
                break
        for start in range(0, len(requested) - 29, 30):
            assert requested[start] == (0, -0.2)
            assert requested[start + 10] == (1, -0.4)
            assert requested[start + 20] == (2, -0.6)
        assert {row[0] for row in influenced} == {0, 1, 2}
        assert {row[1] for row in influenced} == {-0.2, -0.4, -0.6}
        assert all(delta < 0.0 for _, _, delta in influenced)
    finally:
        env.close()


def test_new_transition_cancels_remainder_until_scheduled_boundary() -> None:
    km_h = 1.0 / 3.6
    # Position 516 m places first visibility inside seconds two/three for this
    # deterministic rollout rather than on a scheduled 3 s actor boundary.
    track = Track(
        np.array([0.0, 516.0]),
        np.array([100.0, 60.0]) * km_h,
    )
    env = _environment(track)
    controller = FeedbackResidualController("CHUNK3_RESIDUAL_SAC")
    cancellations: list[dict] = []
    try:
        observation, _ = env.reset(seed=9)
        controller.reset()
        ended = False
        while not ended and controller.native_step < 600:
            actor_observation = controller.prepare_native_step(observation)
            if actor_observation is not None:
                controller.begin_chunk(np.zeros(3))
            action = controller.action(observation)
            decision = controller.last_decision
            if decision.newly_visible_transition and decision.chunk_phase > 0:
                cancellations.append(decision.as_dict())
            observation, _, terminated, truncated, _ = env.step([action])
            controller.observe_executed_action(action)
            ended = bool(terminated or truncated)
        assert cancellations
        assert all(row["chunk_cancelled"] for row in cancellations)
        assert all(
            row["cancellation_reason"] == "NEWLY_VISIBLE_SPEED_LIMIT_TRANSITION"
            for row in cancellations
        )
    finally:
        env.close()
