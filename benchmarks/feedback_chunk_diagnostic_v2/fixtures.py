"""Explicitly non-reserved physical fixtures for cross-runtime parity."""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym

from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.causal_mpc_v2.controller import PriceMPCController
from benchmarks.causal_mpc_v2.fixed_price import FixedPrice
from benchmarks.hybrid_rl_design.baseline import TimeAwareEnergyStrategy
from benchmarks.hybrid_rl_design.controller import HybridTargetSpeedController
from benchmarks.physical_feasibility_reference.causal_controller import (
    ObservationLimitedController,
)
from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from gym_longicontrol.envs.longicontrol import LongiControlEnv

FIXTURE_NAMES = ("maximum_legal_downward_change", "consecutive_downward_changes")
CONTROLLERS = (
    "CAUSAL60",
    "TIME_AWARE",
    "FIXED_PRICE_MPC_V2",
    "ADAPTIVE_PRICE_MPC_V2",
)


def _rollout(fixture_name: str, controller_id: str) -> dict[str, Any]:
    if fixture_name not in FIXTURE_NAMES or controller_id not in CONTROLLERS:
        raise PermissionError("synthetic fixture assignment is not preregistered")
    track = physical_fixtures()[fixture_name]
    environment = gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=lambda _rng: track), max_episode_steps=1_800
    )
    if controller_id == "CAUSAL60":
        controller: Any = ObservationLimitedController()
        strategy: Any = None
        kind = "causal"
    elif controller_id == "TIME_AWARE":
        controller = HybridTargetSpeedController(strategy_enabled=True)
        strategy = TimeAwareEnergyStrategy()
        kind = "hybrid"
    else:
        controller = PriceMPCController()
        strategy = (
            FixedPrice(0.0)
            if controller_id == "FIXED_PRICE_MPC_V2"
            else AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
        )
        kind = "price"
    rows = []
    started = time.perf_counter()
    try:
        observation, _ = environment.reset(seed=20_261_025)
        controller.reset()
        if strategy is not None and hasattr(strategy, "reset"):
            strategy.reset()
        held = None
        step = 0
        ended = False
        while not ended:
            if kind in ("hybrid", "price") and step % 10 == 0:
                strategy_observation = controller.prepare_strategy_step(observation)
                if kind == "hybrid":
                    held = strategy.action(
                        controller.state,
                        step,
                        route_length_m=controller.config.track_length_m,
                    )
                else:
                    held = strategy.action(strategy_observation)
            if kind == "causal":
                action = controller.action(observation)
                decision = None
            else:
                action = controller.action(observation, held)
                decision = controller.last_decision.as_dict()
            next_observation, _, terminated, truncated, info = environment.step(
                [action]
            )
            controller.observe_executed_action(action)
            rows.append(
                {
                    "native_step": step,
                    "action": float(action),
                    "successor": {
                        key: float(info[key])
                        for key in (
                            "position_m",
                            "velocity_m_s",
                            "acceleration_m_s2",
                            "elapsed_time_s",
                            "total_energy_kwh",
                            "speed_limit_m_s",
                        )
                    },
                    "step_energy_kwh": float(info["step_energy_kwh"]),
                    "decision": decision,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            observation = next_observation
            step += 1
            ended = bool(terminated or truncated)
        return {
            "fixture": fixture_name,
            "controller_id": controller_id,
            "rows": rows,
            "metrics": dict(info["episode_metrics"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        environment.close()


def run_fixture_suite() -> dict[str, Any]:
    rows = [
        _rollout(fixture, controller)
        for fixture in FIXTURE_NAMES
        for controller in CONTROLLERS
    ]
    return {
        "scope": "synthetic fixed tracks; no reserved Diagnostic/Validation/Paper task",
        "fixtures": list(FIXTURE_NAMES),
        "controllers": list(CONTROLLERS),
        "rollouts": rows,
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
        "native_steps": sum(len(row["rows"]) for row in rows),
    }
