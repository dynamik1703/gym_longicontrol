"""Deterministic correctness fixtures for the Hybrid architecture."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import gymnasium as gym

from benchmarks.physical_feasibility_reference.causal_controller import (
    ObservationLimitedController,
)
from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from benchmarks.physical_feasibility_reference.evaluation import canonical_sha256
from benchmarks.physical_feasibility_reference.feasibility import CANONICAL_TASK
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import is_feasible
from gym_longicontrol.domain.track import Track
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from .baseline import TimeAwareEnergyStrategy
from .controller import HybridTargetSpeedController

PARITY_FIXTURES = (
    "constant_100_km_h",
    "upward_changes",
    "maximum_legal_downward_change",
    "consecutive_downward_changes",
    "short_braking_window",
    "route_endpoint_at_speed_boundary",
)


def _environment(track: Track):
    base = LongiControlEnv(track_generator=lambda _rng: track)
    return gym.wrappers.TimeLimit(base, max_episode_steps=1_800)


def _trace(info: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(info["position_m"]),
        float(info["velocity_m_s"]),
        float(info["acceleration_m_s2"]),
        float(info["elapsed_time_s"]),
        float(info["total_energy_kwh"]),
        float(info["speed_limit_m_s"]),
    )


def _rollout(
    track: Track,
    controller: ObservationLimitedController | HybridTargetSpeedController,
    strategy: Callable[[HybridTargetSpeedController, int], float] | None = None,
) -> dict[str, Any]:
    environment = _environment(track)
    actions: list[float] = []
    trace: list[tuple[float, ...]] = []
    decisions: list[dict[str, Any]] = []
    try:
        observation, info = environment.reset(seed=123)
        controller.reset()
        ended = False
        step = 0
        while not ended:
            if isinstance(controller, HybridTargetSpeedController):
                strategy_action = strategy(controller, step) if strategy else None
                action = controller.action(observation, strategy_action)
            else:
                action = controller.action(observation)
            observation, _, terminated, truncated, info = environment.step([action])
            controller.observe_executed_action(action)
            actions.append(float(action))
            trace.append(_trace(info))
            if isinstance(controller, HybridTargetSpeedController):
                assert controller.last_decision is not None
                decisions.append(controller.last_decision.as_dict())
            step += 1
            ended = bool(terminated or truncated)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        return {
            "actions": actions,
            "trace": trace,
            "action_sha256": canonical_sha256(actions),
            "trace_sha256": canonical_sha256(trace),
            "metrics": asdict(metrics),
            "feasible": is_feasible(metrics, CANONICAL_TASK),
            "final_position_m": float(info["position_m"]),
            "decisions": decisions,
        }
    finally:
        environment.close()


def fallback_parity_preflight() -> dict[str, Any]:
    results = []
    for name in PARITY_FIXTURES:
        track = physical_fixtures()[name]
        frozen = _rollout(track, ObservationLimitedController())
        hybrid = _rollout(
            track, HybridTargetSpeedController(strategy_enabled=False)
        )
        matching = {
            "actions": frozen["actions"] == hybrid["actions"],
            "action_sha256": frozen["action_sha256"] == hybrid["action_sha256"],
            "state_trace": frozen["trace"] == hybrid["trace"],
            "trace_sha256": frozen["trace_sha256"] == hybrid["trace_sha256"],
            "metrics": frozen["metrics"] == hybrid["metrics"],
            "completion": frozen["metrics"]["completed"]
            == hybrid["metrics"]["completed"],
            "deadline": frozen["metrics"]["travel_time_s"]
            == hybrid["metrics"]["travel_time_s"],
            "speed": frozen["metrics"]["max_speed_violation_m_s"]
            == hybrid["metrics"]["max_speed_violation_m_s"],
            "energy": frozen["metrics"]["energy_kwh"]
            == hybrid["metrics"]["energy_kwh"],
        }
        results.append(
            {
                "fixture": name,
                "matching": matching,
                "all_matching": all(matching.values()),
                "action_sha256": frozen["action_sha256"],
                "trace_sha256": frozen["trace_sha256"],
                "steps": len(frozen["actions"]),
            }
        )
    return {
        "fixture_count": len(results),
        "all_exact": all(row["all_matching"] for row in results),
        "fixtures": results,
    }


def shield_preflight() -> dict[str, Any]:
    rows = []
    total_proposals = 0
    total_overrides = 0
    total_influenced = 0
    total_prediction_calls = 0
    viability_falsifications = []
    for name in PARITY_FIXTURES:
        strategy = TimeAwareEnergyStrategy()
        strategy.reset()
        track = physical_fixtures()[name]
        controller = HybridTargetSpeedController(strategy_enabled=True)

        def target(hybrid: HybridTargetSpeedController, step: int) -> float:
            return strategy.action(
                hybrid.state, step, route_length_m=hybrid.config.track_length_m
            )

        result = _rollout(track, controller, target)
        decisions = result["decisions"]
        overrides = sum(row["overridden"] for row in decisions)
        influenced = sum(row["native_action_influence"] > 0.0 for row in decisions)
        prediction_calls = sum(row["physics_prediction_calls"] for row in decisions)
        final_time = float(result["metrics"]["travel_time_s"])
        for index, decision in enumerate(decisions):
            if (
                decision["accepted"]
                and decision["deadline_viable"]
                and final_time > decision["completion_bound_s"] + 0.2
            ):
                viability_falsifications.append(
                    {
                        "fixture": name,
                        "step": index,
                        "observed_completion_s": final_time,
                        "predicted_bound_s": decision["completion_bound_s"],
                    }
                )
        rows.append(
            {
                "fixture": name,
                "feasible": result["feasible"],
                "completion": result["metrics"]["completed"],
                "deadline_compliant": final_time <= 140.0,
                "speed_compliant": (
                    result["metrics"]["max_speed_violation_m_s"] <= 0.0
                ),
                "energy_kwh": result["metrics"]["energy_kwh"],
                "travel_time_s": final_time,
                "proposal_count": len(decisions),
                "override_count": overrides,
                "influenced_step_count": influenced,
                "intervention_reasons": dict(
                    sorted(
                        Counter(
                            row["intervention_reason"] for row in decisions
                        ).items()
                    )
                ),
                "physics_prediction_calls": prediction_calls,
            }
        )
        total_proposals += len(decisions)
        total_overrides += overrides
        total_influenced += influenced
        total_prediction_calls += prediction_calls
    override_fraction = total_overrides / total_proposals
    influence_fraction = total_influenced / total_proposals
    influenced_fixtures = sum(row["influenced_step_count"] > 0 for row in rows)
    thresholds = {
        "maximum_override_fraction": 0.85,
        "minimum_influence_fraction": 0.10,
        "minimum_influenced_fixture_count": 4,
    }
    shield_dominates = bool(
        override_fraction > thresholds["maximum_override_fraction"]
        or influence_fraction < thresholds["minimum_influence_fraction"]
        or influenced_fixtures < thresholds["minimum_influenced_fixture_count"]
    )
    return {
        "fixture_count": len(rows),
        "fixtures": rows,
        "proposal_count": total_proposals,
        "override_count": total_overrides,
        "override_fraction": override_fraction,
        "influenced_step_count": total_influenced,
        "influence_fraction": influence_fraction,
        "influenced_fixture_count": influenced_fixtures,
        "physics_prediction_calls": total_prediction_calls,
        "mean_physics_prediction_calls_per_proposal": (
            total_prediction_calls / total_proposals
        ),
        "viability_falsifications": viability_falsifications,
        "thresholds": thresholds,
        "SHIELD_DOMINATES_POLICY": shield_dominates,
        "endpoint_safety_passed": all(row["speed_compliant"] for row in rows),
        "deadline_fixture_passed": all(row["deadline_compliant"] for row in rows),
        "feasibility_fixture_passed": all(row["feasible"] for row in rows),
    }


def run_preflight() -> dict[str, Any]:
    parity = fallback_parity_preflight()
    shield = shield_preflight()
    return {
        "schema_version": 1,
        "scope": "deterministic physical fixtures only; no reserved evaluation tasks",
        "fallback_parity": parity,
        "shield": shield,
        "causal_information_boundary": {
            "controller_inputs": ["ordinary 8D observation", "scalar strategy action"],
            "environment_object_input": False,
            "track_input": False,
            "simulator_state_input": False,
            "oracle_input": False,
            "future_realized_state_input": False,
        },
        "passed": bool(
            parity["all_exact"]
            and shield["endpoint_safety_passed"]
            and shield["deadline_fixture_passed"]
            and shield["feasibility_fixture_passed"]
            and not shield["viability_falsifications"]
            and not shield["SHIELD_DOMINATES_POLICY"]
        ),
    }
