"""Deterministic adversarial shield and deadline-viability preparation audit."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from benchmarks.hybrid_rl_design.controller import HybridTargetSpeedController
from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.domain.track import Track
from gym_longicontrol.envs.longicontrol import LongiControlEnv

CANONICAL_TASK = TaskSpecification(140.0, 0.0)


def adversarial_fixtures() -> dict[str, Track]:
    km_h = 1.0 / 3.6
    return {
        "repeated_maximum_reductions": Track(
            np.array([0, 250, 350, 500, 600, 750, 850], dtype=np.float64),
            np.array([60, 20, 60, 20, 60, 20, 60], dtype=np.float64) * km_h,
        ),
        "slow_sections": Track(
            np.array([0, 200, 300, 500, 600, 800, 900], dtype=np.float64),
            np.array([50, 20, 50, 20, 50, 20, 50], dtype=np.float64) * km_h,
        ),
        "consecutive_transitions": Track(
            np.array([0, 300, 400, 500, 600], dtype=np.float64),
            np.array([60, 50, 30, 20, 60], dtype=np.float64) * km_h,
        ),
        "low_speed_acceleration": Track(
            np.array([0, 100, 700], dtype=np.float64),
            np.array([20, 60, 20], dtype=np.float64) * km_h,
        ),
        "interrupted_endpoint_block": physical_fixtures()[
            "route_endpoint_at_speed_boundary"
        ],
    }


def _constant(value: float) -> Callable[[int], float]:
    return lambda _decision: value


def _gradual(decision: int) -> float:
    return float(max(-1.0, 1.0 - 0.08 * decision))


def _fallback_after_slow(decision: int) -> float:
    return -1.0 if decision < 25 else 1.0


def _rollout(track: Track, schedule: Callable[[int], float]) -> dict[str, Any]:
    environment = gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=lambda _rng: track), max_episode_steps=1_800
    )
    controller = HybridTargetSpeedController(strategy_enabled=True)
    decisions: list[dict[str, Any]] = []
    macro_targets: list[float] = []
    viability_falsifications = []
    try:
        observation, info = environment.reset(seed=20261020)
        controller.reset()
        step = 0
        macro = 0
        held = 1.0
        ended = False
        while not ended:
            if step % 10 == 0:
                controller.prepare_strategy_step(observation)
                held = float(schedule(macro))
                macro_targets.append(held)
                macro += 1
            action = controller.action(observation, held)
            observation, _, terminated, truncated, info = environment.step([action])
            controller.observe_executed_action(action)
            assert controller.last_decision is not None
            decisions.append(controller.last_decision.as_dict())
            step += 1
            ended = bool(terminated or truncated)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        for index, decision in enumerate(decisions):
            bound = decision["completion_bound_s"]
            if (
                decision["accepted"]
                and decision["deadline_viable"]
                and bound is not None
                and metrics.travel_time_s > float(bound) + 0.2
            ):
                viability_falsifications.append(
                    {
                        "native_step": index,
                        "completion_bound_s": bound,
                        "observed_completion_s": metrics.travel_time_s,
                    }
                )
        return {
            **metrics.__dict__,
            "feasible": is_feasible(metrics, CANONICAL_TASK),
            "macro_decision_count": len(macro_targets),
            "native_proposal_count": len(decisions),
            "override_count": sum(row["overridden"] for row in decisions),
            "influenced_native_count": sum(
                row["native_action_influence"] > 0.0 for row in decisions
            ),
            "intervention_reasons": dict(
                sorted(Counter(row["intervention_reason"] for row in decisions).items())
            ),
            "minimum_target": min(macro_targets),
            "viability_falsifications": viability_falsifications,
        }
    finally:
        environment.close()


def run_safety_audit() -> dict[str, Any]:
    fixtures = adversarial_fixtures()
    cases = (
        ("repeated_maximum_reductions", _constant(-1.0)),
        ("slow_sections", _constant(-1.0)),
        ("consecutive_transitions", _constant(-1.0)),
        ("consecutive_transitions", _gradual),
        ("repeated_maximum_reductions", _fallback_after_slow),
        ("low_speed_acceleration", _constant(-1.0)),
        ("interrupted_endpoint_block", _constant(-1.0)),
    )
    rows = []
    for index, (fixture_name, schedule) in enumerate(cases):
        result = _rollout(fixtures[fixture_name], schedule)
        rows.append(
            {
                "case": index,
                "fixture": fixture_name,
                "schedule": schedule.__name__,
                **result,
            }
        )
    falsifications = [
        {"fixture": row["fixture"], **failure}
        for row in rows
        for failure in row["viability_falsifications"]
    ]
    return {
        "scope": (
            "deterministic unreserved adversarial fixtures; evidence, not a "
            "global shield proof"
        ),
        "rows": rows,
        "exact_right_endpoint_speed_passed": all(
            row["max_speed_violation_m_s"] <= 0.0 for row in rows
        ),
        "eventual_completion_passed": all(row["completed"] for row in rows),
        "deadline_passed": all(row["travel_time_s"] <= 140.0 for row in rows),
        "feasibility_passed": all(row["feasible"] for row in rows),
        "viability_falsifications": falsifications,
        "passed": bool(all(row["feasible"] for row in rows) and not falsifications),
    }
