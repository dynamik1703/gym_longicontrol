"""Development-only capability evaluation and unopened-panel guards."""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from benchmarks.hybrid_rl_execution.evaluation import (
    evaluate_seed as evaluate_hybrid_seed,
)
from benchmarks.physical_feasibility_reference.causal_controller import (
    decode_observation,
)
from benchmarks.physical_feasibility_reference.evaluation import solve_oracle
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .adaptive_price import AdaptivePrice
from .controller import PriceMPCController
from .fixed_price import FixedPrice
from .integrity import require_not_paper, require_validation_authorized
from .split import canonical_sha256
from .time_price import marginal_price, normalized_action_to_lambda

HERE = Path(__file__).resolve().parent
SPLIT_LEDGER = HERE / "split_ledger.json"
CANONICAL_TASK = TaskSpecification(140.0, 0.0)
DECISION_INTERVAL_NATIVE_STEPS = 10


class PriceSelector(Protocol):
    def reset(self) -> None: ...

    def action(self, strategy_observation: np.ndarray) -> float: ...


def _environment() -> gym.Env:
    return gym.wrappers.TimeLimit(StochasticTrack(), max_episode_steps=1_800)


def evaluate_price_seed(
    *, numeric_seed: int, selector: PriceSelector
) -> dict[str, Any]:
    """Run one permitted task without exposing its Track to the controller."""

    require_not_paper(numeric_seed)
    environment = _environment()
    controller = PriceMPCController()
    selector.reset()
    prices: list[float] = []
    normalized_prices: list[float] = []
    selected_speeds: list[float] = []
    native_actions: list[float] = []
    solve_times: list[float] = []
    decisions: list[dict[str, Any]] = []
    try:
        observation, info = environment.reset(seed=int(numeric_seed))
        controller.reset()
        held_price: float | None = None
        step = 0
        ended = False
        while not ended:
            if step % DECISION_INTERVAL_NATIVE_STEPS == 0:
                strategy_observation = controller.prepare_strategy_step(observation)
                held_price = float(selector.action(strategy_observation))
                normalized_prices.append(held_price)
                prices.append(normalized_action_to_lambda(held_price))
            assert held_price is not None
            started = time.perf_counter()
            native_action = controller.action(observation, held_price)
            control_time = time.perf_counter() - started
            if (
                controller.last_decision is not None
                and controller.last_decision.replanned
            ):
                solve_times.append(control_time)
                if controller.last_decision.planner_target_speed_m_s is not None:
                    selected_speeds.append(
                        controller.last_decision.planner_target_speed_m_s
                    )
            observation, _, terminated, truncated, info = environment.step(
                [native_action]
            )
            controller.observe_executed_action(native_action)
            assert controller.last_decision is not None
            decisions.append(controller.last_decision.as_dict())
            native_actions.append(float(native_action))
            step += 1
            ended = bool(terminated or truncated)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        feasible = is_feasible(metrics, CANONICAL_TASK)
        return {
            **asdict(metrics),
            "numeric_seed": int(numeric_seed),
            "feasible": feasible,
            "deadline_compliant": metrics.travel_time_s <= 140.0,
            "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
            "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
            "normalized_price_actions": normalized_prices,
            "prices_kwh_per_s": prices,
            "selected_target_speeds_m_s": selected_speeds,
            "native_action_sha256": canonical_sha256(native_actions),
            "planner_solve_count": len(solve_times),
            "planner_solve_time_s": sum(solve_times),
            "mean_planner_solve_time_s": (
                statistics.fmean(solve_times) if solve_times else 0.0
            ),
            "maximum_planner_solve_time_s": max(solve_times, default=0.0),
            "physics_prediction_calls": sum(
                int(row["physics_prediction_calls"]) for row in decisions
            ),
            "proposal_count": len(decisions),
            "override_count": sum(bool(row["overridden"]) for row in decisions),
            "fallback_frequency": (
                sum(bool(row["overridden"]) for row in decisions) / len(decisions)
                if decisions
                else 0.0
            ),
            "native_action_influence_count": sum(
                float(row["native_action_influence"]) > 0.0 for row in decisions
            ),
            "intervention_reasons": dict(
                sorted(Counter(row["intervention_reason"] for row in decisions).items())
            ),
            "nonfinite_value_count": 0,
        }
    finally:
        environment.close()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["feasible"]]
    return {
        "episode_count": len(rows),
        "rsr": len(feasible),
        "completion_count": sum(bool(row["completed"]) for row in rows),
        "deadline_compliance_count": sum(
            bool(row["deadline_compliant"]) for row in rows
        ),
        "speed_compliance_count": sum(bool(row["speed_compliant"]) for row in rows),
        "mean_feasible_energy_kwh": (
            statistics.fmean(float(row["energy_kwh"]) for row in feasible)
            if feasible
            else None
        ),
        "mean_travel_time_s": statistics.fmean(
            float(row["travel_time_s"]) for row in rows
        ),
        "mean_planner_solve_time_s": (
            statistics.fmean(
                float(row.get("mean_planner_solve_time_s", 0.0)) for row in rows
            )
            if rows
            else 0.0
        ),
        "optimizer_feasibility_failure_count": sum(
            row.get("intervention_reasons", {}).get(
                "PLANNER_INFEASIBLE_FALLBACK", 0
            )
            for row in rows
        ),
        "override_count": sum(int(row.get("override_count", 0)) for row in rows),
        "proposal_count": sum(int(row.get("proposal_count", 0)) for row in rows),
    }


def _calibration_score(rows: list[dict[str, Any]]) -> tuple[int, float, float]:
    feasible = [row for row in rows if row["feasible"]]
    energy = (
        statistics.fmean(float(row["energy_kwh"]) for row in feasible)
        if feasible
        else float("inf")
    )
    travel = statistics.fmean(float(row["travel_time_s"]) for row in rows)
    return (-len(feasible), energy, travel)


def calibrate_fixed_price(seeds: list[int]) -> dict[str, Any]:
    candidates = (-1.0, -0.5, 0.0, 0.5, 1.0)
    rows = []
    for action in candidates:
        outcomes = [
            evaluate_price_seed(numeric_seed=seed, selector=FixedPrice(action))
            for seed in seeds
        ]
        rows.append(
            {
                "normalized_action": action,
                "price_kwh_per_s": normalized_action_to_lambda(action),
                "score": _calibration_score(outcomes),
                "summary": _summary(outcomes),
            }
        )
    selected = min(
        rows, key=lambda row: (tuple(row["score"]), row["normalized_action"])
    )
    return {
        "procedure": (
            "finite five-point log-price grid; maximize RSR, then minimize "
            "feasible energy, then travel time"
        ),
        "seeds": seeds,
        "candidates": rows,
        "selected_normalized_action": selected["normalized_action"],
        "selected_price_kwh_per_s": selected["price_kwh_per_s"],
    }


def calibrate_adaptive_price(seeds: list[int]) -> dict[str, Any]:
    candidates = ((12.0, 2.0), (18.0, 2.0), (24.0, 2.0))
    rows = []
    for high, low in candidates:
        outcomes = [
            evaluate_price_seed(
                numeric_seed=seed,
                selector=AdaptivePrice(high_reserve_s=high, low_reserve_s=low),
            )
            for seed in seeds
        ]
        rows.append(
            {
                "high_reserve_s": high,
                "low_reserve_s": low,
                "score": _calibration_score(outcomes),
                "summary": _summary(outcomes),
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            tuple(row["score"]),
            row["high_reserve_s"],
            row["low_reserve_s"],
        ),
    )
    return {
        "procedure": (
            "finite reserve-threshold grid; maximize RSR, then minimize "
            "feasible energy, then travel time"
        ),
        "seeds": seeds,
        "candidates": rows,
        "selected_high_reserve_s": selected["high_reserve_s"],
        "selected_low_reserve_s": selected["low_reserve_s"],
    }


def counterfactual_price_sensitivity(numeric_seed: int) -> dict[str, Any]:
    """Find the first causal route state where the price changes the plan.

    All three counterfactual solves receive an identical controller state,
    visible-transition memory and candidate set.  A neutral-price controller is
    used only to causally reach successive states on a Development task.
    """

    require_not_paper(numeric_seed)
    environment = _environment()
    controller = PriceMPCController()
    try:
        observation, _ = environment.reset(seed=int(numeric_seed))
        controller.reset()
        step = 0
        held = 0.0
        best: dict[str, Any] | None = None
        ended = False
        while not ended:
            if step % DECISION_INTERVAL_NATIVE_STEPS == 0:
                controller.prepare_strategy_step(observation)
                decoded = decode_observation(observation)
                transitions = tuple(
                    Transition(position, limit)
                    for position, limit in sorted(
                        controller.reference._known_transitions.items()
                    )
                )
                points = []
                for label, action in (
                    ("low", -1.0),
                    ("medium", 0.0),
                    ("high", 1.0),
                ):
                    price = normalized_action_to_lambda(action)
                    plan = controller.planner.solve(
                        controller.state,
                        current_limit_m_s=float(decoded["current_limit_m_s"]),
                        transitions=transitions,
                        price_kwh_per_s=price,
                    )
                    points.append(
                        {
                            "label": label,
                            "normalized_action": action,
                            "price_kwh_per_s": price,
                            "selected_travel_time_s": plan.selected.travel_time_s,
                            "selected_horizon_energy_kwh": (
                                plan.selected.horizon_energy_kwh
                            ),
                            "selected_terminal_energy_kwh": (
                                plan.selected.terminal_energy_kwh
                            ),
                            "selected_target_speed_m_s": plan.selected.target_speed_m_s,
                            "first_native_action": plan.segment_actions[0],
                            "final_position_m": plan.selected.final_position_m,
                            "final_velocity_m_s": plan.selected.final_velocity_m_s,
                            "feasible": plan.selected.feasible,
                        }
                    )
                durations = [
                    float(point["selected_travel_time_s"]) for point in points
                ]
                distinct_duration_count = len(
                    {round(value, 9) for value in durations}
                )
                marginal = None
                if durations[0] > durations[-1] + 1.0e-9:
                    marginal = marginal_price(
                        slow_energy_kwh=float(
                            points[0]["selected_horizon_energy_kwh"]
                        )
                        + float(points[0]["selected_terminal_energy_kwh"]),
                        fast_energy_kwh=float(
                            points[-1]["selected_horizon_energy_kwh"]
                        )
                        + float(points[-1]["selected_terminal_energy_kwh"]),
                        slow_time_s=durations[0],
                        fast_time_s=durations[-1],
                    )
                candidate = {
                    "numeric_seed": int(numeric_seed),
                    "causal_state_position_m": controller.state.position_m,
                    "causal_state_elapsed_time_s": controller.state.elapsed_time_s,
                    "points": points,
                    "duration_nonincreasing_with_price": all(
                        left + 1.0e-9 >= right
                        for left, right in zip(durations, durations[1:])
                    ),
                    "distinct_first_native_actions": len(
                        {
                            round(float(point["first_native_action"]), 12)
                            for point in points
                        }
                    ),
                    "distinct_selected_durations": distinct_duration_count,
                    "observed_endpoint_marginal_price_kwh_per_s": marginal,
                }
                if best is None or distinct_duration_count > int(
                    best["distinct_selected_durations"]
                ):
                    best = candidate
                if distinct_duration_count >= 2:
                    return candidate
            native_action = controller.action(observation, held)
            observation, _, terminated, truncated, _ = environment.step(
                [native_action]
            )
            controller.observe_executed_action(native_action)
            step += 1
            ended = bool(terminated or truncated)
        assert best is not None
        return best
    finally:
        environment.close()


def _contextual(seed: int, arm: str) -> dict[str, Any]:
    outcome = evaluate_hybrid_seed(numeric_seed=seed, arm=arm)
    return {"numeric_seed": seed, **outcome}


def _paired_energy(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    by_seed_left = {int(row["numeric_seed"]): row for row in left}
    by_seed_right = {int(row["numeric_seed"]): row for row in right}
    values = []
    for seed in sorted(by_seed_left.keys() & by_seed_right.keys()):
        a = by_seed_left[seed]
        b = by_seed_right[seed]
        if a["feasible"] and b["feasible"]:
            values.append(
                {
                    "numeric_seed": seed,
                    "percent": 100.0
                    * (float(a["energy_kwh"]) - float(b["energy_kwh"]))
                    / float(a["energy_kwh"]),
                }
            )
    return {
        "comparison": f"{right_name} savings relative to {left_name}",
        "jointly_feasible_count": len(values),
        "task_level_percentages": values,
        "arithmetic_mean_percent": (
            statistics.fmean(row["percent"] for row in values) if values else None
        ),
    }


def run_development() -> dict[str, Any]:
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    rows = ledger["development_panel"]["accepted_tasks"]
    seeds = [int(row["numeric_seed"]) for row in rows]
    calibration_seeds = seeds[:4]
    capability_seeds = seeds[4:]
    fixed_calibration = calibrate_fixed_price(calibration_seeds)
    adaptive_calibration = calibrate_adaptive_price(calibration_seeds)
    fixed_action = float(fixed_calibration["selected_normalized_action"])
    adaptive_high = float(adaptive_calibration["selected_high_reserve_s"])
    adaptive_low = float(adaptive_calibration["selected_low_reserve_s"])
    fixed = [
        evaluate_price_seed(
            numeric_seed=seed, selector=FixedPrice(fixed_action)
        )
        for seed in capability_seeds
    ]
    adaptive = [
        evaluate_price_seed(
            numeric_seed=seed,
            selector=AdaptivePrice(
                high_reserve_s=adaptive_high, low_reserve_s=adaptive_low
            ),
        )
        for seed in capability_seeds
    ]
    causal = [_contextual(seed, "causal") for seed in capability_seeds]
    time_aware = [_contextual(seed, "time_aware") for seed in capability_seeds]
    oracle = []
    for seed in capability_seeds:
        result = solve_oracle(seed, refinement_iterations=6)
        oracle.append(
            {
                "numeric_seed": seed,
                "feasible": bool(result["canonical_success"]),
                "completed": bool(result["completion"]),
                "deadline_compliant": bool(result["deadline_compliant"]),
                "speed_compliant": bool(result["speed_compliant"]),
                "travel_time_s": float(result["travel_time_s"]),
                "energy_kwh": float(result["signed_net_energy_kwh"]),
                "solver_computation_time_s": float(
                    result["solver_computation_time_s"]
                ),
                "reference_is_global_optimum": False,
            }
        )
    sensitivity = [
        counterfactual_price_sensitivity(seed) for seed in capability_seeds
    ]
    nondegenerate = all(
        row["duration_nonincreasing_with_price"]
        and row["distinct_selected_durations"] >= 2
        for row in sensitivity
    )
    result = {
        "schema_version": 1,
        "panel_id": ledger["development_panel"]["id"],
        "panel_ordered_sha256": ledger["development_panel"]["ordered_sha256"],
        "partition": {
            "calibration_seeds": calibration_seeds,
            "independent_capability_seeds": capability_seeds,
        },
        "fixed_price_calibration": fixed_calibration,
        "adaptive_price_calibration": adaptive_calibration,
        "sensitivity": sensitivity,
        "time_price_action_non_degenerate": nondegenerate,
        "arms": {
            "fixed_price_mpc": {"summary": _summary(fixed), "episodes": fixed},
            "adaptive_price_mpc": {
                "summary": _summary(adaptive),
                "episodes": adaptive,
            },
            "causal60": {"summary": _summary(causal), "episodes": causal},
            "time_aware": {"summary": _summary(time_aware), "episodes": time_aware},
            "privileged_reference": {
                "summary": _summary(oracle),
                "episodes": oracle,
                "status": (
                    "contextual full-route reference; not information-matched "
                    "and not an optimality bound"
                ),
            },
        },
        "paired_feasible_energy": [
            _paired_energy(fixed, adaptive, "FixedPrice MPC", "AdaptivePrice MPC"),
            _paired_energy(causal, adaptive, "CAUSAL60", "AdaptivePrice MPC"),
            _paired_energy(time_aware, adaptive, "TIME_AWARE", "AdaptivePrice MPC"),
            _paired_energy(
                oracle, adaptive, "privileged reference", "AdaptivePrice MPC"
            ),
        ],
        "validation_performance_episodes": 0,
        "paper_episodes": 0,
    }
    return result


def evaluate_validation(*args, **kwargs) -> None:
    """Authorization is checked before any environment reset or task lookup."""

    require_validation_authorized()
    raise NotImplementedError(
        "authorized exactly-once Validation transaction not started"
    )
