"""Calibration and exactly-once Capability evaluation for Causal MPC V2."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

import gym_longicontrol  # noqa: F401 - registers StochasticTrack-v1
from benchmarks.hybrid_rl_execution.evaluation import (
    evaluate_seed as evaluate_hybrid_seed,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .adaptive_price import AdaptivePrice
from .controller import PriceMPCController
from .fixed_price import (
    LAMBDA_MAX_KWH_PER_S,
    LAMBDA_MIN_KWH_PER_S,
    FixedPrice,
    normalized_action_to_lambda,
)
from .integrity import (
    CANONICAL,
    HERE,
    method_hashes,
    require_not_paper,
    verify_frozen_method,
    verify_paper_seal,
)
from .planner import CandidateTrajectory
from .split import canonical_sha256

SPLIT_LEDGER = HERE / "split_ledger.json"
CALIBRATION_JSON = HERE / "calibration_results.json"
CAPABILITY_PROGRESS = HERE / "capability_progress.json"
CAPABILITY_JSON = HERE / "capability_results.json"
RESULTS_JSON = HERE / "results.json"
CANONICAL_TASK = TaskSpecification(140.0, 0.0)
DECISION_INTERVAL_NATIVE_STEPS = 10
FIXED_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
ADAPTIVE_GRID = ((12.0, 2.0), (18.0, 2.0), (24.0, 2.0))
CONTROLLER_ORDER = ("CAUSAL60", "TIME_AWARE", "FIXED_PRICE_V2", "ADAPTIVE_PRICE_V2")


class PriceSelector(Protocol):
    def reset(self) -> None: ...

    def action(self, strategy_observation: np.ndarray) -> float: ...


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _environment() -> gym.Env:
    return gym.make("StochasticTrack-v1", max_episode_steps=1_800)


def _select_for_price(
    candidates: tuple[CandidateTrajectory, ...], price: float
) -> CandidateTrajectory:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    return min(
        feasible,
        key=lambda candidate: (
            round(
                (
                    candidate.total_estimated_energy_kwh
                    + price * candidate.total_estimated_time_s
                )
                / 1.0e-12
            ),
            candidate.total_estimated_time_s,
            candidate.total_estimated_energy_kwh,
            candidate.profile_id,
        ),
    )


def evaluate_price_seed(
    *, numeric_seed: int, selector: PriceSelector
) -> dict[str, Any]:
    """Run one MPC task without exposing Track, seed or simulator state."""

    require_not_paper(numeric_seed)
    environment = _environment()
    controller = PriceMPCController()
    selector.reset()
    prices: list[float] = []
    normalized_prices: list[float] = []
    native_actions: list[float] = []
    solve_times: list[float] = []
    decisions: list[dict[str, Any]] = []
    selected_profiles: list[str] = []
    selected_families: list[str] = []
    terminal_energies: list[float] = []
    terminal_times: list[float] = []
    counterfactuals: list[dict[str, Any]] = []
    feasible_candidate_counts: list[int] = []
    tradeoff_candidate_counts: list[int] = []
    try:
        observation, info = environment.reset(seed=int(numeric_seed))
        controller.reset()
        held_price: float | None = None
        step = 0
        ended = False
        started_episode = time.perf_counter()
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
            decision = controller.last_decision
            if decision is None:
                raise RuntimeError("controller did not record its decision")
            if decision.replanned:
                solve_times.append(control_time)
                plan = controller.last_plan
                if plan is not None:
                    selected_profiles.append(plan.selected.profile_id)
                    selected_families.append(plan.selected.family)
                    terminal_energies.append(plan.selected.terminal_energy_kwh)
                    terminal_times.append(plan.selected.terminal_time_s)
                    feasible = tuple(
                        candidate
                        for candidate in plan.candidates
                        if candidate.feasible
                    )
                    feasible_candidate_counts.append(len(feasible))
                    distinct_tradeoffs = {
                        (
                            round(candidate.total_estimated_energy_kwh, 8),
                            round(candidate.total_estimated_time_s, 6),
                        )
                        for candidate in feasible
                    }
                    tradeoff_candidate_counts.append(len(distinct_tradeoffs))
                    low = _select_for_price(feasible, LAMBDA_MIN_KWH_PER_S)
                    high = _select_for_price(feasible, LAMBDA_MAX_KWH_PER_S)
                    counterfactuals.append(
                        {
                            "native_step": step,
                            "low_profile_id": low.profile_id,
                            "high_profile_id": high.profile_id,
                            "low_first_action": low.actions[0],
                            "high_first_action": high.actions[0],
                            "low_total_time_s": low.total_estimated_time_s,
                            "high_total_time_s": high.total_estimated_time_s,
                            "different_profile": low.profile_id != high.profile_id,
                            "different_first_action": abs(
                                low.actions[0] - high.actions[0]
                            )
                            > 1.0e-12,
                            "duration_nonincreasing": (
                                high.total_estimated_time_s
                                <= low.total_estimated_time_s + 1.0e-9
                            ),
                        }
                    )
            observation, _, terminated, truncated, info = environment.step(
                [native_action]
            )
            controller.observe_executed_action(native_action)
            decisions.append(decision.as_dict())
            native_actions.append(float(native_action))
            step += 1
            ended = bool(terminated or truncated)
        episode_runtime = time.perf_counter() - started_episode
        metrics = EpisodeMetrics(**info["episode_metrics"])
        feasible_episode = is_feasible(metrics, CANONICAL_TASK)
        falsifications = [
            {
                "native_step": index,
                "completion_bound_s": row["completion_bound_s"],
                "observed_completion_s": metrics.travel_time_s,
            }
            for index, row in enumerate(decisions)
            if row["accepted"]
            and row["deadline_viable"]
            and row["completion_bound_s"] is not None
            and metrics.travel_time_s > float(row["completion_bound_s"]) + 0.2
        ]
        counterfactual_count = len(counterfactuals)
        distinct_first = sum(
            bool(row["different_first_action"]) for row in counterfactuals
        )
        return {
            **asdict(metrics),
            "numeric_seed": int(numeric_seed),
            "feasible": feasible_episode,
            "deadline_compliant": metrics.travel_time_s <= 140.0,
            "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
            "time_reserve_s": 140.0 - metrics.travel_time_s,
            "feasible_energy_kwh": metrics.energy_kwh if feasible_episode else None,
            "normalized_price_actions": normalized_prices,
            "price_minimum_kwh_per_s": min(prices),
            "price_maximum_kwh_per_s": max(prices),
            "native_action_sha256": canonical_sha256(native_actions),
            "planner_solve_count": len(solve_times),
            "planner_solve_time_s": sum(solve_times),
            "mean_planner_solve_time_s": (
                statistics.fmean(solve_times) if solve_times else 0.0
            ),
            "maximum_planner_solve_time_s": max(solve_times, default=0.0),
            "episode_wall_time_s": episode_runtime,
            "physics_prediction_calls": sum(
                int(row["physics_prediction_calls"]) for row in decisions
            ),
            "proposal_count": len(decisions),
            "override_count": sum(bool(row["overridden"]) for row in decisions),
            "planner_proposal_identical_to_fallback_count": sum(
                row["intervention_reason"] == "ACCEPTED_IDENTICAL_TO_FALLBACK"
                for row in decisions
            ),
            "native_action_influence_count": sum(
                float(row["native_action_influence"]) > 1.0e-12
                for row in decisions
            ),
            "intervention_reasons": dict(
                sorted(
                    Counter(
                        row["intervention_reason"] for row in decisions
                    ).items()
                )
            ),
            "selected_profile_counts": dict(sorted(Counter(selected_profiles).items())),
            "selected_family_counts": dict(sorted(Counter(selected_families).items())),
            "selected_terminal_energy_sum_kwh": sum(terminal_energies),
            "selected_terminal_energy_mean_kwh": (
                statistics.fmean(terminal_energies) if terminal_energies else 0.0
            ),
            "selected_terminal_time_mean_s": (
                statistics.fmean(terminal_times) if terminal_times else 0.0
            ),
            "counterfactual_count": counterfactual_count,
            "counterfactual_different_profile_count": sum(
                bool(row["different_profile"]) for row in counterfactuals
            ),
            "counterfactual_different_first_action_count": distinct_first,
            "counterfactual_different_first_action_fraction": (
                distinct_first / counterfactual_count
                if counterfactual_count
                else 0.0
            ),
            "counterfactual_duration_monotonic": all(
                bool(row["duration_nonincreasing"]) for row in counterfactuals
            ),
            "mean_feasible_candidate_count": (
                statistics.fmean(feasible_candidate_counts)
                if feasible_candidate_counts
                else 0.0
            ),
            "mean_distinct_energy_time_candidate_count": (
                statistics.fmean(tradeoff_candidate_counts)
                if tradeoff_candidate_counts
                else 0.0
            ),
            "deadline_bound_falsifications": falsifications,
            "nonfinite_value_count": 0,
        }
    finally:
        environment.close()


def _contextual(seed: int, arm: str) -> dict[str, Any]:
    result = evaluate_hybrid_seed(numeric_seed=seed, arm=arm)
    return {
        "numeric_seed": seed,
        **result,
        "time_reserve_s": 140.0 - float(result["travel_time_s"]),
        "episode_wall_time_s": 0.0,
        "planner_solve_time_s": 0.0,
        "planner_solve_count": 0,
        "counterfactual_count": 0,
        "counterfactual_different_first_action_count": 0,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["feasible"]]
    energies = [float(row["energy_kwh"]) for row in feasible]
    proposals = sum(int(row.get("proposal_count", 0)) for row in rows)
    influenced = sum(int(row.get("native_action_influence_count", 0)) for row in rows)
    counterfactuals = sum(int(row.get("counterfactual_count", 0)) for row in rows)
    counterfactual_first = sum(
        int(row.get("counterfactual_different_first_action_count", 0))
        for row in rows
    )
    return {
        "episode_count": len(rows),
        "rsr": len(feasible),
        "completion_count": sum(bool(row["completed"]) for row in rows),
        "deadline_compliance_count": sum(
            bool(row["deadline_compliant"]) for row in rows
        ),
        "speed_compliance_count": sum(
            bool(row["speed_compliant"]) for row in rows
        ),
        "mean_feasible_energy_kwh": statistics.fmean(energies) if energies else None,
        "median_feasible_energy_kwh": statistics.median(energies) if energies else None,
        "mean_travel_time_s": statistics.fmean(
            float(row["travel_time_s"]) for row in rows
        ),
        "mean_time_reserve_s": statistics.fmean(
            float(row["time_reserve_s"]) for row in rows
        ),
        "proposal_count": proposals,
        "override_count": sum(int(row.get("override_count", 0)) for row in rows),
        "native_action_influence_count": influenced,
        "native_action_influence_fraction": (
            influenced / proposals if proposals else 0.0
        ),
        "planner_solve_count": sum(
            int(row.get("planner_solve_count", 0)) for row in rows
        ),
        "planner_solve_time_s": sum(
            float(row.get("planner_solve_time_s", 0.0)) for row in rows
        ),
        "episode_wall_time_s": sum(
            float(row.get("episode_wall_time_s", 0.0)) for row in rows
        ),
        "counterfactual_count": counterfactuals,
        "counterfactual_different_first_action_count": counterfactual_first,
        "counterfactual_different_first_action_fraction": (
            counterfactual_first / counterfactuals if counterfactuals else 0.0
        ),
        "deadline_bound_falsification_count": sum(
            len(row.get("deadline_bound_falsifications", [])) for row in rows
        ),
        "nonfinite_value_count": sum(
            int(row.get("nonfinite_value_count", 0)) for row in rows
        ),
    }


def _score(rows: list[dict[str, Any]]) -> tuple[int, float, float]:
    summary = _summary(rows)
    energy = summary["mean_feasible_energy_kwh"]
    return (
        -int(summary["rsr"]),
        float("inf") if energy is None else float(energy),
        float(summary["planner_solve_time_s"]),
    )


def _paired_energy(
    baseline: list[dict[str, Any]],
    treatment: list[dict[str, Any]],
    baseline_name: str,
    treatment_name: str,
) -> dict[str, Any]:
    left = {int(row["numeric_seed"]): row for row in baseline}
    right = {int(row["numeric_seed"]): row for row in treatment}
    values = []
    for seed in sorted(left.keys() & right.keys()):
        before = left[seed]
        after = right[seed]
        if before["feasible"] and after["feasible"]:
            values.append(
                {
                    "numeric_seed": seed,
                    "baseline_energy_kwh": before["energy_kwh"],
                    "treatment_energy_kwh": after["energy_kwh"],
                    "saving_percent": 100.0
                    * (float(before["energy_kwh"]) - float(after["energy_kwh"]))
                    / float(before["energy_kwh"]),
                }
            )
    return {
        "comparison": f"{treatment_name} relative to {baseline_name}",
        "jointly_feasible_count": len(values),
        "task_level": values,
        "arithmetic_mean_saving_percent": (
            statistics.fmean(row["saving_percent"] for row in values)
            if values
            else None
        ),
    }


def run_calibration() -> dict[str, Any]:
    verify_paper_seal()
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if canonical["execution"]["calibration_complete"]:
        raise PermissionError("MPC_V2_CALIBRATION is already complete")
    if canonical["execution"]["capability_opened"]:
        raise PermissionError("cannot calibrate after Capability opened")
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    seeds = [
        int(row["numeric_seed"])
        for row in ledger["calibration_panel"]["accepted_tasks"]
    ]
    fixed_candidates = []
    for action in FIXED_GRID:
        episodes = [
            evaluate_price_seed(numeric_seed=seed, selector=FixedPrice(action))
            for seed in seeds
        ]
        fixed_candidates.append(
            {
                "normalized_action": action,
                "price_kwh_per_s": normalized_action_to_lambda(action),
                "score": _score(episodes),
                "summary": _summary(episodes),
                "episodes": episodes,
            }
        )
    selected_fixed = min(
        fixed_candidates,
        key=lambda row: (tuple(row["score"]), row["normalized_action"]),
    )
    adaptive_candidates = []
    for high, low in ADAPTIVE_GRID:
        episodes = [
            evaluate_price_seed(
                numeric_seed=seed,
                selector=AdaptivePrice(
                    high_reserve_s=high, low_reserve_s=low
                ),
            )
            for seed in seeds
        ]
        adaptive_candidates.append(
            {
                "high_reserve_s": high,
                "low_reserve_s": low,
                "score": _score(episodes),
                "summary": _summary(episodes),
                "episodes": episodes,
            }
        )
    selected_adaptive = min(
        adaptive_candidates,
        key=lambda row: (
            tuple(row["score"]),
            row["high_reserve_s"],
            row["low_reserve_s"],
        ),
    )
    result = {
        "schema_version": 1,
        "panel_id": ledger["calibration_panel"]["id"],
        "panel_ordered_sha256": ledger["calibration_panel"]["ordered_sha256"],
        "seeds": seeds,
        "budget": {
            "fixed_configurations": len(FIXED_GRID),
            "adaptive_configurations": len(ADAPTIVE_GRID),
            "episodes": len(seeds) * (len(FIXED_GRID) + len(ADAPTIVE_GRID)),
        },
        "objective_hierarchy": (
            "maximum RSR; minimum feasible energy; minimum planner time"
        ),
        "fixed_candidates": fixed_candidates,
        "adaptive_candidates": adaptive_candidates,
        "selected": {
            "fixed_normalized_action": selected_fixed["normalized_action"],
            "fixed_price_kwh_per_s": selected_fixed["price_kwh_per_s"],
            "adaptive_high_reserve_s": selected_adaptive["high_reserve_s"],
            "adaptive_low_reserve_s": selected_adaptive["low_reserve_s"],
        },
        "influence_gate_calibration": {
            "adaptive_executed_influence_fraction": selected_adaptive["summary"][
                "native_action_influence_fraction"
            ],
            "adaptive_counterfactual_distinct_first_action_fraction": selected_adaptive[
                "summary"
            ]["counterfactual_different_first_action_fraction"],
        },
        "paper_episodes": 0,
        "validation_episodes": 0,
        "rl_training_episodes": 0,
        "rl_gradient_updates": 0,
    }
    _write_json(CALIBRATION_JSON, result)
    canonical["execution"]["calibration_complete"] = True
    canonical["calibration_selection"] = result["selected"]
    canonical["frozen_method_sha256"] = method_hashes()
    _write_json(CANONICAL, canonical)
    return result


def _new_progress(canonical: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    canonical["execution"]["capability_opened"] = True
    _write_json(CANONICAL, canonical)
    return {
        "schema_version": 1,
        "state": "OPENED_INCOMPLETE",
        "panel_id": ledger["capability_panel"]["id"],
        "panel_ordered_sha256": ledger["capability_panel"]["ordered_sha256"],
        "controller_order": list(CONTROLLER_ORDER),
        "method_sha256": method_hashes(),
        "episodes": {},
    }


def _capability_episode(
    controller_id: str,
    seed: int,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    if controller_id == "CAUSAL60":
        return _contextual(seed, "causal")
    if controller_id == "TIME_AWARE":
        return _contextual(seed, "time_aware")
    if controller_id == "FIXED_PRICE_V2":
        return evaluate_price_seed(
            numeric_seed=seed,
            selector=FixedPrice(
                float(calibration["selected"]["fixed_normalized_action"])
            ),
        )
    if controller_id == "ADAPTIVE_PRICE_V2":
        return evaluate_price_seed(
            numeric_seed=seed,
            selector=AdaptivePrice(
                high_reserve_s=float(
                    calibration["selected"]["adaptive_high_reserve_s"]
                ),
                low_reserve_s=float(
                    calibration["selected"]["adaptive_low_reserve_s"]
                ),
            ),
        )
    raise ValueError(f"unknown capability controller: {controller_id}")


def _compile_capability(progress: dict[str, Any]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for controller in CONTROLLER_ORDER:
        rows = [
            value
            for key, value in progress["episodes"].items()
            if key.startswith(f"{controller}:")
        ]
        rows.sort(key=lambda row: int(row["numeric_seed"]))
        arms[controller] = {"summary": _summary(rows), "episodes": rows}
    adaptive = arms["ADAPTIVE_PRICE_V2"]["episodes"]
    comparisons = {
        "adaptive_vs_time_aware": _paired_energy(
            arms["TIME_AWARE"]["episodes"],
            adaptive,
            "TIME_AWARE",
            "AdaptivePrice MPC V2",
        ),
        "adaptive_vs_fixed": _paired_energy(
            arms["FIXED_PRICE_V2"]["episodes"],
            adaptive,
            "FixedPrice MPC V2",
            "AdaptivePrice MPC V2",
        ),
        "fixed_vs_time_aware": _paired_energy(
            arms["TIME_AWARE"]["episodes"],
            arms["FIXED_PRICE_V2"]["episodes"],
            "TIME_AWARE",
            "FixedPrice MPC V2",
        ),
    }
    return {
        "schema_version": 1,
        "panel_id": progress["panel_id"],
        "panel_ordered_sha256": progress["panel_ordered_sha256"],
        "controller_order": list(CONTROLLER_ORDER),
        "arms": arms,
        "paired_feasible_energy": comparisons,
        "privileged_reference": {
            "executed": False,
            "gap_available": False,
            "reason": "preregistered optional contextual reference omitted",
        },
        "paper_episodes": 0,
        "validation_episodes": 0,
        "rl_training_episodes": 0,
        "rl_gradient_updates": 0,
    }


def run_capability() -> dict[str, Any]:
    verify_paper_seal()
    verify_frozen_method()
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if not canonical["execution"]["calibration_complete"]:
        raise PermissionError("Calibration must complete before Capability")
    if canonical["execution"]["capability_complete"]:
        raise PermissionError("MPC_V2_CAPABILITY is already complete")
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    if CAPABILITY_PROGRESS.exists():
        progress = json.loads(CAPABILITY_PROGRESS.read_text(encoding="utf-8"))
        if progress["method_sha256"] != method_hashes():
            raise RuntimeError("method changed during capability transaction")
    else:
        progress = _new_progress(canonical, ledger)
        _write_json(CAPABILITY_PROGRESS, progress)
    seeds = [
        int(row["numeric_seed"])
        for row in ledger["capability_panel"]["accepted_tasks"]
    ]
    for seed in seeds:
        for controller_id in CONTROLLER_ORDER:
            key = f"{controller_id}:{seed}"
            if key in progress["episodes"]:
                continue
            progress["episodes"][key] = _capability_episode(
                controller_id, seed, calibration
            )
            _write_json(CAPABILITY_PROGRESS, progress)
    expected = len(seeds) * len(CONTROLLER_ORDER)
    if len(progress["episodes"]) != expected:
        raise RuntimeError("capability transaction ended with missing pairs")
    capability = _compile_capability(progress)
    _write_json(CAPABILITY_JSON, capability)
    progress["state"] = "COMPLETE"
    _write_json(CAPABILITY_PROGRESS, progress)
    ledger["capability_panel"]["state"] = "COMPLETE"
    ledger["capability_panel"]["performance_episode_count"] = expected
    _write_json(SPLIT_LEDGER, ledger)
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    canonical["execution"]["capability_complete"] = True
    _write_json(CANONICAL, canonical)
    return capability


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("calibration", "capability"))
    args = parser.parse_args()
    result = run_calibration() if args.stage == "calibration" else run_capability()
    print(
        json.dumps(
            {
                "stage": args.stage,
                "panel_id": result["panel_id"],
                "episode_budget": (
                    result.get("budget", {}).get("episodes")
                    if args.stage == "calibration"
                    else sum(
                        arm["summary"]["episode_count"]
                        for arm in result["arms"].values()
                    )
                ),
                "complete": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
