"""Original-environment evaluation and independently replayed witnesses."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from statistics import fmean, median
from typing import Any, Literal

import gymnasium as gym

import gym_longicontrol  # noqa: F401 - registers Gymnasium environments
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import is_feasible

from .causal_controller import ObservationLimitedController
from .feasibility import (
    CANONICAL_TASK,
    FeasibilityClassification,
    classify_outcome,
    optimistic_route_time_s,
    unconstrained_minimum_time_lower_bound_s,
)
from .oracle_controller import FullRoutePhysicsOracle

ControllerKind = Literal["oracle", "causal"]


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_environment():
    return gym.make("StochasticTrack-v1", max_episode_steps=1_800)


def _episode_record(
    *,
    seed: int,
    kind: ControllerKind,
    info: dict[str, Any],
    actions: list[float],
    trace: list[tuple[float, ...]],
    solver_runtime_s: float,
    cruise_cap_m_s: float | None,
) -> dict[str, Any]:
    metrics = EpisodeMetrics(**info["episode_metrics"])
    feasible = is_feasible(metrics, CANONICAL_TASK)
    failures = []
    if not metrics.completed:
        failures.append("incomplete")
    if metrics.travel_time_s > CANONICAL_TASK.max_time_s:
        failures.append("deadline")
    if metrics.max_speed_violation_m_s > 0.0:
        failures.append("speed")
    return {
        "track_seed": int(seed),
        "controller": kind,
        "information_available": (
            FullRoutePhysicsOracle.information
            if kind == "oracle"
            else ObservationLimitedController.information
        ),
        "canonical_success": feasible,
        "failure_mode": "+".join(failures) if failures else "feasible",
        "completion": metrics.completed,
        "deadline_compliant": metrics.travel_time_s <= 140.0,
        "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
        "requirement_satisfaction": feasible,
        "travel_time_s": metrics.travel_time_s,
        "max_speed_violation_m_s": metrics.max_speed_violation_m_s,
        "integrated_speed_violation_m": metrics.integrated_speed_violation_m,
        "speed_violation_count": metrics.speed_violation_count,
        "signed_net_energy_kwh": metrics.energy_kwh,
        "final_position_m": float(info["position_m"]),
        "simulator_steps": len(actions),
        "solver_status": "FEASIBLE" if feasible else "NO_WITNESS",
        "solver_computation_time_s": solver_runtime_s,
        "cruise_cap_m_s": cruise_cap_m_s,
        "action_sha256": canonical_sha256(actions),
        "trace_sha256": canonical_sha256(trace),
        "actions": actions,
    }


def rollout(
    seed: int,
    kind: ControllerKind,
    *,
    cruise_cap_m_s: float = 37.0,
) -> dict[str, Any]:
    """Execute a deterministic controller in the unchanged Gymnasium env."""

    started = time.perf_counter()
    environment = make_environment()
    actions: list[float] = []
    trace: list[tuple[float, ...]] = []
    try:
        observation, info = environment.reset(seed=int(seed))
        if kind == "oracle":
            controller = FullRoutePhysicsOracle(
                environment.unwrapped.track, cruise_cap_m_s=cruise_cap_m_s
            )
        elif kind == "causal":
            controller = ObservationLimitedController()
            controller.reset()
        else:
            raise ValueError(f"unknown controller kind: {kind}")
        ended = False
        while not ended:
            if kind == "oracle":
                action = controller.action(environment.unwrapped.state)
            else:
                action = controller.action(observation)
            observation, _, terminated, truncated, info = environment.step([action])
            if kind == "causal":
                controller.observe_executed_action(action)
            actions.append(float(action))
            trace.append(
                (
                    float(info["position_m"]),
                    float(info["velocity_m_s"]),
                    float(info["acceleration_m_s2"]),
                    float(info["elapsed_time_s"]),
                    float(info["total_energy_kwh"]),
                    float(info["speed_limit_m_s"]),
                )
            )
            ended = bool(terminated or truncated)
    finally:
        environment.close()
    return _episode_record(
        seed=seed,
        kind=kind,
        info=info,
        actions=actions,
        trace=trace,
        solver_runtime_s=time.perf_counter() - started,
        cruise_cap_m_s=cruise_cap_m_s if kind == "oracle" else None,
    )


def replay_actions(seed: int, actions: list[float]) -> dict[str, Any]:
    """Replay a frozen action sequence independently in the original env."""

    environment = make_environment()
    trace: list[tuple[float, ...]] = []
    try:
        _, info = environment.reset(seed=int(seed))
        ended = False
        used = 0
        for action in actions:
            _, _, terminated, truncated, info = environment.step([float(action)])
            used += 1
            trace.append(
                (
                    float(info["position_m"]),
                    float(info["velocity_m_s"]),
                    float(info["acceleration_m_s2"]),
                    float(info["elapsed_time_s"]),
                    float(info["total_energy_kwh"]),
                    float(info["speed_limit_m_s"]),
                )
            )
            ended = bool(terminated or truncated)
            if ended:
                break
        if not ended or used != len(actions):
            raise RuntimeError(
                "frozen trajectory does not end exactly at action boundary"
            )
        metrics = asdict(EpisodeMetrics(**info["episode_metrics"]))
        return {
            "track_seed": int(seed),
            "actions_used": used,
            "action_sha256": canonical_sha256(actions),
            "trace_sha256": canonical_sha256(trace),
            "metrics": metrics,
            "final_position_m": float(info["position_m"]),
        }
    finally:
        environment.close()


def solve_oracle(seed: int, *, refinement_iterations: int = 10) -> dict[str, Any]:
    """Find a time-feasible witness, then refine energy without losing it."""

    started = time.perf_counter()
    initial = rollout(seed, "oracle", cruise_cap_m_s=37.0)
    candidates = [initial]
    if initial["canonical_success"]:
        lower, upper = 0.0, 37.0
        for _ in range(refinement_iterations):
            middle = (lower + upper) / 2.0
            candidate = rollout(seed, "oracle", cruise_cap_m_s=middle)
            candidates.append(candidate)
            if candidate["canonical_success"]:
                upper = middle
            else:
                lower = middle
    feasible = [row for row in candidates if row["canonical_success"]]
    selected = (
        min(feasible, key=lambda row: row["signed_net_energy_kwh"])
        if feasible
        else initial
    )
    replay = replay_actions(seed, selected["actions"])
    parity = bool(
        replay["trace_sha256"] == selected["trace_sha256"]
        and replay["action_sha256"] == selected["action_sha256"]
        and replay["metrics"]
        == {
            "completed": selected["completion"],
            "travel_time_s": selected["travel_time_s"],
            "energy_kwh": selected["signed_net_energy_kwh"],
            "speed_violation_count": selected["speed_violation_count"],
            "max_speed_violation_m_s": selected["max_speed_violation_m_s"],
            "integrated_speed_violation_m": selected[
                "integrated_speed_violation_m"
            ],
        }
        and replay["final_position_m"] == selected["final_position_m"]
    )
    if not parity:
        raise RuntimeError("exact original-simulator witness replay parity failed")
    classification, reason = classify_outcome(
        EpisodeMetrics(
            completed=selected["completion"],
            travel_time_s=selected["travel_time_s"],
            energy_kwh=selected["signed_net_energy_kwh"],
            speed_violation_count=selected["speed_violation_count"],
            max_speed_violation_m_s=selected["max_speed_violation_m_s"],
            integrated_speed_violation_m=selected["integrated_speed_violation_m"],
        ),
        unconstrained_time_lower_bound_s=unconstrained_minimum_time_lower_bound_s(),
    )
    result = {key: value for key, value in selected.items() if key != "actions"}
    result.update(
        {
            "classification": classification.value,
            "classification_reason": reason,
            "feasible_witness_found": (
                classification == FeasibilityClassification.FEASIBLE_WITNESS
            ),
            "infeasibility_proved": (
                classification == FeasibilityClassification.PROVEN_INFEASIBLE
            ),
            "unresolved": classification == FeasibilityClassification.UNRESOLVED,
            "optimization_iterations": len(candidates),
            "feasibility_preserving_energy_refinement": bool(
                selected is not initial
                and selected["canonical_success"]
                and selected["signed_net_energy_kwh"]
                <= initial["signed_net_energy_kwh"]
            ),
            "initial_time_efficient_energy_kwh": initial[
                "signed_net_energy_kwh"
            ],
            "initial_time_efficient_travel_time_s": initial["travel_time_s"],
            "numerical_parity": parity,
            "replay": replay,
            "solver_computation_time_s": time.perf_counter() - started,
            "actions": selected["actions"],
        }
    )
    environment = make_environment()
    try:
        environment.reset(seed=int(seed))
        result["optimistic_route_time_s"] = optimistic_route_time_s(
            environment.unwrapped.track
        )
        result["initial_deadline_slack_s"] = (
            CANONICAL_TASK.max_time_s - result["optimistic_route_time_s"]
        )
    finally:
        environment.close()
    return result


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(episodes)
    energies = [
        row["signed_net_energy_kwh"]
        for row in episodes
        if row["canonical_success"]
    ]
    return {
        "episode_count": count,
        "requirement_satisfaction_count": sum(
            row["canonical_success"] for row in episodes
        ),
        "requirement_satisfaction_rate": (
            sum(row["canonical_success"] for row in episodes) / count if count else 0.0
        ),
        "completion_count": sum(row["completion"] for row in episodes),
        "completion_rate": sum(row["completion"] for row in episodes) / count
        if count
        else 0.0,
        "deadline_compliance_count": sum(
            row["deadline_compliant"] for row in episodes
        ),
        "deadline_compliance_rate": sum(
            row["deadline_compliant"] for row in episodes
        )
        / count
        if count
        else 0.0,
        "speed_compliance_count": sum(row["speed_compliant"] for row in episodes),
        "speed_compliance_rate": sum(row["speed_compliant"] for row in episodes)
        / count
        if count
        else 0.0,
        "mean_travel_time_s": fmean(row["travel_time_s"] for row in episodes)
        if episodes
        else None,
        "feasible_only_energy_kwh": {
            "count": len(energies),
            "mean": fmean(energies) if energies else None,
            "median": median(energies) if energies else None,
            "values": energies,
        },
        "mean_solver_computation_time_s": fmean(
            row["solver_computation_time_s"] for row in episodes
        )
        if episodes
        else None,
        "maximum_solver_computation_time_s": max(
            (row["solver_computation_time_s"] for row in episodes), default=None
        ),
    }


def paired_differences(
    oracle: list[dict[str, Any]], causal: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    oracle_by_seed = {row["track_seed"]: row for row in oracle}
    causal_by_seed = {row["track_seed"]: row for row in causal}
    if set(oracle_by_seed) != set(causal_by_seed):
        raise ValueError("paired reference panels differ")
    return [
        {
            "track_seed": seed,
            "oracle_minus_causal_requirement_satisfaction": int(
                oracle_by_seed[seed]["canonical_success"]
            )
            - int(causal_by_seed[seed]["canonical_success"]),
            "oracle_minus_causal_completion": int(
                oracle_by_seed[seed]["completion"]
            )
            - int(causal_by_seed[seed]["completion"]),
            "oracle_minus_causal_deadline_compliance": int(
                oracle_by_seed[seed]["deadline_compliant"]
            )
            - int(causal_by_seed[seed]["deadline_compliant"]),
            "oracle_minus_causal_speed_compliance": int(
                oracle_by_seed[seed]["speed_compliant"]
            )
            - int(causal_by_seed[seed]["speed_compliant"]),
            "oracle_minus_causal_travel_time_s": (
                oracle_by_seed[seed]["travel_time_s"]
                - causal_by_seed[seed]["travel_time_s"]
            ),
        }
        for seed in sorted(oracle_by_seed)
    ]
