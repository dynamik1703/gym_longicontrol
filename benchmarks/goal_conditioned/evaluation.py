"""External physical evaluation for the goal-conditioned comparison."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from statistics import fmean, median
from typing import Any

import numpy as np

from benchmarks.scalar_sac.evaluation import EpisodeRecorder
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import is_feasible

from .experiment import make_environment


def _failure_mode(episode: dict[str, Any], configuration) -> str:
    failures = []
    if not episode["completed"]:
        failures.append("incomplete")
    if episode["travel_time_s"] > configuration.task.max_time_s:
        failures.append("deadline")
    if (
        episode["max_speed_violation_m_s"]
        > configuration.task.max_speed_violation_m_s
    ):
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def summarize_episodes(
    episodes: Iterable[dict[str, Any]], configuration
) -> dict[str, Any]:
    rows = tuple(episodes)
    if not rows:
        raise ValueError("At least one evaluation episode is required")
    count = len(rows)
    feasible = [row for row in rows if row["feasible"]]
    modes = sorted({row["failure_mode"] for row in rows})
    return {
        "episode_count": count,
        "success_count": len(feasible),
        "requirement_satisfaction_rate": len(feasible) / count,
        "completion_count": sum(row["completed"] for row in rows),
        "completion_rate": sum(row["completed"] for row in rows) / count,
        "completed_by_deadline_count": sum(
            row["completed"]
            and row["travel_time_s"] <= configuration.task.max_time_s
            for row in rows
        ),
        "completed_by_deadline_rate": sum(
            row["completed"]
            and row["travel_time_s"] <= configuration.task.max_time_s
            for row in rows
        )
        / count,
        "speed_compliant_count": sum(
            row["max_speed_violation_m_s"]
            <= configuration.task.max_speed_violation_m_s
            for row in rows
        ),
        "speed_compliance_rate": sum(
            row["max_speed_violation_m_s"]
            <= configuration.task.max_speed_violation_m_s
            for row in rows
        )
        / count,
        "failure_mode_counts": {
            mode: sum(row["failure_mode"] == mode for row in rows)
            for mode in modes
        },
        "mean_travel_time_s": fmean(row["travel_time_s"] for row in rows),
        "median_travel_time_s": float(
            median(row["travel_time_s"] for row in rows)
        ),
        "mean_final_position_m": fmean(
            row["final_position_m"] for row in rows
        ),
        "mean_max_speed_violation_m_s": fmean(
            row["max_speed_violation_m_s"] for row in rows
        ),
        "mean_integrated_speed_violation_m": fmean(
            row["integrated_speed_violation_m"] for row in rows
        ),
        "feasible_energy_count": len(feasible),
        "mean_feasible_energy_kwh": (
            fmean(row["energy_kwh"] for row in feasible) if feasible else None
        ),
        "median_feasible_energy_kwh": (
            float(median(row["energy_kwh"] for row in feasible))
            if feasible
            else None
        ),
        "evaluation_simulator_transitions": sum(
            row["step_count"] for row in rows
        ),
    }


def evaluate_model(model, configuration, evaluation_seeds) -> dict[str, Any]:
    """Evaluate deterministic actions using only external episode metrics."""

    seeds = tuple(int(seed) for seed in evaluation_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation_seeds must be non-empty and unique")
    environment = make_environment(configuration)
    episodes = []
    try:
        for seed in seeds:
            observation, _ = environment.reset(seed=seed)
            recorder = EpisodeRecorder()
            while True:
                action, _state = model.predict(observation, deterministic=True)
                action = np.asarray(action, dtype=np.float64).reshape((1,))
                observation, _reward, terminated, truncated, info = (
                    environment.step(action)
                )
                recorder.observe(action, info)
                if not (terminated or truncated):
                    continue
                metrics = EpisodeMetrics(**info["episode_metrics"])
                evaluation = recorder.finalize(seed, metrics, configuration.task)
                row = asdict(evaluation)
                authoritative = is_feasible(metrics, configuration.task)
                if authoritative != bool(info["goal_success"]):
                    raise RuntimeError(
                        "Goal wrapper and external evaluator disagree"
                    )
                if authoritative != evaluation.feasible:
                    raise RuntimeError("Episode recorder changed feasibility")
                row["goal_success"] = bool(info["goal_success"])
                row["failure_mode"] = _failure_mode(row, configuration)
                episodes.append(row)
                break
    finally:
        environment.close()
    return {
        "episodes": episodes,
        "summary": summarize_episodes(episodes, configuration),
    }
