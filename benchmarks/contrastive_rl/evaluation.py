"""Separate deterministic physical evaluation for projected-goal CRL."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from statistics import fmean
from typing import Any

import gymnasium as gym
import jax.numpy as jnp
import numpy as np

import gym_longicontrol  # noqa: F401
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .config import DEVELOPMENT_TRACKS, VALIDATION_TRACKS
from .goal_adapter import (
    OutcomeScales,
    augment_policy_state,
    canonical_command,
    physical_outcome,
)

CANONICAL_TASK = TaskSpecification(140.0, 0.0)


def make_environment():
    return gym.make("StochasticTrack-v1")


def _failure_mode(metrics: EpisodeMetrics) -> str:
    failures = []
    if not metrics.completed:
        failures.append("incomplete")
    if metrics.travel_time_s > CANONICAL_TASK.max_time_s:
        failures.append("deadline")
    if metrics.max_speed_violation_m_s > CANONICAL_TASK.max_speed_violation_m_s:
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def _raw_outcome(info: dict[str, Any], previous_position_m: float) -> np.ndarray:
    return physical_outcome(
        position_m=float(info["position_m"]),
        previous_position_m=float(previous_position_m),
        elapsed_time_s=float(info["elapsed_time_s"]),
        max_speed_violation_m_s=float(info["max_speed_violation_m_s"]),
    )


def summarize_episodes(episodes: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = tuple(episodes)
    if not rows:
        raise ValueError("at least one evaluation episode is required")
    feasible = [row for row in rows if row["feasible"]]
    modes = sorted({row["failure_mode"] for row in rows})
    count = len(rows)
    return {
        "episode_count": count,
        "success_count": len(feasible),
        "requirement_satisfaction_rate": len(feasible) / count,
        "completion_count": sum(row["completed"] for row in rows),
        "completed_by_deadline_count": sum(
            row["completed"] and row["travel_time_s"] <= 140.0 for row in rows
        ),
        "speed_compliant_count": sum(
            row["max_speed_violation_m_s"] <= 0.0 for row in rows
        ),
        "failure_mode_counts": {
            mode: sum(row["failure_mode"] == mode for row in rows)
            for mode in modes
        },
        "mean_final_position_m": fmean(row["final_position_m"] for row in rows),
        "feasible_energy_count": len(feasible),
        "mean_feasible_energy_kwh": (
            fmean(row["energy_kwh"] for row in feasible) if feasible else None
        ),
        "evaluation_simulator_transitions": sum(
            row["step_count"] for row in rows
        ),
    }


def evaluate_actor(
    learner,
    actor_params,
    evaluation_seeds,
    *,
    split: str = "development",
) -> dict[str, Any]:
    """Evaluate without accepting or mutating any training RNG/state object."""

    seeds = tuple(map(int, evaluation_seeds))
    if split not in {"development", "validation"}:
        raise ValueError("split must be development or validation")
    expected = DEVELOPMENT_TRACKS if split == "development" else VALIDATION_TRACKS
    if seeds != expected:
        raise ValueError(f"{split} evaluation must use its complete frozen split")
    scales = OutcomeScales()
    command = jnp.asarray(canonical_command()[None, :], dtype=jnp.float32)
    episodes = []
    environment = make_environment()
    try:
        for seed in seeds:
            observation, info = environment.reset(seed=seed)
            outcome = _raw_outcome(info, float(info["position_m"]))
            step_count = 0
            while True:
                state = augment_policy_state(observation, outcome, scales)
                action = learner.deterministic_action(
                    actor_params,
                    jnp.asarray(state[None, :], dtype=jnp.float32),
                    command,
                )
                previous_position = float(info["position_m"])
                observation, _reward, terminated, truncated, info = environment.step(
                    np.asarray(action[0], dtype=np.float64)
                )
                step_count += 1
                outcome = _raw_outcome(info, previous_position)
                if not (terminated or truncated):
                    continue
                metrics = EpisodeMetrics(**info["episode_metrics"])
                feasible = is_feasible(metrics, CANONICAL_TASK)
                row = {
                    "evaluation_seed": seed,
                    "step_count": step_count,
                    "final_position_m": float(info["position_m"]),
                    "feasible": feasible,
                    "failure_mode": _failure_mode(metrics),
                    **asdict(metrics),
                }
                episodes.append(row)
                break
    finally:
        environment.close()
    return {
        "split": split,
        "command": canonical_command().tolist(),
        "deterministic_actions": True,
        "separate_environment": True,
        "episodes": episodes,
        "summary": summarize_episodes(episodes),
    }
