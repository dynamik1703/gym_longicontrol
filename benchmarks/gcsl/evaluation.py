"""Deterministic canonical evaluation isolated from every training RNG."""

from __future__ import annotations

from dataclasses import asdict
from statistics import fmean
from typing import Any

import gymnasium as gym
import numpy as np

import gym_longicontrol  # noqa: F401
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .config import DEVELOPMENT_TRACKS, SEALED_PAPER_TRACKS, VALIDATION_TRACKS
from .goal_adapter import (
    OutcomeScales,
    augment_policy_state,
    canonical_command,
    physical_outcome,
)

CANONICAL_TASK = TaskSpecification(140.0, 0.0)


def make_environment():
    return gym.make("StochasticTrack-v1")


def raw_outcome(info: dict[str, Any], previous_position_m: float) -> np.ndarray:
    return physical_outcome(
        position_m=float(info["position_m"]),
        previous_position_m=float(previous_position_m),
        elapsed_time_s=float(info["elapsed_time_s"]),
        max_speed_violation_m_s=float(info["max_speed_violation_m_s"]),
    )


def failure_mode(metrics: EpisodeMetrics) -> str:
    failures = []
    if not metrics.completed:
        failures.append("incomplete")
    if metrics.travel_time_s > CANONICAL_TASK.max_time_s:
        failures.append("deadline")
    if metrics.max_speed_violation_m_s > CANONICAL_TASK.max_speed_violation_m_s:
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def _assert_split(track_seeds: tuple[int, ...], split: str) -> None:
    expected = {
        "development": DEVELOPMENT_TRACKS,
        "validation": VALIDATION_TRACKS,
    }
    opens_paper_tracks = any(seed in SEALED_PAPER_TRACKS for seed in track_seeds)
    if split == "paper-final" or opens_paper_tracks:
        raise PermissionError("paper-final tracks 4000-4017 remain sealed")
    if split not in expected or tuple(track_seeds) != expected[split]:
        raise PermissionError(f"track seeds do not match the declared {split} split")


def evaluate_policy(
    learner,
    track_seeds,
    *,
    split: str,
    environment_factory=make_environment,
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in track_seeds)
    _assert_split(seeds, split)
    episodes = []
    total_transitions = 0
    for seed in seeds:
        environment = environment_factory()
        try:
            observation, info = environment.reset(seed=seed)
            outcome = raw_outcome(info, float(info["position_m"]))
            ended = False
            while not ended:
                state = augment_policy_state(observation, outcome, OutcomeScales())
                action = learner.deterministic_action(
                    state[None, :], canonical_command()[None, :]
                )[0]
                previous_position = float(outcome[0])
                observation, _reward, terminated, truncated, info = environment.step(
                    action
                )
                outcome = raw_outcome(info, previous_position)
                total_transitions += 1
                ended = bool(terminated or truncated)
            metrics = EpisodeMetrics(**info["episode_metrics"])
            feasible = is_feasible(metrics, CANONICAL_TASK)
            episodes.append(
                {
                    "track_seed": seed,
                    "canonical_success": feasible,
                    "failure_mode": failure_mode(metrics),
                    "deadline_compliant": metrics.travel_time_s <= 140.0,
                    "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
                    "final_progress": min(float(info["position_m"]) / 1000.0, 1.0),
                    "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
                    **asdict(metrics),
                }
            )
        finally:
            environment.close()
    feasible_energy = [
        row["feasible_energy_kwh"]
        for row in episodes
        if row["feasible_energy_kwh"] is not None
    ]
    modes = sorted({row["failure_mode"] for row in episodes})
    return {
        "split": split,
        "episodes": episodes,
        "summary": {
            "canonical_successes": sum(row["canonical_success"] for row in episodes),
            "episode_count": len(episodes),
            "completion_count": sum(row["completed"] for row in episodes),
            "deadline_compliance_count": sum(
                row["deadline_compliant"] for row in episodes
            ),
            "speed_compliance_count": sum(
                row["speed_compliant"] for row in episodes
            ),
            "failure_mode_counts": {
                mode: sum(row["failure_mode"] == mode for row in episodes)
                for mode in modes
            },
            "mean_final_progress": fmean(row["final_progress"] for row in episodes),
            "mean_feasible_energy_kwh": (
                fmean(feasible_energy) if feasible_energy else None
            ),
            "evaluation_simulator_transitions": total_transitions,
        },
    }
