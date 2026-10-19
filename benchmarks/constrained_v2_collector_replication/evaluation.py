"""Actor-only deterministic evaluation on the two frozen collector sets."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict
from typing import Any

import numpy as np

from benchmarks.constrained_rl.adapter import deterministic_policy_adapter
from benchmarks.constrained_v2_repro_audit.harness import _module_sha256
from benchmarks.scalar_sac.evaluation import evaluate_policy
from benchmarks.scalar_sac.experiment import _base_environment

from .config import StudyConfiguration
from .split import diagnostic_seeds, validation_seeds


def _rng_state() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
    }


def _rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    import torch

    numpy_left = left["numpy"]
    numpy_right = right["numpy"]
    return (
        left["python"] == right["python"]
        and numpy_left[0] == numpy_right[0]
        and np.array_equal(numpy_left[1], numpy_right[1])
        and numpy_left[2:] == numpy_right[2:]
        and torch.equal(left["torch"], right["torch"])
    )


def _counts(episodes: list[dict[str, Any]], deadline_s: float) -> dict[str, int]:
    return {
        "requirement_satisfaction": sum(bool(row["feasible"]) for row in episodes),
        "completion": sum(bool(row["completed"]) for row in episodes),
        "deadline_compliance": sum(
            bool(row["completed"] and row["travel_time_s"] <= deadline_s)
            for row in episodes
        ),
        "speed_compliance": sum(
            float(row["max_speed_violation_m_s"]) <= 0.0 for row in episodes
        ),
    }


def _failure_mode(row: dict[str, Any], deadline_s: float) -> str:
    speed = float(row["max_speed_violation_m_s"]) > 0.0
    if row["feasible"]:
        return "FEASIBLE"
    if not row["completed"]:
        return "INCOMPLETE_WITH_SPEED_VIOLATION" if speed else "INCOMPLETE_SAFE"
    deadline = float(row["travel_time_s"]) > deadline_s
    if deadline and speed:
        return "COMPLETED_DEADLINE_AND_SPEED_VIOLATION"
    if deadline:
        return "COMPLETED_DEADLINE_VIOLATION"
    if speed:
        return "COMPLETED_SPEED_VIOLATION"
    return "OTHER_INFEASIBLE"


def _behavior_states(
    episodes: list[dict[str, Any]], deadline_s: float
) -> dict[str, int]:
    """Frozen descriptive states; speed violation is intentionally non-exclusive."""

    return {
        "standstill": sum(
            not row["completed"] and float(row["energy_kwh"]) <= 1e-6
            for row in episodes
        ),
        "crawling": sum(
            not row["completed"] and float(row["energy_kwh"]) > 1e-6 for row in episodes
        ),
        "timely_completion": sum(
            row["completed"] and float(row["travel_time_s"]) <= deadline_s
            for row in episodes
        ),
        "speed_violation": sum(
            float(row["max_speed_violation_m_s"]) > 0.0 for row in episodes
        ),
    }


def evaluate_diagnostic(
    policy: Any, configuration: StudyConfiguration, *, real_transitions: int
) -> dict[str, Any]:
    return evaluate_actor_on_seeds(
        policy,
        configuration,
        evaluation_seeds=diagnostic_seeds(),
        real_transitions=real_transitions,
        evaluation_id="COLLECTOR_DIAGNOSTIC_V1",
    )


def evaluate_validation(
    policy: Any, configuration: StudyConfiguration
) -> dict[str, Any]:
    return evaluate_actor_on_seeds(
        policy,
        configuration,
        evaluation_seeds=validation_seeds(),
        real_transitions=configuration.raw["real_transition_budget"],
        evaluation_id="COLLECTOR_VALIDATION_V1",
    )


def evaluate_actor_on_seeds(
    policy: Any,
    configuration: StudyConfiguration,
    *,
    evaluation_seeds: tuple[int, ...],
    real_transitions: int,
    evaluation_id: str,
) -> dict[str, Any]:
    """Evaluate deterministically while requiring zero learner/RNG interference."""

    before_rng = _rng_state()
    actor_before = _module_sha256(policy.actor)
    environment = _base_environment(configuration.v2)
    policy.eval()
    try:
        episodes_raw, summary_raw = evaluate_policy(
            deterministic_policy_adapter(policy),
            environment,
            task=configuration.v2.task,
            evaluation_seeds=evaluation_seeds,
        )
    finally:
        environment.close()
        policy.train()
    after_rng = _rng_state()
    actor_after = _module_sha256(policy.actor)
    if actor_after != actor_before:
        raise RuntimeError("Diagnostic evaluation changed actor parameters")
    if not _rng_equal(before_rng, after_rng):
        raise RuntimeError("Diagnostic evaluation changed training RNG state")
    episodes = [asdict(row) for row in episodes_raw]
    deadline = float(configuration.v2.task.max_time_s)
    modes: dict[str, int] = {}
    for row in episodes:
        mode = _failure_mode(row, deadline)
        modes[mode] = modes.get(mode, 0) + 1
    feasible_energy = [float(row["energy_kwh"]) for row in episodes if row["feasible"]]
    return {
        "real_transitions": int(real_transitions),
        "evaluation_id": evaluation_id,
        "evaluation_seeds": list(evaluation_seeds),
        "episode_count": len(episodes),
        "actor_only": True,
        "deterministic": True,
        "actor_sha256_before": actor_before,
        "actor_sha256_after": actor_after,
        "training_rng_unchanged": True,
        "episodes": episodes,
        "summary": asdict(summary_raw),
        "counts": _counts(episodes, deadline),
        "failure_modes": modes,
        "behavior_state_definition": {
            "standstill": "incomplete and energy_kwh <= 1e-6",
            "crawling": "incomplete and energy_kwh > 1e-6",
            "timely_completion": "completed and travel_time_s <= 140",
            "speed_violation": "max_speed_violation_m_s > 0 (non-exclusive)",
        },
        "behavior_states": _behavior_states(episodes, deadline),
        "feasible_only_energy_kwh": {
            "count": len(feasible_energy),
            "mean": float(np.mean(feasible_energy)) if feasible_energy else None,
            "median": float(np.median(feasible_energy)) if feasible_energy else None,
            "values": feasible_energy,
        },
    }


def evaluate_legacy_or_paper(*_args, **_kwargs):
    raise PermissionError("Collector replication forbids legacy and Paper evaluation")
