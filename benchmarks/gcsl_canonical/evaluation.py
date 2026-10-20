"""Deterministic canonical evaluation isolated from all training RNG state."""

from __future__ import annotations

import hashlib
import io
import random
from dataclasses import asdict
from statistics import fmean
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import gym_longicontrol  # noqa: F401
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .config import forbid_paper, require_validation_authorized
from .goal_adapter import (
    OutcomeScales,
    augment_policy_state,
    canonical_command,
    physical_outcome,
)
from .split import diagnostic_seeds, validation_seeds

CANONICAL_TASK = TaskSpecification(140.0, 0.0)


def make_environment():
    return gym.make("StochasticTrack-v1", max_episode_steps=1800)


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


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def _rng_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["python"] == right["python"]
        and left["numpy"][0] == right["numpy"][0]
        and np.array_equal(left["numpy"][1], right["numpy"][1])
        and left["numpy"][2:] == right["numpy"][2:]
        and torch.equal(left["torch"], right["torch"])
    )


def _policy_sha256(policy: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(policy.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def evaluate_policy(
    learner,
    seeds: tuple[int, ...],
    *,
    evaluation_id: str,
    environment_factory=make_environment,
) -> dict[str, Any]:
    expected = {
        "GCSL_DIAGNOSTIC_V1": diagnostic_seeds(),
        "GCSL_VALIDATION_V1": validation_seeds(),
    }
    if evaluation_id not in expected or tuple(seeds) != expected[evaluation_id]:
        raise PermissionError("evaluation seeds do not match the frozen split")
    if any(seed in range(4000, 4018) for seed in seeds):
        forbid_paper()
    rng_before = _rng_state()
    actor_before = _policy_sha256(learner.policy)
    was_training = learner.policy.training
    learner.policy.eval()
    episodes: list[dict[str, Any]] = []
    transitions = 0
    try:
        for seed in seeds:
            environment = environment_factory()
            try:
                observation, info = environment.reset(seed=int(seed))
                outcome = raw_outcome(info, float(info["position_m"]))
                ended = False
                while not ended:
                    state = augment_policy_state(observation, outcome, OutcomeScales())
                    action = learner.deterministic_action(
                        state[None, :], canonical_command()[None, :]
                    )[0]
                    previous = float(outcome[0])
                    observation, _reward, terminated, truncated, info = (
                        environment.step(action)
                    )
                    outcome = raw_outcome(info, previous)
                    transitions += 1
                    ended = bool(terminated or truncated)
                metrics = EpisodeMetrics(**info["episode_metrics"])
                feasible = is_feasible(metrics, CANONICAL_TASK)
                episodes.append(
                    {
                        "track_seed": int(seed),
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
    finally:
        learner.policy.train(was_training)
    actor_after = _policy_sha256(learner.policy)
    if actor_after != actor_before or not _rng_equal(rng_before, _rng_state()):
        raise RuntimeError("evaluation interfered with learner parameters or RNG")
    energies = [
        row["feasible_energy_kwh"]
        for row in episodes
        if row["feasible_energy_kwh"] is not None
    ]
    modes = sorted({row["failure_mode"] for row in episodes})
    return {
        "evaluation_id": evaluation_id,
        "evaluation_seeds": list(seeds),
        "deterministic_action": "tanh(mu)",
        "goal": list(canonical_command()),
        "actor_sha256_before": actor_before,
        "actor_sha256_after": actor_after,
        "training_rng_unchanged": True,
        "episodes": episodes,
        "summary": {
            "diagnostic_rsr": sum(row["canonical_success"] for row in episodes),
            "episode_count": len(episodes),
            "completion_count": sum(row["completed"] for row in episodes),
            "deadline_compliance_count": sum(
                row["deadline_compliant"] for row in episodes
            ),
            "speed_compliance_count": sum(row["speed_compliant"] for row in episodes),
            "exclusive_failure_mode_counts": {
                mode: sum(row["failure_mode"] == mode for row in episodes)
                for mode in modes
            },
            "mean_final_progress": fmean(row["final_progress"] for row in episodes),
            "feasible_only_energy_kwh": {
                "count": len(energies),
                "mean": fmean(energies) if energies else None,
                "values": energies,
            },
            "evaluation_simulator_transitions": transitions,
        },
    }


def evaluate_diagnostic(learner) -> dict[str, Any]:
    return evaluate_policy(
        learner,
        diagnostic_seeds(),
        evaluation_id="GCSL_DIAGNOSTIC_V1",
    )


def evaluate_validation(learner, configuration) -> dict[str, Any]:
    require_validation_authorized(configuration)
    return evaluate_policy(
        learner,
        validation_seeds(),
        evaluation_id="GCSL_VALIDATION_V1",
    )


def evaluate_constrained_reference_on_validation(
    policy,
    authorization_configuration,
    reference_configuration,
):
    """Optional future adapter; it shares a split but is not a strict control."""

    require_validation_authorized(authorization_configuration)
    from benchmarks.constrained_v2_collector_replication.evaluation import (
        evaluate_actor_on_seeds,
    )

    return evaluate_actor_on_seeds(
        policy,
        reference_configuration,
        evaluation_seeds=validation_seeds(),
        real_transitions=300_000,
        evaluation_id="GCSL_VALIDATION_V1_CONTEXTUAL_CONSTRAINED",
    )


def evaluate_paper(*_args, **_kwargs):
    forbid_paper()
