"""Actor-only evaluation with explicit split gates."""

from __future__ import annotations

from typing import Any

from benchmarks.constrained_rl.adapter import deterministic_policy_adapter
from benchmarks.scalar_sac.evaluation import evaluate_policy
from benchmarks.scalar_sac.experiment import _base_environment

from .config import StudyConfiguration


def evaluate_development(policy: Any, configuration: StudyConfiguration):
    """Evaluate deterministic actor only; no model object is accepted or queried."""

    environment = _base_environment(configuration.v2)
    policy.eval()
    try:
        return evaluate_policy(
            deterministic_policy_adapter(policy),
            environment,
            task=configuration.v2.task,
            evaluation_seeds=configuration.raw["track_splits"]["development"],
        )
    finally:
        environment.close()
        policy.train()


def validation_ready(manifest: dict[str, Any]) -> bool:
    runs = tuple(manifest.get("runs", {}).values())
    return len(runs) == 6 and all(
        run.get("state") == "COMPLETED"
        and run.get("real_transitions") == 300_000
        and run.get("rl_gradient_updates") == 30_000
        and run.get("development_complete") is True
        and bool(run.get("final_checkpoint_sha256"))
        for run in runs
    )


def evaluate_validation(
    policy: Any, configuration: StudyConfiguration, manifest: dict[str, Any]
):
    if not configuration.raw["authorization"]["validation_authorized"]:
        raise PermissionError("Validation remains sealed in the preparation snapshot")
    if not validation_ready(manifest):
        raise RuntimeError("All six final policies must be frozen before Validation")
    if manifest.get("validation") != {"state": "SEALED", "opened_at": None}:
        raise RuntimeError("Validation may be opened exactly once")
    environment = _base_environment(configuration.v2)
    policy.eval()
    try:
        return evaluate_policy(
            deterministic_policy_adapter(policy),
            environment,
            task=configuration.v2.task,
            evaluation_seeds=configuration.raw["track_splits"]["validation"],
        )
    finally:
        environment.close()
        policy.train()


def evaluate_paper_test(*_args, **_kwargs):
    raise PermissionError("Paper-final tracks 4000-4017 remain sealed")
