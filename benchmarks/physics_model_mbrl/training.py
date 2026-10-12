"""Physics-only execution adapter around the byte-frozen MBRL trainer.

The adapter changes only manifest routing and observational checkpoint diagnostics.
The parent Physics model, update path, configuration, replay, RNGs and budgets are
used directly without copying or modifying their implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.model_based_rl import execution as parent_execution
from benchmarks.model_based_rl import training as parent_training

from .config import PHYSICS_CONDITION, TRAINING_SEEDS, load_configuration
from .execution import load_or_create_manifest


def _finite_logger_metrics(policy: Any) -> dict[str, float]:
    result = {}
    for name in tuple(policy.logger.logger_keys):
        value = float(policy.logger.get_mean(name))
        if np.isfinite(value):
            result[name] = value
    return result


def _checkpoint_with_observational_diagnostics(**kwargs):
    diagnostics = kwargs["diagnostics"]
    counters = kwargs["counters"]
    policy = kwargs["policy"]
    recorder = kwargs["recorder"]
    real_count = int(counters["real_transitions"])
    marker = "physics_only_checkpoint_diagnostics"
    already_recorded = any(
        row.get("record_type") == marker
        and int(row.get("real_transitions", -1)) == real_count
        for row in diagnostics[-3:]
    )
    if not already_recorded:
        completed = recorder.env.last_completed_episode
        costs = (
            None if completed is None else [float(value) for value in completed.costs]
        )
        diagnostics.append(
            {
                "record_type": marker,
                "real_transitions": real_count,
                "lambda_speed": float(policy.lag_optims[0].get_lag()),
                "lambda_deadline": float(policy.lag_optims[1].get_lag()),
                "latest_real_episode_speed_cost": None if costs is None else costs[0],
                "latest_real_episode_deadline_cost": None
                if costs is None
                else costs[1],
                "learner_metrics": _finite_logger_metrics(policy),
            }
        )
    return _PARENT_CHECKPOINT_BUILDER(**kwargs)


_PARENT_CHECKPOINT_BUILDER = parent_training._make_checkpoint_payload  # noqa: SLF001


def run_policy(
    *,
    training_seed: int,
    output_root: str | Path,
    device: str = "cpu",
    threads: int = 4,
    resume: bool = False,
    restart: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Run one Physics policy through the unchanged parent scientific trainer."""

    configuration = load_configuration()
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("Unauthorized Physics-only training seed")
    if (
        restart
        and not configuration.raw["authorization"]["automatic_fresh_restart_authorized"]
    ):
        raise PermissionError("Fresh restart requires separate explicit authorization")
    original_loader = parent_execution.load_or_create_manifest
    original_checkpoint_builder = parent_training._make_checkpoint_payload  # noqa: SLF001
    parent_execution.load_or_create_manifest = load_or_create_manifest
    parent_training._make_checkpoint_payload = (  # noqa: SLF001
        _checkpoint_with_observational_diagnostics
    )
    try:
        return parent_training.run_policy(
            condition=PHYSICS_CONDITION,
            training_seed=training_seed,
            output_root=output_root,
            device=device,
            threads=threads,
            resume=resume,
            restart=restart,
            checkpoint_path=checkpoint_path,
        )
    finally:
        parent_training._make_checkpoint_payload = original_checkpoint_builder  # noqa: SLF001
        parent_execution.load_or_create_manifest = original_loader
