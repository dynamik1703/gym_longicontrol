"""Narrow V2 adapter around the frozen V1 FSRL integration."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.constrained_rl.adapter import (
    alpha_value,
    deterministic_policy_adapter,
    enable_vector_costs,
    load_checkpoint,
    logger_snapshot,
    multiplier_values,
    save_checkpoint,
    split_reward_and_cost_metrics,
)
from benchmarks.constrained_rl.adapter import (
    build_agent as _build_v1_agent,
)

from .config import ConstrainedRLV2Configuration
from .costs import COST_NAMES, CompletedTrainingEpisode, DenseDeadlineTaskWrapper

__all__ = [
    "alpha_value",
    "build_agent",
    "collect_training_episode",
    "deterministic_policy_adapter",
    "enable_vector_costs",
    "load_checkpoint",
    "logger_snapshot",
    "multiplier_values",
    "save_checkpoint",
    "split_reward_and_cost_metrics",
]


def collect_training_episode(
    collector: Any,
    environment: DenseDeadlineTaskWrapper,
) -> tuple[dict[str, Any], CompletedTrainingEpisode]:
    """Collect one episode and restore exact separate episodic V2 costs."""

    environment.last_completed_episode = None
    raw_stats = collector.collect(n_episode=1)
    completed = environment.last_completed_episode
    if completed is None:
        raise RuntimeError("FSRL collector returned without a completed episode")
    if int(raw_stats["n/st"]) != completed.simulator_steps:
        raise RuntimeError("Collector and physical episode step counts differ")
    costs = np.asarray(completed.costs, dtype=np.float64)
    if costs.shape != (len(COST_NAMES),) or not np.isfinite(costs).all():
        raise RuntimeError("Invalid completed-episode cost vector")
    stats = dict(raw_stats)
    stats["cost"] = costs
    stats["total_cost"] = costs.copy()
    return stats, completed


def build_agent(
    configuration: ConstrainedRLV2Configuration,
    environment: DenseDeadlineTaskWrapper,
    *,
    training_seed: int,
    device: str,
    threads: int,
):
    """Build the exact V1 SACLag configuration for the V2 cost vector."""

    return _build_v1_agent(
        configuration,
        environment,
        training_seed=training_seed,
        device=device,
        threads=threads,
    )
