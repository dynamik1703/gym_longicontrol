"""Pinned FSRL adapter for requirement-conditioned observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.constrained_rl.adapter import (
    alpha_value,
    deterministic_policy_adapter,
    load_checkpoint,
    logger_snapshot,
    multiplier_values,
    save_checkpoint,
)
from benchmarks.constrained_rl.adapter import build_agent as _build_agent
from benchmarks.constrained_rl_v2.costs import COST_NAMES

from .requirements import CompletedRequirementEpisode, RequirementConditionedTaskWrapper


def build_agent(configuration, environment, *, training_seed, device, threads):
    return _build_agent(
        configuration,
        environment,
        training_seed=training_seed,
        device=device,
        threads=threads,
    )


def collect_training_episode(
    collector: Any,
    environment: RequirementConditionedTaskWrapper,
    *,
    requirement_margin_s: float,
) -> tuple[dict[str, Any], CompletedRequirementEpisode]:
    reset_options = {"options": {"requirement_margin_s": requirement_margin_s}}
    # FastCollector resets once when an episode ends and once again before returning.
    # Set the intended requirement immediately before collection so those internal,
    # discarded resets cannot advance the balanced requirement schedule.
    collector.reset_env(reset_options)
    environment.last_completed_episode = None
    raw_stats = collector.collect(n_episode=1, gym_reset_kwargs=reset_options)
    completed = environment.last_completed_episode
    if completed is None:
        raise RuntimeError("Collector returned without a completed episode")
    if int(raw_stats["n/st"]) != completed.simulator_steps:
        raise RuntimeError("Collector and simulator step counts differ")
    costs = np.asarray(completed.costs, dtype=np.float64)
    if costs.shape != (len(COST_NAMES),) or not np.isfinite(costs).all():
        raise RuntimeError("Invalid episodic cost vector")
    stats = dict(raw_stats)
    stats["cost"] = costs
    stats["total_cost"] = costs.copy()
    return stats, completed


__all__ = [
    "alpha_value",
    "build_agent",
    "collect_training_episode",
    "deterministic_policy_adapter",
    "load_checkpoint",
    "logger_snapshot",
    "multiplier_values",
    "save_checkpoint",
]
