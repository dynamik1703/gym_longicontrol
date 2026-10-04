"""Narrow adapter between LongiControl cost vectors and pinned FSRL."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

from .config import ConstrainedRLConfiguration
from .costs import COST_NAMES, CompletedTrainingEpisode, ConstrainedTaskWrapper


def split_reward_and_cost_metrics(batch: Any, constraint_count: int) -> list[Any]:
    """Return one reward column followed by one column per explicit cost.

    FSRL's pinned base policy returns ``[reward, cost_matrix]`` although its critic
    loop expects one metric per critic. This function performs only that missing
    column split and validates ordering/shape.
    """

    if constraint_count <= 0:
        raise ValueError("constraint_count must be positive")
    reward = np.asarray(batch.rew)
    raw = batch.info.get("cost")
    if raw is None:
        raise ValueError("FSRL batches must retain info['cost']")
    costs = np.asarray(raw, dtype=reward.dtype)
    expected_shape = (*reward.shape, constraint_count)
    if costs.shape != expected_shape:
        raise ValueError(
            f"Expected cost shape {expected_shape}, received {costs.shape}"
        )
    if not np.isfinite(costs).all():
        raise ValueError("Constraint costs must remain finite")
    return [reward, *(costs[..., index] for index in range(constraint_count))]


def enable_vector_costs(policy: Any, constraint_count: int) -> None:
    """Attach the pinned-FSRL vector-cost compatibility method to one policy."""

    if getattr(policy, "critics_num", None) != constraint_count + 1:
        raise ValueError("FSRL critic count does not match the constraint vector")

    def get_metrics(_policy, batch):
        return split_reward_and_cost_metrics(batch, constraint_count)

    policy.get_metrics = MethodType(get_metrics, policy)


def collect_training_episode(
    collector: Any,
    environment: ConstrainedTaskWrapper,
) -> tuple[dict[str, Any], CompletedTrainingEpisode]:
    """Collect one episode and restore exact per-constraint FSRL statistics."""

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
    # Pinned FastCollector sums every cost column into one scalar. PID-Lagrangian
    # requires the two undiscounted episodic returns independently.
    stats["cost"] = costs
    stats["total_cost"] = costs.copy()
    return stats, completed


def build_agent(
    configuration: ConstrainedRLConfiguration,
    environment: ConstrainedTaskWrapper,
    *,
    training_seed: int,
    device: str,
    threads: int,
):
    """Construct pinned FSRL SACLag with frozen, explicit settings."""

    from fsrl.agent import SACLagAgent
    from fsrl.utils import BaseLogger

    values = configuration.algorithm
    logger = BaseLogger(log_dir=None, log_txt=False, name="longicontrol-sacl-ag")
    agent = SACLagAgent(
        environment,
        logger=logger,
        cost_limit=list(configuration.constraints.cost_limits),
        device=device,
        thread=threads,
        seed=training_seed,
        actor_lr=values.actor_learning_rate,
        critic_lr=values.critic_learning_rate,
        hidden_sizes=values.hidden_sizes,
        auto_alpha=values.automatic_entropy_tuning,
        alpha_lr=values.alpha_learning_rate,
        tau=values.tau,
        n_step=values.n_step,
        use_lagrangian=True,
        lagrangian_pid=values.lagrangian_pid,
        rescaling=values.lagrangian_rescaling,
        gamma=values.gamma,
        deterministic_eval=values.deterministic_evaluation,
        action_scaling=True,
        action_bound_method="clip",
    )
    enable_vector_costs(agent.policy, len(configuration.constraints.names))
    actual = tuple(float(item.get_lag()) for item in agent.policy.lag_optims)
    if actual != values.initial_multipliers:
        raise RuntimeError(f"Unexpected initial multipliers: {actual}")
    return agent, logger


def deterministic_policy_adapter(policy: Any):
    """Expose an FSRL policy through the shared deterministic evaluator API."""

    from tianshou.data import Batch, to_numpy

    def act(observation: np.ndarray) -> np.ndarray:
        batch = Batch(
            obs=np.asarray(observation, dtype=np.float32).reshape((1, -1)),
            info=Batch(),
        )
        result = policy(batch)
        action = policy.map_action(to_numpy(result.act))
        return np.asarray(action, dtype=np.float64).reshape((1,))

    return act


def multiplier_values(policy: Any) -> tuple[float, ...]:
    values = tuple(float(item.get_lag()) for item in policy.lag_optims)
    if not np.isfinite(values).all():
        raise RuntimeError("A Lagrange multiplier became non-finite")
    return values


def alpha_value(policy: Any) -> float:
    value = getattr(policy, "_alpha")
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError("Entropy alpha became non-finite")
    return result


def logger_snapshot(logger: Any) -> dict[str, float]:
    result = {}
    for name in tuple(logger.logger_keys):
        value = float(logger.get_mean(name))
        if not np.isfinite(value):
            raise RuntimeError(f"Non-finite FSRL diagnostic: {name}")
        result[name] = value
    return result


def save_checkpoint(path: str | Path, policy: Any, *, metadata: Mapping[str, Any]):
    """Persist model plus PID state explicitly (pinned FSRL load omits PID state)."""

    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "lagrangian_states": [item.state_dict() for item in policy.lag_optims],
            "metadata": dict(metadata),
        },
        destination,
    )


def load_checkpoint(path: str | Path, policy: Any, *, device: str = "cpu") -> dict:
    import torch

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    policy.load_state_dict(payload["policy_state_dict"])
    for optimizer, state in zip(policy.lag_optims, payload["lagrangian_states"]):
        optimizer.load_state_dict(state)
    return dict(payload["metadata"])
