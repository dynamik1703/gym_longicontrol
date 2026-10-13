"""Exact multi-step soft targets built after a two-transition real prefix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .adapter import RealPhysicsReplay, TrackContext
from .physics_rollout import batched_step, observations, pack_states


@dataclass(frozen=True)
class TargetDecomposition:
    targets: np.ndarray
    real_prefix: np.ndarray
    model_expansion: np.ndarray
    bootstrap: np.ndarray
    model_steps: np.ndarray
    true_terminal: np.ndarray
    time_limit_bootstrap: np.ndarray

    def __post_init__(self) -> None:
        shape = self.targets.shape
        if shape != self.real_prefix.shape:
            raise ValueError("Target and real-prefix shapes differ")
        if shape != self.model_expansion.shape or shape != self.bootstrap.shape:
            raise ValueError("Target decomposition shapes differ")
        reconstructed = self.real_prefix + self.model_expansion + self.bootstrap
        if not np.allclose(self.targets, reconstructed, rtol=0.0, atol=2e-6):
            raise ValueError("Target decomposition does not reconstruct targets")


def recursive_soft_target(
    *,
    real_metrics: np.ndarray,
    model_metrics: np.ndarray,
    model_log_probabilities: np.ndarray,
    bootstrap_q: np.ndarray,
    bootstrap_log_probability: float,
    alpha: float,
    gamma: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytical nonterminal target used by deterministic semantic tests.

    ``real_metrics`` has shape ``(2, critics)`` and ``model_metrics`` has shape
    ``(M, critics)``. FSRL subtracts entropy from every critic, including both cost
    critics. Expanding its terminal soft value recursively therefore subtracts the
    sampled-action entropy at each modeled policy state and at final bootstrap.
    """

    real = np.asarray(real_metrics, dtype=np.float64)
    model = np.asarray(model_metrics, dtype=np.float64)
    log_prob = np.asarray(model_log_probabilities, dtype=np.float64)
    q = np.asarray(bootstrap_q, dtype=np.float64)
    if real.ndim != 2 or real.shape[0] != 2:
        raise ValueError("Exactly two real metric rows are required")
    if model.ndim != 2 or model.shape[1] != real.shape[1]:
        raise ValueError("Model metrics must match the critic count")
    if log_prob.shape != (model.shape[0],) or q.shape != (real.shape[1],):
        raise ValueError("Entropy or bootstrap shape changed")
    real_part = real[0] + gamma * real[1]
    model_part = np.zeros(real.shape[1], dtype=np.float64)
    for step in range(model.shape[0]):
        model_part += gamma ** (2 + step) * (
            model[step] - alpha * log_prob[step]
        )
    bootstrap_part = gamma ** (2 + model.shape[0]) * (
        q - alpha * float(bootstrap_log_probability)
    )
    return (
        real_part + model_part + bootstrap_part,
        real_part,
        model_part,
        bootstrap_part,
    )


def _policy_draw(
    policy: Any, observations_array: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Sample the pinned squashed Gaussian using only the dedicated rollout RNG."""

    import torch
    from tianshou.data import Batch

    values = np.asarray(observations_array, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("Policy draw requires a nonempty observation matrix")
    with torch.no_grad():
        logits, _state = policy.actor(values, state=None, info=Batch())
        mean, std = logits
        noise = torch.as_tensor(
            rng.standard_normal(tuple(mean.shape)),
            dtype=mean.dtype,
            device=mean.device,
        )
        raw = mean + std * noise
        distribution = torch.distributions.Independent(
            torch.distributions.Normal(mean, std), 1
        )
        log_probability = distribution.log_prob(raw).unsqueeze(-1)
        squashed = torch.tanh(raw)
        epsilon = np.finfo(np.float32).eps.item()
        log_probability = log_probability - torch.log(
            1 - squashed.pow(2) + epsilon
        ).sum(-1, keepdim=True)
    critic_actions = squashed.detach()
    physical_actions = np.asarray(
        policy.map_action(squashed.detach().cpu().numpy()), dtype=np.float64
    ).reshape(-1)
    return (
        physical_actions,
        log_probability.detach().cpu().numpy().reshape(-1).astype(np.float64),
        critic_actions,
    )


def _soft_bootstrap(
    policy: Any, observations_array: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    physical_actions, log_probability, critic_actions = _policy_draw(
        policy, observations_array, rng
    )
    del physical_actions
    values = []
    with __import__("torch").no_grad():
        for critic in policy.critics_old:
            q_value, _twins = critic.predict(observations_array, critic_actions)
            values.append(q_value.detach().cpu().numpy().reshape(-1))
    alpha = policy._alpha  # noqa: SLF001
    if hasattr(alpha, "detach"):
        alpha = alpha.detach().cpu().item()
    return np.column_stack(values) - float(alpha) * log_probability[:, None]


def _metrics(batch: Any, policy: Any) -> np.ndarray:
    columns = policy.get_metrics(batch)
    result = np.column_stack([np.asarray(column).reshape(-1) for column in columns])
    if result.shape[1] != policy.critics_num or not np.isfinite(result).all():
        raise RuntimeError("Replay metrics do not match the frozen critic order")
    return result.astype(np.float64)


def _transition_ids(batch: Any) -> np.ndarray:
    raw = batch.info.get("real_transition_id")
    if raw is None:
        raise RuntimeError("Real replay lost its immutable transition IDs")
    return np.asarray(raw, dtype=np.int64).reshape(-1)


def build_expanded_targets(
    policy: Any,
    batch: Any,
    buffer: Any,
    indices: np.ndarray,
    sidecar: RealPhysicsReplay,
    *,
    model_steps: int,
    rollout_rng: np.random.Generator,
    vehicle: Any,
    simulation_config: Any,
    energy_scale_kwh: float = 0.25,
    deadline_s: float = 140.0,
    max_episode_steps: int = 1800,
) -> tuple[Any, TargetDecomposition]:
    """Attach 2-real-plus-M-exact-physics targets to a sampled real batch."""

    import torch

    if model_steps <= 0:
        raise ValueError("M>0 is required; M=0 must use the untouched parity path")
    gamma = float(policy._gamma)  # noqa: SLF001
    if gamma != 0.99:
        raise RuntimeError("Frozen gamma changed")
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    batch_size = len(indices)
    if len(batch) != batch_size:
        raise ValueError("Sampled batch and replay indices differ")
    second_indices = np.asarray(buffer.next(indices), dtype=np.int64)
    second_batch = buffer[second_indices]
    first_metrics = _metrics(batch, policy)
    second_metrics = _metrics(second_batch, policy)
    critic_count = policy.critics_num

    unfinished = np.asarray(buffer.unfinished_index(), dtype=np.int64)
    first_end = np.asarray(buffer.done[indices], dtype=bool) | np.isin(
        indices, unfinished
    )
    second_end = np.asarray(buffer.done[second_indices], dtype=bool) | np.isin(
        second_indices, unfinished
    )
    first_true_terminal = np.asarray(buffer.terminated[indices], dtype=bool)
    second_true_terminal = np.asarray(
        buffer.terminated[second_indices], dtype=bool
    )

    real = first_metrics.copy()
    reaches_second = ~first_end
    real[reaches_second] += gamma * second_metrics[reaches_second]
    model = np.zeros((batch_size, critic_count), dtype=np.float64)
    bootstrap = np.zeros_like(model)
    targets = real.copy()
    steps_used = np.zeros(batch_size, dtype=np.int64)
    true_terminal = first_end & first_true_terminal
    time_limit_bootstrap = first_end & ~first_true_terminal

    first_bootstrap_rows = np.flatnonzero(first_end & ~first_true_terminal)
    if len(first_bootstrap_rows):
        soft = _soft_bootstrap(
            policy,
            np.asarray(batch.obs_next[first_bootstrap_rows], dtype=np.float32),
            rollout_rng,
        )
        bootstrap[first_bootstrap_rows] = gamma * soft

    second_stops = reaches_second & second_end
    second_terminal_rows = np.flatnonzero(second_stops & second_true_terminal)
    true_terminal[second_terminal_rows] = True
    second_bootstrap_rows = np.flatnonzero(second_stops & ~second_true_terminal)
    time_limit_bootstrap[second_bootstrap_rows] = True
    if len(second_bootstrap_rows):
        soft = _soft_bootstrap(
            policy,
            np.asarray(
                second_batch.obs_next[second_bootstrap_rows], dtype=np.float32
            ),
            rollout_rng,
        )
        bootstrap[second_bootstrap_rows] = gamma**2 * soft

    expandable_rows = np.flatnonzero(reaches_second & ~second_end)
    if len(expandable_rows):
        transition_ids = _transition_ids(second_batch)[expandable_rows]
        records = [sidecar.get(int(value)) for value in transition_ids]
        active_rows = expandable_rows.copy()
        active_state = pack_states([record.next_state for record in records])
        active_contexts: list[TrackContext] = [record.context for record in records]
        alpha = policy._alpha  # noqa: SLF001
        if hasattr(alpha, "detach"):
            alpha = alpha.detach().cpu().item()
        alpha = float(alpha)
        for imagined_step in range(model_steps):
            current_observations = observations(
                active_state, active_contexts, vehicle, simulation_config
            )
            actions, log_probability, _critic_actions = _policy_draw(
                policy, current_observations, rollout_rng
            )
            transition = batched_step(
                active_state,
                actions,
                contexts=active_contexts,
                vehicle=vehicle,
                config=simulation_config,
                energy_scale_kwh=energy_scale_kwh,
                deadline_s=deadline_s,
                max_episode_steps=max_episode_steps,
            )
            step_metrics = np.column_stack((transition.objective, transition.costs))
            discount = gamma ** (2 + imagined_step)
            model[active_rows] += discount * (
                step_metrics - alpha * log_probability[:, None]
            )
            steps_used[active_rows] += 1

            terminal_mask = np.asarray(transition.terminated, dtype=bool)
            truncated_mask = np.asarray(transition.truncated, dtype=bool)
            terminal_rows = active_rows[terminal_mask]
            true_terminal[terminal_rows] = True
            truncated_local = np.flatnonzero(truncated_mask)
            if len(truncated_local):
                truncated_rows = active_rows[truncated_local]
                soft = _soft_bootstrap(
                    policy,
                    transition.next_observation[truncated_local].astype(np.float32),
                    rollout_rng,
                )
                bootstrap[truncated_rows] += discount * gamma * soft
                time_limit_bootstrap[truncated_rows] = True
            survive = ~(terminal_mask | truncated_mask)
            if not np.any(survive):
                active_rows = np.empty(0, dtype=np.int64)
                break
            keep = np.flatnonzero(survive)
            active_rows = active_rows[keep]
            active_state = transition.state.take(keep)
            active_contexts = [active_contexts[int(index)] for index in keep]

        if len(active_rows):
            final_observations = observations(
                active_state, active_contexts, vehicle, simulation_config
            )
            bootstrap[active_rows] += gamma ** (2 + model_steps) * _soft_bootstrap(
                policy, final_observations.astype(np.float32), rollout_rng
            )

    targets = real + model + bootstrap
    dtype = next(policy.critics.parameters()).dtype
    device = next(policy.critics.parameters()).device
    batch.rets = torch.as_tensor(targets, dtype=dtype, device=device).unsqueeze(-2)
    decomposition = TargetDecomposition(
        targets=targets,
        real_prefix=real,
        model_expansion=model,
        bootstrap=bootstrap,
        model_steps=steps_used,
        true_terminal=true_terminal,
        time_limit_bootstrap=time_limit_bootstrap,
    )
    return batch, decomposition


def real_replay_update(
    policy: Any,
    real_buffer: Any,
    sidecar: RealPhysicsReplay,
    *,
    model_steps: int,
    rollout_rng: np.random.Generator,
    vehicle: Any,
    simulation_config: Any,
    batch_size: int = 256,
    energy_scale_kwh: float = 0.25,
    deadline_s: float = 140.0,
    max_episode_steps: int = 1800,
) -> TargetDecomposition | None:
    """Run one SACLag update; M=0 is the strict frozen parity path."""

    if model_steps == 0:
        policy.update(batch_size, real_buffer)
        return None
    if model_steps < 0:
        raise ValueError("model_steps must be nonnegative")
    batch, indices = real_buffer.sample(batch_size)
    policy.updating = True
    try:
        batch, decomposition = build_expanded_targets(
            policy,
            batch,
            real_buffer,
            indices,
            sidecar,
            model_steps=model_steps,
            rollout_rng=rollout_rng,
            vehicle=vehicle,
            simulation_config=simulation_config,
            energy_scale_kwh=energy_scale_kwh,
            deadline_s=deadline_s,
            max_episode_steps=max_episode_steps,
        )
        policy.learn(batch)
        policy.post_process_fn(batch, real_buffer, indices)
        if policy.lr_scheduler is not None:
            policy.lr_scheduler.step()
        return decomposition
    finally:
        policy.updating = False


def parameter_snapshot(policy: Any) -> dict[str, Any]:
    """Deep tensor snapshot used by strict M=0 update parity tests."""

    import copy

    return {
        "policy": copy.deepcopy(policy.state_dict()),
        "actor_optimizer": copy.deepcopy(policy.actor_optim.state_dict()),
        "critic_optimizer": copy.deepcopy(policy.critics_optim.state_dict()),
        "lagrangian": [copy.deepcopy(item.state_dict()) for item in policy.lag_optims],
        "alpha_optimizer": copy.deepcopy(policy._alpha_optim.state_dict()),  # noqa: SLF001
        "log_alpha": policy._log_alpha.detach().clone(),  # noqa: SLF001
    }


def snapshots_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Recursively require exact equality, including optimizer and target state."""

    import torch

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            snapshots_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            snapshots_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(left == right)
