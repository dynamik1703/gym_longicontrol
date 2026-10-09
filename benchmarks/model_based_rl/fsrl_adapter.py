"""Narrow FSRL adapter for matched mixed-source SACLag updates.

FSRL's frozen ``n_step=2`` processing requires contiguous indices from one replay
buffer. H=1 synthetic counterfactuals have no legitimate second synthetic step.
Consequently real samples retain the exact V2 two-step target, while synthetic
samples use a one-step bootstrapped target. Actor, critics, optimizers, gamma, tau,
PID and update count remain frozen. Model-disabled parity uses the unmodified V2
``policy.update`` path. This unavoidable target-horizon adaptation is explicit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .model_state import ProjectedTransition


def synthetic_batch(rows: list[ProjectedTransition]):
    from tianshou.data import Batch

    return Batch(
        obs=np.stack([row.observation for row in rows]).astype(np.float32),
        act=np.asarray([[row.action] for row in rows], dtype=np.float32),
        rew=np.asarray([row.objective for row in rows], dtype=np.float32),
        terminated=np.asarray([row.terminated for row in rows]),
        truncated=np.asarray([row.truncated for row in rows]),
        obs_next=np.stack([row.next_observation for row in rows]).astype(np.float32),
        info=Batch(
            cost=np.asarray([row.costs for row in rows], dtype=np.float32),
            source=np.asarray(["model"] * len(rows)),
            source_real_transition_id=np.asarray(
                [row.source_real_transition_id for row in rows]
            ),
            model_version=np.asarray([row.model_version for row in rows]),
        ),
        policy=Batch(),
    )


def process_one_step_synthetic(policy: Any, batch):
    """Create correct one-step bootstrapped reward/cost targets for H=1 rows."""

    import torch

    metrics = [batch.rew, *np.asarray(batch.info.cost).T]
    with torch.no_grad():
        result = policy(batch, input="obs_next")
        targets = []
        valid = torch.as_tensor(
            (~np.asarray(batch.terminated, dtype=bool)).astype(np.float32),
            device=result.act.device,
        )
        for index, critic in enumerate(policy.critics_old):
            target_q, _ = critic.predict(batch.obs_next, result.act)
            target_q = target_q.flatten() - (
                policy._alpha * result.log_prob.flatten()  # noqa: SLF001
            )
            metric = torch.as_tensor(
                metrics[index], dtype=target_q.dtype, device=target_q.device
            )
            targets.append(metric + policy._gamma * valid * target_q)  # noqa: SLF001
    # Pinned FSRL keeps a singleton target dimension: (batch, 1, critics).
    batch.rets = torch.stack(targets, dim=-1).unsqueeze(-2)
    return batch


def source_td_errors(policy: Any, batch, real_count: int) -> dict[str, float]:
    import torch

    with torch.no_grad():
        errors = []
        for index, critic in enumerate(policy.critics):
            q1, q2 = critic(batch.obs, batch.act)
            target = batch.rets[..., index].flatten()
            errors.append(((q1.flatten() + q2.flatten()) * 0.5 - target).abs())
        values = torch.stack(errors).mean(dim=0).detach().cpu().numpy()
    return {
        "td_error_real_mean": float(values[:real_count].mean()),
        "td_error_model_mean": float(values[real_count:].mean()),
    }


def mixed_policy_update(
    policy: Any,
    real_buffer: Any,
    synthetic_rows: list[ProjectedTransition],
    *,
    real_batch_size: int = 128,
) -> dict[str, float]:
    """Run one matched 128-real/128-model optimizer step."""

    from tianshou.data import Batch

    if len(synthetic_rows) != real_batch_size:
        raise ValueError("Frozen mixture requires 128 real and 128 model samples")
    real_batch, real_indices = real_buffer.sample(real_batch_size)
    policy.updating = True
    try:
        processed_real = policy.process_fn(real_batch, real_buffer, real_indices)
        processed_model = process_one_step_synthetic(
            policy, synthetic_batch(synthetic_rows)
        )
        mixed = Batch.cat([processed_real, processed_model])
        diagnostics = source_td_errors(policy, mixed, real_batch_size)
        policy.learn(mixed)
        policy.post_process_fn(processed_real, real_buffer, real_indices)
        if policy.lr_scheduler is not None:
            policy.lr_scheduler.step()
        return diagnostics
    finally:
        policy.updating = False


def model_disabled_update(policy: Any, real_buffer: Any, batch_size: int = 256) -> None:
    """Exact frozen V2 update path for bounded implementation parity checks."""

    policy.update(batch_size, real_buffer)
