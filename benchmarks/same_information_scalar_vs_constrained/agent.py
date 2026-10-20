"""Matched one-critic-system scalar SAC construction in the pinned FSRL stack."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def build_scalar_agent(
    environment: Any, *, training_seed: int, device: str = "cpu", threads: int = 4
):
    import torch
    from fsrl.agent import SACLagAgent
    from fsrl.policy import SACLagrangian
    from fsrl.utils import BaseLogger

    logger = BaseLogger(log_dir=None, log_txt=False, name="same-information-scalar-sac")
    # Pinned FSRL constructs every network, then orthogonally initializes the whole
    # ActorCritic. Consequently, the number of allocated critics changes even the
    # actor's initial bytes. Reproduce the immutable historical three-system
    # constructor once, retain actor + reward critic, and discard both cost critics.
    historical = SACLagAgent(
        environment,
        logger=logger,
        cost_limit=[0.0, 0.0],
        device=device,
        thread=threads,
        seed=int(training_seed),
        actor_lr=0.0005,
        critic_lr=0.001,
        hidden_sizes=(128, 128),
        auto_alpha=True,
        alpha_lr=0.0003,
        alpha=1.0,
        tau=0.05,
        n_step=2,
        use_lagrangian=True,
        lagrangian_pid=(0.05, 0.0005, 0.1),
        rescaling=True,
        gamma=0.99,
        deterministic_eval=True,
        action_scaling=True,
        action_bound_method="clip",
    )
    source = historical.policy
    reward_critic = source.critics[0]
    critic_optim = torch.optim.Adam(reward_critic.parameters(), lr=0.001)
    policy = SACLagrangian(
        actor=source.actor,
        critics=[reward_critic],
        actor_optim=source.actor_optim,
        critic_optim=critic_optim,
        logger=logger,
        alpha=(source._target_entropy, source._log_alpha, source._alpha_optim),  # noqa: SLF001
        tau=0.05,
        n_step=2,
        use_lagrangian=False,
        lagrangian_pid=(0.05, 0.0005, 0.1),
        cost_limit=[],
        rescaling=False,
        gamma=0.99,
        reward_normalization=False,
        deterministic_eval=True,
        action_scaling=True,
        action_bound_method="clip",
        observation_space=environment.observation_space,
        action_space=environment.action_space,
    )
    agent = SimpleNamespace(policy=policy)
    if policy.critics_num != 1 or len(policy.critics) != 1:
        raise RuntimeError("Scalar SAC must have exactly one double-Q critic system")
    if policy.use_lagrangian or policy.lag_optims:
        raise RuntimeError("Scalar SAC must not contain PID/Lagrange state")
    if tuple(policy.cost_limit):
        raise RuntimeError("Scalar SAC must not contain constraint limits")
    return agent, logger


def scalar_policy_invariants(policy: Any) -> dict[str, Any]:
    return {
        "critic_systems": int(policy.critics_num),
        "double_q_heads": 2,
        "cost_critics": 0,
        "use_lagrangian": bool(policy.use_lagrangian),
        "pid_optimizers": len(policy.lag_optims),
        "automatic_alpha": bool(policy._is_auto_alpha),  # noqa: SLF001
        "target_entropy": float(policy._target_entropy),  # noqa: SLF001
        "n_step": int(policy._n_step),  # noqa: SLF001
    }
