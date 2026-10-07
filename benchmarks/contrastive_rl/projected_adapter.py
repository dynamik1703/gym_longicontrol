"""Mechanically map recorded physical future outcomes into CRL batches."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gym_longicontrol.domain.task import TaskSpecification

from .goal_adapter import project_outcome
from .learner import ContrastiveBatch


def contrastive_batch_from_recorded(
    *,
    states,
    actions,
    future_outcomes,
    future_terminated,
    task: TaskSpecification,
    historical_reward=None,
) -> ContrastiveBatch:
    """Use only actual stored futures for critic and upstream-style actor goals.

    The canonical collection command is deliberately absent: this function
    cannot insert an unobserved success into replay. Raw outcomes stay with the
    caller's replay/provenance record and are only read by the pure projection.
    """

    state_array = np.asarray(states)
    action_array = np.asarray(actions)
    outcome_array = np.asarray(future_outcomes, dtype=np.float64)
    if not (
        len(state_array) == len(action_array) == len(outcome_array)
    ):
        raise ValueError("states, actions, and future outcomes must align")
    projected = project_outcome(
        outcome_array,
        terminated=future_terminated,
        task=task,
    )
    rewards = None
    if historical_reward is not None:
        rewards = np.asarray(historical_reward)
        if len(rewards) != len(state_array):
            raise ValueError("historical rewards must align with the batch")
    return ContrastiveBatch(
        states=jnp.asarray(state_array),
        actions=jnp.asarray(action_array),
        critic_goals=jnp.asarray(projected),
        actor_goals=jnp.asarray(projected),
        historical_reward=None if rewards is None else jnp.asarray(rewards),
    )
