"""Non-intervening diagnostics computed from already sampled CRL batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .goal_adapter import canonical_command
from .losses import alpha_objective, association_score
from .replay import SampledFutureBatch
from .sampling import equivalent_goal_rate, lags_to_seconds


def _tree_l2_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    total = sum(float(jnp.sum(jnp.square(value))) for value in leaves)
    return float(np.sqrt(total))


def _finite_summary(values) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"minimum": None, "maximum": None, "mean": None, "std": None}
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def sampled_batch_diagnostics(
    sampled: SampledFutureBatch,
    projected_goals,
    *,
    dt_s: float,
) -> dict[str, Any]:
    """Describe the exact pair/goal batch without drawing another sample."""

    goals = np.asarray(projected_goals, dtype=np.float64)
    if goals.ndim != 2 or goals.shape != (len(sampled.lags), 3):
        raise ValueError("projected goals must align with sampled pairs")
    unique_goals = np.unique(goals, axis=0)
    canonical = np.all(goals == canonical_command(), axis=1)
    timely = goals[:, 1] == 1.0
    safe = goals[:, 2] == 1.0
    safe_timely_progress = goals[timely & safe, 0]
    source_progress = sampled.source_outcomes[:, 0] / 1000.0
    progress_distance = goals[:, 0] - source_progress
    requirement_frequencies = {}
    for deadline, compliance in ((1, 1), (0, 1), (1, 0), (0, 0)):
        key = f"star-{deadline}-{compliance}"
        requirement_frequencies[key] = float(
            np.mean((goals[:, 1] == deadline) & (goals[:, 2] == compliance))
        )
    return {
        "sampled_pair_count": int(len(goals)),
        "unique_source_transition_count": int(len(np.unique(sampled.source_indices))),
        "future_lag_decisions": _finite_summary(sampled.lags),
        "future_lag_seconds": _finite_summary(lags_to_seconds(sampled.lags, dt_s=dt_s)),
        "target_progress_distance": _finite_summary(progress_distance),
        "terminal_future_fraction": float(np.mean(sampled.future_terminated)),
        "late_future_fraction": float(np.mean(~timely)),
        "unsafe_future_fraction": float(np.mean(~safe)),
        "timely_safe_future_fraction": float(np.mean(timely & safe)),
        "exact_projected_goal_support": {
            "unique_goal_count": int(len(unique_goals)),
            "duplicate_column_count": int(len(goals) - len(unique_goals)),
            "duplicate_column_fraction": float(1.0 - len(unique_goals) / len(goals)),
            "pair_collision_rate": equivalent_goal_rate(goals),
            "all_identical": bool(len(unique_goals) == 1),
            "distinct_progress_count": int(len(np.unique(goals[:, 0]))),
            "requirement_bit_frequencies": requirement_frequencies,
        },
        "diagnostic_binning": None,
        "canonical_positive_count": int(np.sum(canonical)),
        "contains_canonical_positive": bool(np.any(canonical)),
        "minimum_goal_distance_to_canonical": float(
            np.min(np.linalg.norm(goals - canonical_command(), axis=1))
        ),
        "safe_timely_progress": _finite_summary(safe_timely_progress),
    }


def learning_batch_diagnostics(learner, state, batch, actor_key) -> dict[str, Any]:
    """Compute gradients/scores from the existing batch and existing RNG key.

    JAX functions are pure: reusing ``actor_key`` here neither consumes nor
    advances the runner's key. Returned values are never fed into an update.
    """

    def actor_loss(params):
        return learner.actor_loss(
            params,
            state.critic.params,
            state.alpha.params["log_alpha"],
            batch,
            actor_key,
        )

    (actor_loss_value, actor_aux), actor_gradients = jax.value_and_grad(
        actor_loss, has_aux=True
    )(state.actor.params)

    def critic_loss(params):
        return learner.critic_loss(params, batch)

    (critic_loss_value, critic_aux), critic_gradients = jax.value_and_grad(
        critic_loss, has_aux=True
    )(state.critic.params)

    def temperature_loss(params):
        return alpha_objective(
            params["log_alpha"],
            actor_aux["log_probability"],
            learner.config.target_entropy,
        )

    alpha_loss_value, alpha_gradients = jax.value_and_grad(temperature_loss)(
        state.alpha.params
    )
    state_action = learner.state_action_encoder.apply(
        state.critic.params["state_action"], batch.states, batch.actions
    )
    goal_embedding = learner.goal_encoder.apply(
        state.critic.params["goal"], batch.critic_goals
    )
    logits = np.asarray(critic_aux["logits"], dtype=np.float64)
    positive = np.diag(logits)
    reference = logits[~np.eye(len(logits), dtype=bool)]
    command = jnp.broadcast_to(
        jnp.asarray(canonical_command(), dtype=batch.critic_goals.dtype),
        batch.critic_goals.shape,
    )
    command_embedding = learner.goal_encoder.apply(state.critic.params["goal"], command)
    canonical_scores = association_score(state_action, command_embedding)
    sampled_scores = association_score(state_action, goal_embedding)
    canonical_actions = learner.deterministic_action(
        state.actor.params, batch.states, command
    )
    sampled_actions = learner.deterministic_action(
        state.actor.params, batch.states, batch.actor_goals
    )
    values = [
        actor_loss_value,
        critic_loss_value,
        alpha_loss_value,
        *jax.tree_util.tree_leaves(actor_gradients),
        *jax.tree_util.tree_leaves(critic_gradients),
        *jax.tree_util.tree_leaves(alpha_gradients),
    ]
    nonfinite = sum(
        int(np.size(value) - np.count_nonzero(np.isfinite(np.asarray(value))))
        for value in values
    )
    return {
        "optimization": {
            "actor_loss": float(actor_loss_value),
            "critic_loss": float(critic_loss_value),
            "alpha_loss": float(alpha_loss_value),
            "log_alpha": float(state.alpha.params["log_alpha"]),
            "mean_log_probability": float(actor_aux["mean_log_probability"]),
            "estimated_entropy": -float(actor_aux["mean_log_probability"]),
            "actor_gradient_norm": _tree_l2_norm(actor_gradients),
            "critic_gradient_norm": _tree_l2_norm(critic_gradients),
            "state_action_encoder_gradient_norm": _tree_l2_norm(
                critic_gradients["state_action"]
            ),
            "goal_encoder_gradient_norm": _tree_l2_norm(critic_gradients["goal"]),
            "nonfinite_value_count": nonfinite,
        },
        "contrastive": {
            "infonce_component": float(critic_aux["classification_loss"]),
            "logsumexp_regularization": float(critic_aux["logsumexp_regularizer"]),
            "positive_scores": _finite_summary(positive),
            "reference_scores": _finite_summary(reference),
            "positive_minus_reference_mean": float(
                np.mean(positive) - np.mean(reference)
            ),
            "state_action_embedding_norms": _finite_summary(
                np.linalg.norm(np.asarray(state_action), axis=1)
            ),
            "goal_embedding_norms": _finite_summary(
                np.linalg.norm(np.asarray(goal_embedding), axis=1)
            ),
            "state_action_embedding_variance": float(
                np.mean(np.var(np.asarray(state_action), axis=0))
            ),
            "goal_embedding_variance": float(
                np.mean(np.var(np.asarray(goal_embedding), axis=0))
            ),
        },
        "canonical_query": {
            "actual_action_canonical_goal_score": _finite_summary(canonical_scores),
            "actual_action_sampled_goal_score": _finite_summary(sampled_scores),
            "canonical_minus_sampled_score": _finite_summary(
                np.asarray(canonical_scores) - np.asarray(sampled_scores)
            ),
            "actor_action_canonical_goal": _finite_summary(canonical_actions),
            "actor_action_sampled_goal": _finite_summary(sampled_actions),
        },
    }


@dataclass
class DiagnosticsAccumulator:
    """Checkpointable compact accounting; it never samples on its own."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    seen_source_indices: set[int] = field(default_factory=set)
    batch_count: int = 0
    batches_with_canonical_positive: int = 0
    canonical_positive_count: int = 0
    all_identical_goal_batches: int = 0
    first_sampled_canonical_positive_transition: int | None = None
    nonfinite_value_count: int = 0

    def record(
        self,
        *,
        transition_count: int,
        update_cycle: int,
        sampled: SampledFutureBatch,
        projected_goals,
        dt_s: float,
        learning: dict[str, Any] | None,
    ) -> dict[str, Any]:
        batch = sampled_batch_diagnostics(sampled, projected_goals, dt_s=dt_s)
        self.seen_source_indices.update(map(int, sampled.source_indices))
        self.batch_count += 1
        if batch["contains_canonical_positive"]:
            self.batches_with_canonical_positive += 1
            if self.first_sampled_canonical_positive_transition is None:
                self.first_sampled_canonical_positive_transition = transition_count
        self.canonical_positive_count += batch["canonical_positive_count"]
        if batch["exact_projected_goal_support"]["all_identical"]:
            self.all_identical_goal_batches += 1
        if learning is not None:
            self.nonfinite_value_count += learning["optimization"][
                "nonfinite_value_count"
            ]
        row = {
            "transition_count": int(transition_count),
            "update_cycle": int(update_cycle),
            "batch": batch,
            "learning": learning,
        }
        self.rows.append(row)
        return row

    def summary(self) -> dict[str, Any]:
        return {
            "learning_batch_count": self.batch_count,
            "unique_source_transition_coverage": len(self.seen_source_indices),
            "batches_with_canonical_positive": self.batches_with_canonical_positive,
            "observed_canonical_future_positive_count": (self.canonical_positive_count),
            "canonical_positive_batch_fraction": (
                self.batches_with_canonical_positive / self.batch_count
                if self.batch_count
                else 0.0
            ),
            "all_identical_goal_batch_count": self.all_identical_goal_batches,
            "all_identical_goal_batch_frequency": (
                self.all_identical_goal_batches / self.batch_count
                if self.batch_count
                else 0.0
            ),
            "first_sampled_canonical_positive_transition": (
                self.first_sampled_canonical_positive_transition
            ),
            "nonfinite_value_count": self.nonfinite_value_count,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "seen_source_indices": sorted(self.seen_source_indices),
            **self.summary(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> DiagnosticsAccumulator:
        return cls(
            rows=list(state["rows"]),
            seen_source_indices=set(state["seen_source_indices"]),
            batch_count=int(state["learning_batch_count"]),
            batches_with_canonical_positive=int(
                state["batches_with_canonical_positive"]
            ),
            canonical_positive_count=int(
                state["observed_canonical_future_positive_count"]
            ),
            all_identical_goal_batches=int(state["all_identical_goal_batch_count"]),
            first_sampled_canonical_positive_transition=state[
                "first_sampled_canonical_positive_transition"
            ],
            nonfinite_value_count=int(state["nonfinite_value_count"]),
        )
