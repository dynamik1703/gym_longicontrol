"""Small source-aligned CRL learner core for synthetic verification only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training.train_state import TrainState

from .config import ReferenceCoreConfig
from .losses import (
    actor_objective,
    alpha_objective,
    association_score,
    contrastive_loss,
    tanh_gaussian_sample,
)
from .networks import Actor, GoalEncoder, StateActionEncoder


@flax.struct.dataclass
class ContrastiveBatch:
    """Learning inputs; ``historical_reward`` is retained only for leakage tests."""

    states: jnp.ndarray
    actions: jnp.ndarray
    critic_goals: jnp.ndarray
    actor_goals: jnp.ndarray
    historical_reward: jnp.ndarray | None = None


@flax.struct.dataclass
class LearnerState:
    actor: TrainState
    critic: TrainState
    alpha: TrainState
    gradient_steps: jnp.ndarray


@dataclass(frozen=True)
class ReferenceLearner:
    config: ReferenceCoreConfig
    actor: Actor
    state_action_encoder: StateActionEncoder
    goal_encoder: GoalEncoder

    @classmethod
    def create(
        cls, config: ReferenceCoreConfig, *, seed: int = 0
    ) -> tuple[ReferenceLearner, LearnerState]:
        actor = Actor(config.action_dim, config.width, config.depth)
        state_action_encoder = StateActionEncoder(
            config.width, config.depth, config.embedding_dim
        )
        goal_encoder = GoalEncoder(config.width, config.depth, config.embedding_dim)
        key = jax.random.PRNGKey(seed)
        actor_key, state_action_key, goal_key = jax.random.split(key, 3)
        actor_params = actor.init(
            actor_key, jnp.ones((1, config.state_dim + config.goal_dim))
        )
        state_action_params = state_action_encoder.init(
            state_action_key,
            jnp.ones((1, config.state_dim)),
            jnp.ones((1, config.action_dim)),
        )
        goal_params = goal_encoder.init(
            goal_key, jnp.ones((1, config.goal_dim))
        )
        state = LearnerState(
            actor=TrainState.create(
                apply_fn=actor.apply,
                params=actor_params,
                tx=optax.adam(config.actor_lr),
            ),
            critic=TrainState.create(
                apply_fn=None,
                params={"state_action": state_action_params, "goal": goal_params},
                tx=optax.adam(config.critic_lr),
            ),
            alpha=TrainState.create(
                apply_fn=None,
                params={"log_alpha": jnp.asarray(0.0, dtype=jnp.float32)},
                tx=optax.adam(config.alpha_lr),
            ),
            gradient_steps=jnp.asarray(0, dtype=jnp.int32),
        )
        return cls(config, actor, state_action_encoder, goal_encoder), state

    def critic_loss(self, critic_params, batch: ContrastiveBatch):
        """Environment rewards intentionally do not enter this computation."""

        state_action = self.state_action_encoder.apply(
            critic_params["state_action"], batch.states, batch.actions
        )
        goals = self.goal_encoder.apply(critic_params["goal"], batch.critic_goals)
        return contrastive_loss(
            state_action,
            goals,
            epsilon=self.config.distance_epsilon,
            logsumexp_penalty=self.config.logsumexp_penalty,
        )

    def actor_loss(self, actor_params, critic_params, log_alpha, batch, key):
        """Use the caller-provided actor-goal distribution, as upstream does."""

        state_goal = jnp.concatenate((batch.states, batch.actor_goals), axis=-1)
        mean, log_std = self.actor.apply(actor_params, state_goal)
        noise = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        actions, log_probability = tanh_gaussian_sample(mean, log_std, noise)
        state_action = self.state_action_encoder.apply(
            critic_params["state_action"], batch.states, actions
        )
        goals = self.goal_encoder.apply(critic_params["goal"], batch.actor_goals)
        score = association_score(
            state_action, goals, epsilon=self.config.distance_epsilon
        )
        return actor_objective(score, log_probability, log_alpha), {
            "mean_score": jnp.mean(score),
            "mean_log_probability": jnp.mean(log_probability),
            "log_probability": log_probability,
        }

    def actor_alpha_step(self, state: LearnerState, batch, key):
        """Actor/alpha update first; critic parameters are read-only."""

        (actor_loss_value, actor_metrics), actor_gradients = jax.value_and_grad(
            self.actor_loss, argnums=0, has_aux=True
        )(
            state.actor.params,
            state.critic.params,
            state.alpha.params["log_alpha"],
            batch,
            key,
        )
        actor = state.actor.apply_gradients(grads=actor_gradients)

        def temperature_loss(alpha_params):
            return alpha_objective(
                alpha_params["log_alpha"],
                actor_metrics["log_probability"],
                self.config.target_entropy,
            )

        alpha_loss_value, alpha_gradients = jax.value_and_grad(temperature_loss)(
            state.alpha.params
        )
        alpha = state.alpha.apply_gradients(grads=alpha_gradients)
        new_state = state.replace(actor=actor, alpha=alpha)
        metrics = {
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "mean_score": actor_metrics["mean_score"],
            "mean_log_probability": actor_metrics["mean_log_probability"],
        }
        return new_state, metrics

    def critic_step(self, state: LearnerState, batch):
        (loss, metrics), gradients = jax.value_and_grad(
            self.critic_loss, has_aux=True
        )(state.critic.params, batch)
        critic = state.critic.apply_gradients(grads=gradients)
        return state.replace(critic=critic), {"critic_loss": loss, **metrics}

    def update(self, state: LearnerState, batch, key):
        """Match upstream order: actor/alpha, then critic."""

        state, actor_metrics = self.actor_alpha_step(state, batch, key)
        state, critic_metrics = self.critic_step(state, batch)
        state = state.replace(gradient_steps=state.gradient_steps + 1)
        return state, {**actor_metrics, **critic_metrics}

    def deterministic_action(self, actor_params, states, goals):
        mean, _ = self.actor.apply(
            actor_params, jnp.concatenate((states, goals), axis=-1)
        )
        return jnp.tanh(mean)


def save_state(path: str | Path, state: LearnerState) -> None:
    Path(path).write_bytes(serialization.to_bytes(state))


def load_state(path: str | Path, template: LearnerState) -> LearnerState:
    return serialization.from_bytes(template, Path(path).read_bytes())


def tree_parameter_count(tree) -> int:
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(tree))
