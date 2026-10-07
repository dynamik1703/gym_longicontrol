"""Source-aligned contrastive, actor, and temperature objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def association_score(
    state_action_embedding: jnp.ndarray,
    goal_embedding: jnp.ndarray,
    *,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    """Return negative Euclidean distance: higher means stronger association.

    Upstream uses an unguarded square root. The tiny positive radicand is the
    sole numerical deviation and prevents undefined gradients for coincident
    embeddings; it neither calibrates the score nor turns it into probability.
    """

    squared_distance = jnp.sum(
        jnp.square(state_action_embedding - goal_embedding), axis=-1
    )
    return -jnp.sqrt(squared_distance + epsilon)


def pairwise_association_scores(
    state_action_embeddings: jnp.ndarray,
    goal_embeddings: jnp.ndarray,
    *,
    epsilon: float = 1e-8,
) -> jnp.ndarray:
    return association_score(
        state_action_embeddings[:, None, :],
        goal_embeddings[None, :, :],
        epsilon=epsilon,
    )


def infonce_loss_from_logits(
    logits: jnp.ndarray, *, logsumexp_penalty: float = 0.1
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Diagonal-positive InfoNCE plus upstream logsumexp regularization."""

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("logits must be a square rank-two array")
    row_logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
    classification = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
    regularizer = logsumexp_penalty * jnp.mean(jnp.square(row_logsumexp))
    return classification + regularizer, {
        "classification_loss": classification,
        "logsumexp_regularizer": regularizer,
        "mean_positive_score": jnp.mean(jnp.diag(logits)),
        "mean_logsumexp": jnp.mean(row_logsumexp),
    }


def contrastive_loss(
    state_action_embeddings: jnp.ndarray,
    goal_embeddings: jnp.ndarray,
    *,
    epsilon: float = 1e-8,
    logsumexp_penalty: float = 0.1,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    logits = pairwise_association_scores(
        state_action_embeddings, goal_embeddings, epsilon=epsilon
    )
    loss, metrics = infonce_loss_from_logits(
        logits, logsumexp_penalty=logsumexp_penalty
    )
    return loss, {**metrics, "logits": logits}


def tanh_gaussian_sample(
    mean: jnp.ndarray, log_std: jnp.ndarray, noise: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a squashed Gaussian and return summed log density."""

    std = jnp.exp(log_std)
    pre_tanh = mean + std * noise
    action = jnp.tanh(pre_tanh)
    log_prob = jax.scipy.stats.norm.logpdf(pre_tanh, loc=mean, scale=std)
    log_prob -= jnp.log(1.0 - jnp.square(action) + 1e-6)
    return action, jnp.sum(log_prob, axis=-1)


def actor_objective(
    score: jnp.ndarray, log_probability: jnp.ndarray, log_alpha: jnp.ndarray
) -> jnp.ndarray:
    """Minimize alpha*log pi - score, so higher association is preferred."""

    return jnp.mean(jnp.exp(log_alpha) * log_probability - score)


def alpha_objective(
    log_alpha: jnp.ndarray,
    log_probability: jnp.ndarray,
    target_entropy: float,
) -> jnp.ndarray:
    return jnp.exp(log_alpha) * jnp.mean(
        jax.lax.stop_gradient(-log_probability - target_entropy)
    )
