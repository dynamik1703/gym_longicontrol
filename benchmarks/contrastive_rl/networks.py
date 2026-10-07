"""JAX/Flax networks derived from the pinned Scaling-CRL implementation.

Modified for a simulator-independent core: dimensions are explicit and Brax,
W&B, environment wrappers, and command-line concerns are absent. Architecture,
initialization family, normalization, activation, and depth convention follow
upstream ``train.py`` at the revision recorded in ``upstream.json``.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import variance_scaling

_KERNEL_INIT = variance_scaling(1.0 / 3.0, "fan_in", "uniform")
_BIAS_INIT = nn.initializers.zeros


class ResidualBlock(nn.Module):
    """Exactly four Dense-LayerNorm-Swish operations and one residual add."""

    width: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        hidden = inputs
        for _ in range(4):
            hidden = nn.Dense(
                self.width, kernel_init=_KERNEL_INIT, bias_init=_BIAS_INIT
            )(hidden)
            hidden = nn.LayerNorm()(hidden)
            hidden = nn.swish(hidden)
        return hidden + inputs


class ResidualTrunk(nn.Module):
    """Input projection plus residual blocks; ``depth`` excludes the input."""

    width: int
    depth: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        if self.depth <= 0 or self.depth % 4:
            raise ValueError("depth must be a positive multiple of four")
        hidden = nn.Dense(
            self.width, kernel_init=_KERNEL_INIT, bias_init=_BIAS_INIT
        )(inputs)
        hidden = nn.LayerNorm()(hidden)
        hidden = nn.swish(hidden)
        for _ in range(self.depth // 4):
            hidden = ResidualBlock(self.width)(hidden)
        return hidden


class StateActionEncoder(nn.Module):
    width: int = 256
    depth: int = 4
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        hidden = ResidualTrunk(self.width, self.depth)(
            jnp.concatenate((state, action), axis=-1)
        )
        return nn.Dense(
            self.embedding_dim, kernel_init=_KERNEL_INIT, bias_init=_BIAS_INIT
        )(hidden)


class GoalEncoder(nn.Module):
    width: int = 256
    depth: int = 4
    embedding_dim: int = 64

    @nn.compact
    def __call__(self, goal: jnp.ndarray) -> jnp.ndarray:
        hidden = ResidualTrunk(self.width, self.depth)(goal)
        return nn.Dense(
            self.embedding_dim, kernel_init=_KERNEL_INIT, bias_init=_BIAS_INIT
        )(hidden)


class Actor(nn.Module):
    action_dim: int
    width: int = 256
    depth: int = 4
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, state_goal: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        hidden = ResidualTrunk(self.width, self.depth)(state_goal)
        mean = nn.Dense(
            self.action_dim, kernel_init=_KERNEL_INIT, bias_init=_BIAS_INIT
        )(hidden)
        log_std = nn.Dense(
            self.action_dim, kernel_init=_KERNEL_INIT, bias_init=_BIAS_INIT
        )(hidden)
        log_std = nn.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1.0)
        return mean, log_std
