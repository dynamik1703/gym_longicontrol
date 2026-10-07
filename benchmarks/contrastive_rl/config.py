"""Configuration for the preparation-only contrastive reference core."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceCoreConfig:
    """Source-aligned network and loss settings.

    ``depth`` counts Dense layers inside residual blocks. The input and output
    projections are deliberately excluded, matching the scaling paper.
    """

    state_dim: int = 12
    goal_dim: int = 3
    action_dim: int = 1
    depth: int = 4
    width: int = 256
    embedding_dim: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    entropy_fraction: float = 0.5
    logsumexp_penalty: float = 0.1
    distance_epsilon: float = 1e-8

    def __post_init__(self):
        for name in ("state_dim", "goal_dim", "action_dim", "width"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.depth <= 0 or self.depth % 4:
            raise ValueError("depth must be a positive multiple of four")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        if self.distance_epsilon <= 0.0:
            raise ValueError("distance_epsilon must be positive")

    @property
    def target_entropy(self) -> float:
        return -self.entropy_fraction * self.action_dim


PLANNED_DEPTHS = (4, 16)
PLANNED_SEEDS = (11, 29, 47)
PLANNED_TRANSITIONS_PER_POLICY = 300_000
PREFILL_TRANSITIONS = 10_000
UPDATE_INTERVAL_TRANSITIONS = 40
PLANNED_COMPLETE_UPDATE_CYCLES = (
    PLANNED_TRANSITIONS_PER_POLICY - PREFILL_TRANSITIONS
) // UPDATE_INTERVAL_TRANSITIONS
DEVELOPMENT_CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
DEVELOPMENT_TRACKS = tuple(range(2000, 2009))
VALIDATION_TRACKS = tuple(range(3000, 3009))
SEALED_PAPER_TRACKS = tuple(range(4000, 4018))
