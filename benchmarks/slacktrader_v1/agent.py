"""SlackTrader actor/critic identity built on corrected Hybrid SMDP SAC."""

from __future__ import annotations

from benchmarks.hybrid_rl_execution.agent import (
    ACTION_DIM,
    HIDDEN_SIZES,
    OBSERVATION_DIM,
    AgentConfig,
    ConstrainedSAC,
    Critic,
    GaussianActor,
)


class SlackTraderSAC(ConstrainedSAC):
    """One-scalar SAC whose normalized action denotes λ, never native actuation."""


__all__ = [
    "ACTION_DIM",
    "HIDDEN_SIZES",
    "OBSERVATION_DIM",
    "AgentConfig",
    "Critic",
    "GaussianActor",
    "SlackTraderSAC",
]
