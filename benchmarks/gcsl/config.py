"""Frozen scientific and execution configuration for LongiControl GCSL V1."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GCSLConfig:
    """The single preregistered policy condition.

    ``depth`` follows the projected-goal CRL convention: it counts Dense
    layers inside residual blocks and excludes input/output projections.
    """

    state_dim: int = 12
    goal_dim: int = 3
    action_dim: int = 1
    width: int = 256
    depth: int = 4
    learning_rate: float = 5e-4
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_epsilon: float = 1e-6
    future_discount: float = 0.99
    batch_size: int = 256
    replay_capacity: int = 300_000
    horizon_conditioning: bool = False

    def __post_init__(self) -> None:
        for name in ("state_dim", "goal_dim", "action_dim", "width", "depth"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.depth % 4:
            raise ValueError("depth must be a positive multiple of four")
        if not 0.0 < self.future_discount < 1.0:
            raise ValueError("future_discount must lie strictly between zero and one")
        if not 0.0 < self.action_epsilon < 0.01:
            raise ValueError("action_epsilon must be small and positive")
        if self.horizon_conditioning:
            raise ValueError("GCSL V1 freezes horizon conditioning off")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


PLANNED_SEEDS = (11, 29, 47)
PLANNED_TRANSITIONS_PER_POLICY = 300_000
PREFILL_TRANSITIONS = 10_000
UPDATE_INTERVAL_TRANSITIONS = 40
PLANNED_UPDATE_CYCLES = (
    PLANNED_TRANSITIONS_PER_POLICY - PREFILL_TRANSITIONS
) // UPDATE_INTERVAL_TRANSITIONS
DEVELOPMENT_CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
DEVELOPMENT_TRACKS = tuple(range(2000, 2009))
VALIDATION_TRACKS = tuple(range(3000, 3009))
SEALED_PAPER_TRACKS = tuple(range(4000, 4018))
CANONICAL_GOAL = (1.0, 1.0, 1.0)
NATIVE_DT_S = 0.1


def should_update(transition_count: int) -> bool:
    return bool(
        transition_count > PREFILL_TRANSITIONS
        and (transition_count - PREFILL_TRANSITIONS)
        % UPDATE_INTERVAL_TRANSITIONS
        == 0
        and transition_count <= PLANNED_TRANSITIONS_PER_POLICY
    )


def scheduled_update_transitions() -> tuple[int, ...]:
    return tuple(
        range(
            PREFILL_TRANSITIONS + UPDATE_INTERVAL_TRANSITIONS,
            PLANNED_TRANSITIONS_PER_POLICY + 1,
            UPDATE_INTERVAL_TRANSITIONS,
        )
    )
