"""Deterministic decoding and lifecycle for residual actor actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NATIVE_STEPS_PER_SECOND = 10
RESIDUAL_SCALE = 0.25


@dataclass(frozen=True)
class MethodSpec:
    name: str
    action_dim: int
    actor_interval_native_steps: int
    target_entropy: float

    def __post_init__(self) -> None:
        if self.action_dim not in (1, 3):
            raise ValueError("V1 supports only one- or three-dimensional actions")
        if self.actor_interval_native_steps != self.action_dim * 10:
            raise ValueError("each residual component must last exactly one second")
        if self.target_entropy != -float(self.action_dim):
            raise ValueError("target entropy must equal minus action dimension")


METHODS = {
    "CHUNK3_RESIDUAL_SAC": MethodSpec("CHUNK3_RESIDUAL_SAC", 3, 30, -3.0),
    "RESIDUAL1S_SAC": MethodSpec("RESIDUAL1S_SAC", 1, 10, -1.0),
}


def method_spec(name: str) -> MethodSpec:
    try:
        return METHODS[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown feedback-chunk method: {name}") from error


def validate_actor_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64)
    if value.shape != (action_dim,) or not np.isfinite(value).all():
        raise ValueError(f"actor action must be a finite {action_dim}-vector")
    if np.any(value < -1.0) or np.any(value > 1.0):
        raise ValueError("actor action must be in [-1, 1]")
    return value.copy()


@dataclass
class ResidualChunk:
    action: np.ndarray
    start_native_step: int
    chunk_id: int
    cancelled: bool = False
    cancellation_reason: str | None = None
    cancellation_native_step: int | None = None
    residual_influence_native_steps: int = 0

    def __post_init__(self) -> None:
        self.action = validate_actor_action(self.action, int(self.action.size))
        if self.start_native_step < 0 or self.chunk_id < 0:
            raise ValueError("chunk indices must be nonnegative")

    @property
    def action_dim(self) -> int:
        return int(self.action.size)

    @property
    def duration_native_steps(self) -> int:
        return self.action_dim * NATIVE_STEPS_PER_SECOND

    def phase(self, native_step: int) -> int:
        offset = int(native_step) - self.start_native_step
        if not 0 <= offset < self.duration_native_steps:
            raise RuntimeError("native step is outside the active chunk")
        return offset // NATIVE_STEPS_PER_SECOND

    def correction(self, native_step: int) -> float:
        return float(self.action[self.phase(native_step)])

    def cancel(self, *, native_step: int, reason: str) -> None:
        if not self.cancelled:
            self.cancelled = True
            self.cancellation_reason = str(reason)
            self.cancellation_native_step = int(native_step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.copy(),
            "start_native_step": self.start_native_step,
            "chunk_id": self.chunk_id,
            "cancelled": self.cancelled,
            "cancellation_reason": self.cancellation_reason,
            "cancellation_native_step": self.cancellation_native_step,
            "residual_influence_native_steps": self.residual_influence_native_steps,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ResidualChunk:
        result = cls(
            action=np.asarray(state["action"], dtype=np.float64),
            start_native_step=int(state["start_native_step"]),
            chunk_id=int(state["chunk_id"]),
        )
        result.cancelled = bool(state["cancelled"])
        result.cancellation_reason = state["cancellation_reason"]
        cancelled_at = state["cancellation_native_step"]
        result.cancellation_native_step = (
            None if cancelled_at is None else int(cancelled_at)
        )
        result.residual_influence_native_steps = int(
            state["residual_influence_native_steps"]
        )
        return result


def combined_proposal(mpc_action: float, residual: float) -> float:
    values = (float(mpc_action), float(residual))
    if not all(np.isfinite(value) for value in values):
        raise ValueError("proposal inputs must be finite")
    if not -1.0 <= residual <= 1.0:
        raise ValueError("residual must be in [-1, 1]")
    return float(np.clip(mpc_action + RESIDUAL_SCALE * residual, -1.0, 1.0))
