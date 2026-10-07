"""Pure LongiControl outcome semantics; no learner or live evaluator state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from gym_longicontrol.domain.task import TaskSpecification

CURRENT_POSITION = 0
PREVIOUS_POSITION = 1
ABSOLUTE_TIME = 2
CUMULATIVE_MAX_VIOLATION = 3
OUTCOME_DIM = 4


@dataclass(frozen=True)
class OutcomeScales:
    route_length_m: float = 1000.0
    horizon_s: float = 180.0
    speed_scale_m_s: float = 37.0

    def __post_init__(self):
        for name in ("route_length_m", "horizon_s", "speed_scale_m_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def physical_outcome(
    *,
    position_m: float,
    previous_position_m: float,
    elapsed_time_s: float,
    max_speed_violation_m_s: float,
) -> np.ndarray:
    """Create the exact achieved outcome required for first-arrival semantics."""

    values = np.asarray(
        (
            position_m,
            previous_position_m,
            elapsed_time_s,
            max_speed_violation_m_s,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("physical outcome values must be finite and nonnegative")
    if previous_position_m > position_m:
        raise ValueError("previous position cannot exceed current position")
    return values


def normalize_outcome(outcome, scales: OutcomeScales) -> np.ndarray:
    """Linearly scale without clipping, preserving crossing and violations."""

    array = np.asarray(outcome, dtype=np.float64)
    if array.shape[-1:] != (OUTCOME_DIM,) or not np.isfinite(array).all():
        raise ValueError("outcome must be finite with final dimension four")
    return array / np.asarray(
        (
            scales.route_length_m,
            scales.route_length_m,
            scales.horizon_s,
            scales.speed_scale_m_s,
        ),
        dtype=np.float64,
    )


def augment_policy_state(observation, outcome, scales: OutcomeScales) -> np.ndarray:
    """Retain sensor observation and append only legitimate task state."""

    observation_array = np.asarray(observation, dtype=np.float64)
    if observation_array.ndim != 1 or not np.isfinite(observation_array).all():
        raise ValueError("observation must be a finite vector")
    return np.concatenate((observation_array, normalize_outcome(outcome, scales)))


def canonical_goal_set_membership(
    outcome,
    *,
    task: TaskSpecification,
    route_length_m: float = 1000.0,
):
    """Exact membership in the canonical first-arrival requirement set.

    Absolute episode time is compared by inequality. The cumulative maximum
    violation makes any previous unsafe step permanently invalidating.
    """

    array = np.asarray(outcome, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != OUTCOME_DIM:
        raise ValueError("outcome must have final dimension four")
    if not np.isfinite(array).all():
        raise ValueError("outcome must be finite")
    membership = (
        (array[..., PREVIOUS_POSITION] < route_length_m)
        & (array[..., CURRENT_POSITION] >= route_length_m)
        & (array[..., ABSOLUTE_TIME] <= task.max_time_s)
        & (
            array[..., CUMULATIVE_MAX_VIOLATION]
            <= task.max_speed_violation_m_s
        )
    )
    if membership.ndim == 0:
        return bool(membership)
    return membership
