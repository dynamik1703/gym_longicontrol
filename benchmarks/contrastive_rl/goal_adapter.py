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
PROJECTED_PROGRESS = 0
PROJECTED_WITHIN_DEADLINE = 1
PROJECTED_COMPLIANT_SO_FAR = 2
PROJECTED_GOAL_DIM = 3
CANONICAL_ROUTE_LENGTH_M = 1000.0
CANONICAL_MAX_TIME_S = 140.0
CANONICAL_MAX_SPEED_VIOLATION_M_S = 0.0


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


def canonical_command() -> np.ndarray:
    """Return the fixed requirement-aware command, independent of replay."""

    return np.ones(PROJECTED_GOAL_DIM, dtype=np.float64)


def _validated_transition_outcomes(
    outcome,
    *,
    terminated,
    route_length_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the original monotone, first-crossing transition domain.

    Each row represents a transition from ``previous_position`` to
    ``current_position``. Sources must precede route termination, movement is
    monotone, and physical termination must agree exactly with route crossing.
    This rejects post-terminal rows rather than letting saturation create a
    second apparent success.
    """

    array = np.asarray(outcome, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != OUTCOME_DIM:
        raise ValueError("outcome must have final dimension four")
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError("outcome must contain finite nonnegative physical values")
    terminal = np.asarray(terminated)
    if terminal.dtype != np.bool_:
        raise ValueError("terminated must be boolean")
    try:
        terminal = np.broadcast_to(terminal, array.shape[:-1])
    except ValueError as error:
        raise ValueError(
            "terminated is not broadcast-compatible with outcome"
        ) from error
    if not isfinite(route_length_m) or route_length_m <= 0.0:
        raise ValueError("route_length_m must be finite and positive")
    if (array[..., PREVIOUS_POSITION] >= route_length_m).any():
        raise ValueError("post-terminal source transitions are not valid CRL data")
    if (
        array[..., CURRENT_POSITION] < array[..., PREVIOUS_POSITION]
    ).any():
        raise ValueError("the original route transition must be monotone")
    route_crossed = array[..., CURRENT_POSITION] >= route_length_m
    if not np.array_equal(terminal, route_crossed):
        raise ValueError("terminated must agree with the first route crossing")
    return array, terminal


def project_outcome(
    outcome,
    *,
    terminated,
    task: TaskSpecification,
    route_length_m: float = CANONICAL_ROUTE_LENGTH_M,
) -> np.ndarray:
    """Project recorded raw outcomes into the fixed three-entry task goal.

    Requirement predicates are evaluated in float64 physical units before the
    result is handed to a network. Position saturation is deliberate task
    abstraction; no raw replay/provenance value is changed.
    """

    if (
        task.max_time_s != CANONICAL_MAX_TIME_S
        or task.max_speed_violation_m_s
        != CANONICAL_MAX_SPEED_VIOLATION_M_S
        or route_length_m != CANONICAL_ROUTE_LENGTH_M
    ):
        raise ValueError("projected-goal CRL is defined only for the canonical task")
    array, _ = _validated_transition_outcomes(
        outcome,
        terminated=terminated,
        route_length_m=route_length_m,
    )
    progress = np.minimum(array[..., CURRENT_POSITION] / route_length_m, 1.0)
    within_deadline = array[..., ABSOLUTE_TIME] <= task.max_time_s
    compliant_so_far = (
        array[..., CUMULATIVE_MAX_VIOLATION]
        <= task.max_speed_violation_m_s
    )
    return np.stack(
        (
            progress,
            within_deadline.astype(np.float64),
            compliant_so_far.astype(np.float64),
        ),
        axis=-1,
    )


def projected_goal_is_canonical(projected_goal):
    """Exact equality is valid because all requirement entries are binary."""

    array = np.asarray(projected_goal, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != PROJECTED_GOAL_DIM:
        raise ValueError("projected_goal must have final dimension three")
    result = np.all(array == canonical_command(), axis=-1)
    if result.ndim == 0:
        return bool(result)
    return result


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
