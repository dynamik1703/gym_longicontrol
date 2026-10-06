"""Pure goal encoding, success, reward, and relabeling semantics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from gym_longicontrol.domain.task import TaskSpecification

CURRENT_POSITION = 0
PREVIOUS_POSITION = 1
ELAPSED_TIME = 2
MAX_SPEED_VIOLATION = 3
GOAL_SIZE = 4


@dataclass(frozen=True)
class GoalScales:
    """Fixed physical scales used to map goal quantities into [0, 1]."""

    route_length_m: float = 1000.0
    horizon_s: float = 180.0
    speed_scale_m_s: float = 37.0

    def __post_init__(self):
        for name in ("route_length_m", "horizon_s", "speed_scale_m_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _goal_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != GOAL_SIZE:
        raise ValueError(f"{name} must have final dimension {GOAL_SIZE}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if (array < 0.0).any() or (array > 1.0).any():
        raise ValueError(f"{name} must lie in [0, 1]")
    return array


def encode_achieved_goal(
    *,
    position_m: float,
    previous_position_m: float,
    elapsed_time_s: float,
    max_speed_violation_m_s: float,
    scales: GoalScales,
) -> np.ndarray:
    """Encode the stored transition state needed for first-arrival success.

    The violation value is cumulative since reset. It is deliberately not the
    current speed excess, so slowing down cannot erase an earlier violation.
    """

    values = (
        position_m,
        previous_position_m,
        elapsed_time_s,
        max_speed_violation_m_s,
    )
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("Achieved-goal quantities must be finite and nonnegative")
    return np.array(
        [
            np.clip(position_m / scales.route_length_m, 0.0, 1.0),
            np.clip(previous_position_m / scales.route_length_m, 0.0, 1.0),
            np.clip(elapsed_time_s / scales.horizon_s, 0.0, 1.0),
            np.clip(
                max_speed_violation_m_s / scales.speed_scale_m_s, 0.0, 1.0
            ),
        ],
        dtype=np.float64,
    )


def desired_goal(
    *,
    target_position_m: float,
    max_time_s: float,
    max_speed_violation_m_s: float,
    scales: GoalScales,
) -> np.ndarray:
    """Encode a first-arrival task.

    The target position is intentionally duplicated. Slots zero and one are
    respectively compared with current and previous position, expressing the
    crossing ``previous < target <= current`` without live environment state.
    """

    values = (target_position_m, max_time_s, max_speed_violation_m_s)
    if any(not isfinite(value) or value < 0 for value in values):
        raise ValueError("Desired-goal quantities must be finite and nonnegative")
    if target_position_m > scales.route_length_m:
        raise ValueError("Target position exceeds the original route")
    if max_time_s > scales.horizon_s:
        raise ValueError("Goal deadline exceeds the finite task horizon")
    if max_speed_violation_m_s > scales.speed_scale_m_s:
        raise ValueError("Speed tolerance exceeds its normalization scale")
    target = target_position_m / scales.route_length_m
    return np.array(
        [
            target,
            target,
            max_time_s / scales.horizon_s,
            max_speed_violation_m_s / scales.speed_scale_m_s,
        ],
        dtype=np.float64,
    )


def canonical_desired_goal(
    task: TaskSpecification, scales: GoalScales
) -> np.ndarray:
    """Return the original-route task used by every real rollout."""

    return desired_goal(
        target_position_m=scales.route_length_m,
        max_time_s=task.max_time_s,
        max_speed_violation_m_s=task.max_speed_violation_m_s,
        scales=scales,
    )


def goal_success(achieved_goal, desired_goal_value):
    """Evaluate strict first arrival from stored, potentially batched goals."""

    achieved = _goal_array(achieved_goal, name="achieved_goal")
    desired = _goal_array(desired_goal_value, name="desired_goal")
    try:
        achieved, desired = np.broadcast_arrays(achieved, desired)
    except ValueError as error:
        raise ValueError("Goal arrays are not broadcast-compatible") from error
    if not np.array_equal(
        desired[..., CURRENT_POSITION], desired[..., PREVIOUS_POSITION]
    ):
        raise ValueError("Desired goal must duplicate its target position")
    if (achieved[..., CURRENT_POSITION] < achieved[..., PREVIOUS_POSITION]).any():
        raise ValueError("Achieved position cannot move behind previous position")
    success = (
        (achieved[..., CURRENT_POSITION] >= desired[..., CURRENT_POSITION])
        & (achieved[..., PREVIOUS_POSITION] < desired[..., PREVIOUS_POSITION])
        & (achieved[..., ELAPSED_TIME] <= desired[..., ELAPSED_TIME])
        & (
            achieved[..., MAX_SPEED_VIOLATION]
            <= desired[..., MAX_SPEED_VIOLATION]
        )
    )
    if success.ndim == 0:
        return bool(success)
    return success


def goal_transition_reward(achieved_goal, desired_goal_value):
    """Return exactly 1 on a valid first-arrival transition, otherwise 0."""

    success = goal_success(achieved_goal, desired_goal_value)
    if isinstance(success, bool):
        return float(success)
    return np.asarray(success, dtype=np.float32)


def relabel_desired_goal(original_desired_goal, future_achieved_goal):
    """Relabel only route position; keep deadline and tolerance unchanged."""

    original = _goal_array(original_desired_goal, name="original_desired_goal")
    future = _goal_array(future_achieved_goal, name="future_achieved_goal")
    try:
        original, future = np.broadcast_arrays(original, future)
    except ValueError as error:
        raise ValueError("Goal arrays are not broadcast-compatible") from error
    relabeled = np.array(original, dtype=np.float64, copy=True)
    relabeled[..., CURRENT_POSITION] = future[..., CURRENT_POSITION]
    relabeled[..., PREVIOUS_POSITION] = future[..., CURRENT_POSITION]
    return relabeled


def relabeled_terminal_mask(original_done, achieved_goal, desired_goal_value):
    """End a virtual task on first arrival or an original finite-horizon end."""

    done = np.asarray(original_done, dtype=bool)
    success = np.asarray(goal_success(achieved_goal, desired_goal_value), dtype=bool)
    try:
        result = np.logical_or(done, success)
    except ValueError as error:
        raise ValueError(
            "Done and goal batches are not broadcast-compatible"
        ) from error
    if result.ndim == 0:
        return bool(result)
    return result
