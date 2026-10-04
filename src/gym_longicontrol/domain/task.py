"""Operational requirements, independent of simulator rewards and algorithms."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .metrics import EpisodeMetrics


@dataclass(frozen=True)
class TaskSpecification:
    """Complete the route within these bounds while minimizing net energy.

    The speed tolerance bounds the maximum instantaneous excess, not its
    duration or event count. This specification does not configure the simulator
    or change the environment's reward, observations, or termination rules.
    """

    max_time_s: float
    max_speed_violation_m_s: float = 0.0

    def __post_init__(self):
        for name in ("max_time_s", "max_speed_violation_m_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")
        if self.max_time_s <= 0:
            raise ValueError("max_time_s must be positive")
        if self.max_speed_violation_m_s < 0:
            raise ValueError("max_speed_violation_m_s must be nonnegative")


def is_feasible(metrics: EpisodeMetrics, task: TaskSpecification) -> bool:
    """Check physical requirements with inclusive bounds and no hidden epsilon.

    Comparisons use the recorded floats exactly. Even a representable value
    immediately above a bound fails. Reward and energy do not affect feasibility;
    energy is the quantity to minimize among feasible completed episodes.
    """
    return bool(
        metrics.completed
        and metrics.travel_time_s <= task.max_time_s
        and metrics.max_speed_violation_m_s <= task.max_speed_violation_m_s
    )
