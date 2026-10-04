"""Reward-independent measurements of an episode's physical trajectory."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

from .state import VehicleState


@dataclass(frozen=True)
class EpisodeMetrics:
    """Immutable measurements since reset (possibly an unfinished episode).

    Time and signed net energy are taken from the simulator state; regeneration
    may make energy negative. Speed excess is sampled at each step's END using
    the limit at that position. Events are contiguous samples with excess > 0;
    equality with the limit ends an event. The integral is the right-endpoint
    sum of excess_m_s * dt_s, measured in metres.
    No task tolerance or reward component enters these measurements.
    """

    completed: bool
    travel_time_s: float
    energy_kwh: float
    speed_violation_count: int
    max_speed_violation_m_s: float
    integrated_speed_violation_m: float

    def __post_init__(self):
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be a boolean")
        if (
            isinstance(self.speed_violation_count, bool)
            or not isinstance(self.speed_violation_count, Integral)
            or self.speed_violation_count < 0
        ):
            raise ValueError("speed_violation_count must be a nonnegative integer")
        for name in (
            "travel_time_s",
            "energy_kwh",
            "max_speed_violation_m_s",
            "integrated_speed_violation_m",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")
            if name != "energy_kwh" and value < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass
class _EpisodeMetricsAccumulator:
    """Internal streaming speed statistics; never writes simulation state."""

    speed_excess_m_s: float = 0.0
    speed_violation_count: int = 0
    max_speed_violation_m_s: float = 0.0
    integrated_speed_violation_m: float = 0.0

    def update(self, *, velocity_m_s: float, speed_limit_m_s: float, dt_s: float):
        """Observe one completed integration step, not the reset sample."""
        for name, value in (
            ("velocity_m_s", velocity_m_s),
            ("speed_limit_m_s", speed_limit_m_s),
            ("dt_s", dt_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        excess = max(0.0, float(velocity_m_s - speed_limit_m_s))
        integral = self.integrated_speed_violation_m + excess * dt_s
        if not isfinite(excess) or not isfinite(integral):
            raise ValueError("Speed-violation accumulation must remain finite")
        if excess > 0 and self.speed_excess_m_s == 0:
            self.speed_violation_count += 1
        self.speed_excess_m_s = excess
        self.max_speed_violation_m_s = max(self.max_speed_violation_m_s, excess)
        self.integrated_speed_violation_m = integral

    def snapshot(self, state: VehicleState, *, completed: bool) -> EpisodeMetrics:
        """Combine statistics with the existing time/energy accounting."""
        return EpisodeMetrics(
            completed=completed,
            travel_time_s=state.elapsed_time_s,
            energy_kwh=state.total_energy_kwh,
            speed_violation_count=self.speed_violation_count,
            max_speed_violation_m_s=self.max_speed_violation_m_s,
            integrated_speed_violation_m=self.integrated_speed_violation_m,
        )
