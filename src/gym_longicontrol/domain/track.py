"""Track generation and look-ahead sensing, using SI units internally."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorReading:
    current_limit_m_s: float
    future_limits_m_s: tuple[float, float]
    distances_m: tuple[float, float]


@dataclass(frozen=True)
class Track:
    positions_m: np.ndarray
    limits_m_s: np.ndarray

    def __post_init__(self):
        positions = np.array(self.positions_m, dtype=np.float64, copy=True)
        limits = np.array(self.limits_m_s, dtype=np.float64, copy=True)
        if (
            positions.ndim != 1
            or not len(positions)
            or limits.shape != positions.shape
            or not np.isfinite(positions).all()
            or not np.isfinite(limits).all()
            or positions[0] != 0
            or not (np.diff(positions) > 0).all()
            or not ((limits > 0) & (limits <= 37)).all()
        ):
            raise ValueError(
                "Track arrays must match, start at 0 m, have strictly increasing "
                "finite positions and speed limits in (0, 37] m/s"
            )
        positions.setflags(write=False)
        limits.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "limits_m_s", limits)

    def sense(self, position_m: float, sensor_range_m: float) -> SensorReading:
        index = max(
            0, int(np.searchsorted(self.positions_m, position_m, side="right")) - 1
        )
        current = float(self.limits_m_s[index])
        future, distances = [], []
        visible_limit = current
        for offset in (1, 2):
            next_index = index + offset
            distance = sensor_range_m
            if next_index < len(self.positions_m):
                ahead = float(self.positions_m[next_index] - position_m)
                if ahead <= sensor_range_m:
                    visible_limit = float(self.limits_m_s[next_index])
                    distance = ahead
            future.append(visible_limit)
            distances.append(distance)
        return SensorReading(current, tuple(future), tuple(distances))


class FixedTrackGenerator:
    def __init__(
        self,
        positions_km=(0.0, 0.25, 0.5, 0.75),
        limits_km_h=(50, 80, 40, 50),
        track_length_m=1000.0,
    ):
        # Retain legacy input quantization to keep existing deterministic rollouts.
        self.track = Track(
            np.asarray(positions_km, dtype=np.float32) * 1000.0,
            np.asarray(limits_km_h, dtype=np.float32) / 3.6,
        )
        if self.track.positions_m[-1] >= track_length_m:
            raise ValueError("Speed-limit positions must be before the track finish")

    def __call__(self, rng: np.random.Generator) -> Track:
        return self.track


class StochasticTrackGenerator:
    def __init__(self, track_length_m=1000.0):
        self.track_length_m = track_length_m

    def __call__(self, rng: np.random.Generator) -> Track:
        possible = np.arange(100.0, self.track_length_m, 100.0)
        count = min(len(possible), max(int(rng.integers(max(len(possible) + 1, 1))), 1))
        positions = np.r_[0.0, np.sort(rng.choice(possible, count, replace=False))]
        limits = []
        previous_km_h, previous_m_s = 40, 20.0 / 3.6
        for _ in positions:
            choices = (
                np.arange(
                    max(20, previous_km_h - 40), min(100, previous_km_h + 40) + 1, 10
                )
                / 3.6
            )
            choices = np.setdiff1d(choices, previous_m_s)
            previous_m_s = float(rng.choice(choices))
            previous_km_h = int(previous_m_s * 3.6)
            limits.append(previous_m_s)
        return Track(positions, np.array(limits))
