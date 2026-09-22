"""Explicit units and immutable configuration for the simulator."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SimulationConfig:
    track_length_m: float = 1000.0
    dt_s: float = 0.1
    sensor_range_m: float = 150.0

    def __post_init__(self):
        for name in ("track_length_m", "dt_s", "sensor_range_m"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class VehicleState:
    position_m: float = 0.0
    velocity_m_s: float = 0.0
    acceleration_m_s2: float = 0.0
    jerk_m_s3: float = 0.0
    elapsed_time_s: float = 0.0
    total_energy_kwh: float = 0.0
