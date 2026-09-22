"""The original eight-feature observation with consistent declared bounds."""

import numpy as np

from .state import SimulationConfig, VehicleState
from .track import SensorReading
from .vehicle import VehicleSpecs


def observation(
    state: VehicleState,
    sensor: SensorReading,
    specs: VehicleSpecs,
    config: SimulationConfig,
    energy_factor: float,
) -> np.ndarray:
    raw = np.array(
        [
            state.velocity_m_s,
            state.acceleration_m_s2,
            sensor.current_limit_m_s,
            *sensor.future_limits_m_s,
            *sensor.distances_m,
            energy_factor,
        ],
        dtype=np.float64,
    )
    low = np.array([0, specs.acceleration_limits[0], 0, 0, 0, 0, 0, 0])
    high = np.array(
        [
            specs.velocity_limits[1],
            specs.acceleration_limits[1],
            *([specs.velocity_limits[1]] * 3),
            config.sensor_range_m,
            config.sensor_range_m,
            5.0,
        ]
    )
    return (raw - low) / (high - low)
