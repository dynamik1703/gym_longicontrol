"""Pure reward calculation; energy accumulation belongs to the dynamics."""

from .state import SimulationConfig, VehicleState
from .track import SensorReading
from .vehicle import VehicleSpecs


def reward_components(
    state: VehicleState,
    sensor: SensorReading,
    power_kw: float,
    specs: VehicleSpecs,
    config: SimulationConfig,
) -> dict[str, float]:
    return {
        "forward": -abs(state.velocity_m_s - sensor.current_limit_m_s)
        / sensor.current_limit_m_s,
        "energy": -power_kw / specs.power_limits_kw[1],
        "jerk": -state.jerk_m_s3
        * config.dt_s
        / (specs.acceleration_limits[1] - specs.acceleration_limits[0]),
        "shock": -float(state.velocity_m_s > sensor.current_limit_m_s),
    }
