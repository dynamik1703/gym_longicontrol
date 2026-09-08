"""Constant-acceleration integration and energy accounting."""

from .state import SimulationConfig, VehicleState
from .vehicle import VehicleModel


def advance(
    state: VehicleState, action: float, vehicle: VehicleModel, config: SimulationConfig
) -> tuple[VehicleState, float, float]:
    dt = config.dt_s
    velocity = state.velocity_m_s
    acceleration = vehicle.acceleration_from_action(velocity, action)
    low, high = vehicle.specs.velocity_limits
    # Resolve within-step velocity boundary crossings without changing dt.
    acceleration = min(max(acceleration, (low - velocity) / dt), (high - velocity) / dt)
    new_velocity = velocity + acceleration * dt
    power = vehicle.power_kw(new_velocity, acceleration)
    energy = power * dt / 3600.0
    return (
        VehicleState(
            position_m=state.position_m + velocity * dt + 0.5 * acceleration * dt**2,
            velocity_m_s=new_velocity,
            acceleration_m_s2=acceleration,
            jerk_m_s3=abs(acceleration - state.acceleration_m_s2) / dt,
            elapsed_time_s=state.elapsed_time_s + dt,
            total_energy_kwh=state.total_energy_kwh + energy,
        ),
        power,
        energy,
    )
