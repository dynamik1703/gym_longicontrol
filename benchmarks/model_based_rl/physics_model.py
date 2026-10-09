"""Oracle-dynamics model that delegates to the simulator's exact equations."""

from __future__ import annotations

import numpy as np

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig
from gym_longicontrol.domain.vehicle import VehicleModel

from .model_state import ModelState, VehiclePrediction


class PhysicsDynamicsModel:
    """Pure one-step adapter over ``domain.dynamics.advance``."""

    condition = "physics"
    version = 1

    def __init__(self, vehicle: VehicleModel, config: SimulationConfig):
        self.vehicle = vehicle
        self.config = config

    def predict(
        self, state: ModelState, action: float, *, rng: np.random.Generator
    ) -> VehiclePrediction:
        del rng
        next_vehicle, _power_kw, energy = advance(
            state.vehicle, float(action), self.vehicle, self.config
        )
        return VehiclePrediction(
            next_vehicle=next_vehicle,
            signed_step_energy_kwh=float(energy),
        )
