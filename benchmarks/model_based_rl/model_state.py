"""Benchmark-internal Markov state; never supplied to the policy actor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.track import Track


@dataclass(frozen=True)
class ModelState:
    """Minimal state needed for one physical transition and metric history."""

    vehicle: VehicleState
    maximum_speed_excess_m_s: float = 0.0
    speed_violation_count: int = 0
    violation_active: bool = False
    integrated_speed_violation_m: float = 0.0
    episode_step: int = 0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.vehicle.position_m,
                self.vehicle.velocity_m_s,
                self.vehicle.acceleration_m_s2,
                self.vehicle.jerk_m_s3,
                self.vehicle.elapsed_time_s,
                self.vehicle.total_energy_kwh,
                self.maximum_speed_excess_m_s,
                self.integrated_speed_violation_m,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values[[0, 1, 4, 6, 7]] < 0):
            raise ValueError("ModelState contains invalid physical quantities")
        if self.speed_violation_count < 0 or self.episode_step < 0:
            raise ValueError("ModelState counters must be nonnegative")

    def model_input(self, action: float) -> np.ndarray:
        """Learned vehicle input; exogenous track state is deliberately absent."""

        vector = np.asarray(
            [
                self.vehicle.position_m,
                self.vehicle.velocity_m_s,
                self.vehicle.acceleration_m_s2,
                float(action),
            ],
            dtype=np.float64,
        )
        if not np.isfinite(vector).all() or not -1 <= vector[-1] <= 1:
            raise ValueError("action must be finite and in [-1, 1]")
        return vector


@dataclass(frozen=True)
class TrackContext:
    """Known exogenous route map shared by both model conditions."""

    track: Track
    sensor_range_m: float
    track_length_m: float
    energy_factor: float


@dataclass(frozen=True)
class VehiclePrediction:
    """Model-dependent quantities required by the deterministic projection."""

    next_vehicle: VehicleState
    signed_step_energy_kwh: float
    ensemble_member: int | None = None
    predictive_variance: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ProjectedTransition:
    """Complete synthetic transition with explicit source provenance."""

    state: ModelState
    action: float
    next_state: ModelState
    observation: np.ndarray
    next_observation: np.ndarray
    objective: float
    costs: tuple[float, float]
    terminated: bool
    truncated: bool
    source: str
    model_condition: str
    source_real_transition_id: int
    model_version: int
    ensemble_member: int | None
    predictive_variance: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if self.source not in {"real", "model"}:
            raise ValueError("Transition source must be real or model")
        if self.model_condition not in {"learned", "physics", "none"}:
            raise ValueError("Unknown model condition")
        if self.source_real_transition_id < 0 or self.model_version < 0:
            raise ValueError("Provenance counters must be nonnegative")
        if self.observation.shape != (8,) or self.next_observation.shape != (8,):
            raise ValueError("Policy observations must retain the public 8D shape")
