"""Shared deterministic projection from model physics to frozen V2 signals."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol

import numpy as np

from benchmarks.constrained_rl.costs import objective_reward, speed_cost
from benchmarks.constrained_rl_v2.costs import deadline_deficit_cost, deadline_state
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import SimulationConfig
from gym_longicontrol.domain.vehicle import VehicleModel

from .model_state import (
    ModelState,
    ProjectedTransition,
    TrackContext,
    VehiclePrediction,
)


class OneStepModel(Protocol):
    condition: str
    version: int

    def predict(
        self, state: ModelState, action: float, *, rng: np.random.Generator
    ) -> VehiclePrediction: ...


def policy_observation(
    state: ModelState,
    context: TrackContext,
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> np.ndarray:
    sensor = context.track.sense(state.vehicle.position_m, context.sensor_range_m)
    return observation(
        state.vehicle, sensor, vehicle.specs, config, context.energy_factor
    )


def project_prediction(
    *,
    state: ModelState,
    action: float,
    prediction: VehiclePrediction,
    context: TrackContext,
    vehicle: VehicleModel,
    config: SimulationConfig,
    energy_scale_kwh: float,
    deadline_s: float,
    max_episode_steps: int,
    source_real_transition_id: int,
    model_condition: str,
    model_version: int,
) -> ProjectedTransition:
    """Apply track lookup, task signals, metric history, and terminal semantics."""

    next_vehicle = prediction.next_vehicle
    # Learned outputs may contain tiny domain excursions. The V1 definition clips
    # only physical state domains, never speed limits, costs, or safety margins.
    next_vehicle = replace(
        next_vehicle,
        position_m=max(0.0, next_vehicle.position_m),
        velocity_m_s=float(
            np.clip(next_vehicle.velocity_m_s, *vehicle.specs.velocity_limits)
        ),
        elapsed_time_s=state.vehicle.elapsed_time_s + config.dt_s,
        total_energy_kwh=(
            state.vehicle.total_energy_kwh + prediction.signed_step_energy_kwh
        ),
    )
    sensor = context.track.sense(next_vehicle.position_m, context.sensor_range_m)
    excess = max(0.0, next_vehicle.velocity_m_s - sensor.current_limit_m_s)
    violating = excess > 0.0
    next_state = ModelState(
        vehicle=next_vehicle,
        maximum_speed_excess_m_s=max(state.maximum_speed_excess_m_s, excess),
        speed_violation_count=(
            state.speed_violation_count + int(violating and not state.violation_active)
        ),
        violation_active=violating,
        integrated_speed_violation_m=(
            state.integrated_speed_violation_m + excess * config.dt_s
        ),
        episode_step=state.episode_step + 1,
    )
    _remaining, _slack, deficit = deadline_state(
        elapsed_time_s=next_vehicle.elapsed_time_s,
        position_m=next_vehicle.position_m,
        track=context.track,
        track_length_m=context.track_length_m,
        deadline_s=deadline_s,
    )
    terminated = next_vehicle.position_m >= context.track_length_m
    truncated = next_state.episode_step >= max_episode_steps and not terminated
    return ProjectedTransition(
        state=state,
        action=float(action),
        next_state=next_state,
        observation=policy_observation(state, context, vehicle, config),
        next_observation=policy_observation(next_state, context, vehicle, config),
        objective=objective_reward(
            prediction.signed_step_energy_kwh, energy_scale_kwh
        ),
        costs=(
            speed_cost(excess, config.dt_s),
            deadline_deficit_cost(
                deficit_s=deficit,
                dt_s=config.dt_s,
                normalization_s=deadline_s,
            ),
        ),
        terminated=terminated,
        truncated=truncated,
        source="model",
        model_condition=model_condition,
        source_real_transition_id=source_real_transition_id,
        model_version=model_version,
        ensemble_member=prediction.ensemble_member,
        predictive_variance=prediction.predictive_variance,
    )


def model_step(
    model: OneStepModel,
    state: ModelState,
    action: float,
    *,
    context: TrackContext,
    vehicle: VehicleModel,
    config: SimulationConfig,
    energy_scale_kwh: float,
    deadline_s: float,
    max_episode_steps: int,
    source_real_transition_id: int,
    rng: np.random.Generator,
) -> ProjectedTransition:
    prediction = model.predict(state, action, rng=rng)
    return project_prediction(
        state=state,
        action=action,
        prediction=prediction,
        context=context,
        vehicle=vehicle,
        config=config,
        energy_scale_kwh=energy_scale_kwh,
        deadline_s=deadline_s,
        max_episode_steps=max_episode_steps,
        source_real_transition_id=source_real_transition_id,
        model_condition=model.condition,
        model_version=model.version,
    )
