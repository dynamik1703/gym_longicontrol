from dataclasses import replace

import numpy as np
import pytest

from benchmarks.model_based_rl.adapter import model_step
from benchmarks.model_based_rl.config import load_configuration
from benchmarks.model_based_rl.learned_model import Normalization, ProbabilisticEnsemble
from benchmarks.model_based_rl.model_state import ModelState, TrackContext
from benchmarks.model_based_rl.model_training import (
    RealModelDataset,
    RealModelExample,
    train_ensemble,
)
from benchmarks.model_based_rl.physics_model import PhysicsDynamicsModel
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.track import Track
from gym_longicontrol.domain.vehicle import VehicleModel


def context():
    return TrackContext(
        Track(np.asarray([0.0, 10.0]), np.asarray([8.0, 5.0])),
        sensor_range_m=150.0,
        track_length_m=20.0,
        energy_factor=1.0,
    )


def test_physics_acceleration_braking_energy_and_boundary_projection():
    configuration = load_configuration()
    simulation = SimulationConfig(track_length_m=20.0)
    vehicle = VehicleModel()
    model = PhysicsDynamicsModel(vehicle, simulation)
    rng = np.random.default_rng(1)
    accelerating = ModelState(VehicleState(position_m=9.8, velocity_m_s=7.0))
    transition = model_step(
        model,
        accelerating,
        1.0,
        context=context(),
        vehicle=vehicle,
        config=simulation,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_s=configuration.v2.task.max_time_s,
        max_episode_steps=1800,
        source_real_transition_id=4,
        rng=rng,
    )
    assert transition.next_state.vehicle.acceleration_m_s2 > 0
    assert transition.next_state.vehicle.total_energy_kwh > 0
    assert transition.next_state.maximum_speed_excess_m_s > 0
    assert transition.next_state.speed_violation_count == 1
    continued = model_step(
        model,
        transition.next_state,
        -1.0,
        context=context(),
        vehicle=vehicle,
        config=simulation,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_s=configuration.v2.task.max_time_s,
        max_episode_steps=1800,
        source_real_transition_id=5,
        rng=rng,
    )
    assert continued.next_state.vehicle.acceleration_m_s2 < 0
    assert continued.next_state.speed_violation_count == 1


def test_route_completion_timeout_and_metric_history():
    configuration = load_configuration()
    simulation = SimulationConfig(track_length_m=20.0)
    vehicle = VehicleModel()
    model = PhysicsDynamicsModel(vehicle, simulation)
    state = ModelState(
        VehicleState(position_m=19.9, velocity_m_s=2.0, elapsed_time_s=179.9),
        episode_step=1799,
    )
    transition = model_step(
        model,
        state,
        0.0,
        context=context(),
        vehicle=vehicle,
        config=simulation,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_s=configuration.v2.task.max_time_s,
        max_episode_steps=1800,
        source_real_transition_id=0,
        rng=np.random.default_rng(2),
    )
    assert transition.terminated
    assert not transition.truncated
    timeout_state = replace(state, vehicle=replace(state.vehicle, position_m=1.0))
    timeout = model_step(
        model,
        timeout_state,
        0.0,
        context=context(),
        vehicle=vehicle,
        config=simulation,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_s=configuration.v2.task.max_time_s,
        max_episode_steps=1800,
        source_real_transition_id=1,
        rng=np.random.default_rng(3),
    )
    assert timeout.truncated
    assert not timeout.terminated


def test_normalization_and_ensemble_properties_are_finite_and_independent():
    values = np.asarray([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
    normalization = Normalization.fit(values)
    assert np.isfinite(normalization.normalize(values)).all()
    assert np.allclose(
        normalization.denormalize(normalization.normalize(values)), values
    )

    configuration = load_configuration()
    model = ProbabilisticEnsemble(
        configuration.model, SimulationConfig(), seed=7
    )
    first = next(model.members[0].parameters()).detach().numpy()
    second = next(model.members[1].parameters()).detach().numpy()
    assert not np.array_equal(first, second)
    assert model.parameter_count == 862_456


def test_model_training_rejects_synthetic_and_selects_five_elites():
    configuration = load_configuration()
    state = ModelState(VehicleState())
    with pytest.raises(ValueError, match="real"):
        RealModelExample(0, state, 0.0, state, 0.0, source="model")
    dataset = RealModelDataset()
    vehicle = VehicleModel()
    simulation = SimulationConfig()
    physics = PhysicsDynamicsModel(vehicle, simulation)
    for index in range(32):
        action = -1.0 + 2.0 * index / 31
        prediction = physics.predict(state, action, rng=np.random.default_rng(index))
        next_state = ModelState(prediction.next_vehicle, episode_step=1)
        dataset.append(
            RealModelExample(
                index, state, action, next_state, prediction.signed_step_energy_kwh
            )
        )
    model = ProbabilisticEnsemble(configuration.model, simulation, seed=9)
    report = train_ensemble(
        model, dataset, rng=np.random.default_rng(9), maximum_epochs=1
    )
    assert len(report.elite_indices) == 5
    assert np.isfinite(report.holdout_losses).all()
    assert model.version == 1
    assert model.training_examples_seen == 32
    repeated = ProbabilisticEnsemble(configuration.model, simulation, seed=9)
    repeated_report = train_ensemble(
        repeated, dataset, rng=np.random.default_rng(9), maximum_epochs=1
    )
    assert repeated_report.elite_indices == report.elite_indices
    assert np.allclose(repeated_report.holdout_losses, report.holdout_losses)
    for left, right in zip(model.members.parameters(), repeated.members.parameters()):
        assert np.array_equal(left.detach().numpy(), right.detach().numpy())
