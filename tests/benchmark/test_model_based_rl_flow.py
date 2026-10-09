import numpy as np
import pytest

from benchmarks.model_based_rl.config import load_configuration
from benchmarks.model_based_rl.imagination import (
    RealEpisodePIDGate,
    RealReplaySource,
    RealSourceReplay,
    SyntheticReplay,
    exact_mixed_batch,
    generate_synthetic_transitions,
    refresh_due,
)
from benchmarks.model_based_rl.model_state import ModelState, TrackContext
from benchmarks.model_based_rl.physics_model import PhysicsDynamicsModel
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.track import Track
from gym_longicontrol.domain.vehicle import VehicleModel


def setup_replay():
    state = ModelState(VehicleState())
    context = TrackContext(
        Track(np.asarray([0.0]), np.asarray([10.0])), 150.0, 1000.0, 1.0
    )
    replay = RealSourceReplay()
    replay.append(RealReplaySource(17, state, context, np.zeros(8)))
    return replay


def test_schedule_and_one_step_provenance_use_current_actor():
    configuration = load_configuration()
    assert not refresh_due(9999, configuration.imagination)
    assert refresh_due(10_000, configuration.imagination)
    assert refresh_due(10_250, configuration.imagination)
    assert not refresh_due(10_251, configuration.imagination)
    calls = []

    def actor(observation, rng):
        calls.append(observation.copy())
        return np.asarray([0.25])

    vehicle = VehicleModel()
    simulation = SimulationConfig()
    rows = generate_synthetic_transitions(
        model=PhysicsDynamicsModel(vehicle, simulation),
        real_replay=setup_replay(),
        actor=actor,
        count=3,
        rng=np.random.default_rng(1),
        vehicle=vehicle,
        simulation_config=simulation,
        energy_scale_kwh=0.25,
        deadline_s=140.0,
        max_episode_steps=1800,
    )
    assert len(calls) == len(rows) == 3
    assert all(row.source == "model" for row in rows)
    assert all(row.model_condition == "physics" for row in rows)
    assert all(row.source_real_transition_id == 17 for row in rows)
    assert all(
        row.next_state.episode_step == row.state.episode_step + 1 for row in rows
    )


def test_exact_batch_ratio_and_no_real_budget_side_effect():
    configuration = load_configuration()
    vehicle = VehicleModel()
    simulation = SimulationConfig()
    rows = generate_synthetic_transitions(
        model=PhysicsDynamicsModel(vehicle, simulation),
        real_replay=setup_replay(),
        actor=lambda _observation, _rng: np.asarray([0.0]),
        count=128,
        rng=np.random.default_rng(2),
        vehicle=vehicle,
        simulation_config=simulation,
        energy_scale_kwh=0.25,
        deadline_s=140.0,
        max_episode_steps=1800,
    )
    synthetic = SyntheticReplay(1000)
    synthetic.extend(rows)
    real, model = exact_mixed_batch(
        [object()] * 128,
        synthetic,
        config=configuration.imagination,
        rng=np.random.default_rng(3),
    )
    assert len(real) == len(model) == 128
    assert synthetic.generated_total == 128
    assert synthetic.sampled_total == 128


class FakePolicy:
    def __init__(self):
        self.calls = 0

    def pre_update_fn(self, **_kwargs):
        self.calls += 1


def test_pid_accepts_real_episode_and_rejects_synthetic_episode():
    gate = RealEpisodePIDGate()
    policy = FakePolicy()
    gate.update(policy, source="real", stats={"cost": np.zeros(2)})
    assert gate.real_episode_updates == policy.calls == 1
    with pytest.raises(ValueError, match="Synthetic episodes"):
        gate.update(policy, source="model", stats={"cost": np.zeros(2)})
    assert policy.calls == 1
