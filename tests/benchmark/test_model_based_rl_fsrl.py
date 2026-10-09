"""Pinned-stack integration; skipped when the optional FSRL stack is absent."""

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("fsrl")
pytest.importorskip("tianshou")

from tianshou.data import Batch, Collector, ReplayBuffer  # noqa: E402
from tianshou.policy import BasePolicy  # noqa: E402

from benchmarks.model_based_rl.adapter import model_step  # noqa: E402
from benchmarks.model_based_rl.config import load_configuration  # noqa: E402
from benchmarks.model_based_rl.learned_model import (  # noqa: E402
    ProbabilisticEnsemble,
)
from benchmarks.model_based_rl.model_disabled_parity import run_check  # noqa: E402
from benchmarks.model_based_rl.model_state import (  # noqa: E402
    ModelState,
    TrackContext,
)
from benchmarks.model_based_rl.physics_model import PhysicsDynamicsModel  # noqa: E402
from benchmarks.model_based_rl.resource_check import optional_fsrl_probe  # noqa: E402
from gym_longicontrol.domain.state import SimulationConfig, VehicleState  # noqa: E402
from gym_longicontrol.domain.track import Track  # noqa: E402
from gym_longicontrol.domain.vehicle import VehicleModel  # noqa: E402


class _OneStepCostEnvironment(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return (
            np.ones(2, dtype=np.float32),
            0.0,
            False,
            False,
            {"cost": np.asarray([0.25, 0.5], dtype=np.float64)},
        )


class _ZeroPolicy(BasePolicy):
    def __init__(self):
        super().__init__(action_space=_OneStepCostEnvironment.action_space)

    def forward(self, batch, state=None, **kwargs):
        return Batch(act=np.zeros((len(batch), 1)), state=state)

    def learn(self, batch, **kwargs):
        return {}


def test_pinned_tianshou_collector_preserves_one_step_vector_cost():
    collector = Collector(
        _ZeroPolicy(), _OneStepCostEnvironment(), ReplayBuffer(8)
    )
    stats = collector.collect(n_step=1)
    assert int(stats["n/st"]) == 1
    assert int(stats["n/ep"]) == 0
    assert len(collector.buffer) == 1
    np.testing.assert_allclose(collector.buffer[0].info.cost, [0.25, 0.5])


def test_pinned_fsrl_mixed_update_is_finite_and_counted():
    configuration = load_configuration()
    simulation = SimulationConfig()
    vehicle = VehicleModel()
    physics = PhysicsDynamicsModel(vehicle, simulation)
    context = TrackContext(
        Track(
            np.asarray([0.0, 200.0, 500.0]),
            np.asarray([10.0, 20.0, 8.0]),
        ),
        150.0,
        1000.0,
        1.0,
    )
    rng = np.random.default_rng(77)
    rows = []
    for index in range(512):
        state = ModelState(
            VehicleState(
                position_m=float(index % 450),
                velocity_m_s=float(1 + index % 25),
                acceleration_m_s2=float(index % 5 - 2),
                elapsed_time_s=float(index % 1000) * 0.1,
            ),
            episode_step=index % 1000,
        )
        rows.append(
            model_step(
                physics,
                state,
                float(-1 + 2 * (index % 101) / 100),
                context=context,
                vehicle=vehicle,
                config=simulation,
                energy_scale_kwh=0.25,
                deadline_s=140.0,
                max_episode_steps=1800,
                source_real_transition_id=index,
                rng=rng,
            )
        )
    model = ProbabilisticEnsemble(configuration.model, simulation, seed=77)
    result = optional_fsrl_probe(configuration, model, [], rows)
    assert result["available"] is True
    assert result["updates"] == 10
    assert result["updates_per_second"] > 0
    assert result["policy_parameter_count"] == 232_974


def test_model_disabled_adapter_is_bit_identical_to_frozen_update():
    result = run_check()
    assert result["verified"] is True
    assert result["maximum_parameter_difference"] == 0.0
    assert result["simulator_transitions"] == 0
