import dataclasses
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gym_longicontrol import DeterministicTrack, StochasticTrack, register_envs
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.track import FixedTrackGenerator, StochasticTrackGenerator
from gym_longicontrol.domain.vehicle import VehicleModel


@pytest.mark.parametrize("env_id", ["DeterministicTrack-v1", "StochasticTrack-v1"])
def test_gymnasium_contract(env_id):
    env = gym.make(env_id)
    try:
        check_env(env.unwrapped, skip_render_check=True)
        obs, info = env.reset(seed=5)
        assert obs.shape == (8,) and env.observation_space.contains(obs)
        assert info["total_energy_kwh"] == 0
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(np.array([0.8]))
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            assert isinstance(terminated, bool) and isinstance(truncated, bool)
            assert reward == pytest.approx(
                np.dot(
                    env.unwrapped.reward_weights,
                    list(info["reward_components"].values()),
                )
            )
    finally:
        env.close()


def test_seed_replay_and_independent_instances():
    first, second = StochasticTrack(), StochasticTrack()
    trajectory = []
    for env in (first, second):
        env.reset(seed=42)
        samples = [env.step([float(action)])[0] for action in np.linspace(-1, 1, 100)]
        trajectory.append(samples)
    np.testing.assert_array_equal(trajectory[0], trajectory[1])
    positions = first.track.positions_m.copy()
    first.reset(seed=42)
    np.testing.assert_array_equal(first.track.positions_m, positions)
    limits = first.track.limits_m_s.copy()
    first.reset()
    assert not (
        np.array_equal(first.track.positions_m, positions)
        and np.array_equal(first.track.limits_m_s, limits)
    )


def test_sensor_visibility_boundaries():
    track = FixedTrackGenerator([0, 0.1, 0.2], [50, 80, 40]).track
    at_start = track.sense(0, 150)
    assert at_start.future_limits_m_s == pytest.approx([80 / 3.6] * 2)
    assert at_start.distances_m == (100, 150)
    at_range = track.sense(50, 150)
    assert at_range.future_limits_m_s == pytest.approx([80 / 3.6, 40 / 3.6])
    assert at_range.distances_m == (50, 150)
    at_sign = track.sense(100, 150)
    assert at_sign.current_limit_m_s == pytest.approx(80 / 3.6)
    assert at_sign.distances_m == (100, 150)
    assert track.sense(-10, 150).current_limit_m_s == pytest.approx(50 / 3.6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speed_limits": [50]},
        {"speed_limit_positions": [0.1, 0.2, 0.3, 0.4]},
        {"speed_limits": [0, 50, 60, 70]},
        {"speed_limits": [float("nan")] * 4},
        {"speed_limit_positions": [0, 0.1, 0.1, 0.2]},
        {"speed_limit_positions": [0, 0.1, 0.2, 1.1]},
        {"reward_weights": [1, 2]},
        {"energy_factor": -1},
        {"energy_factor": 6},
        {"render_mode": "invalid"},
        {"car_id": "missing"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        DeterministicTrack(**kwargs)


@pytest.mark.parametrize("action", [0.5, [], [0, 0], [np.nan], [np.inf]])
def test_invalid_actions(action):
    env = DeterministicTrack()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)


def test_clipping_energy_and_context_semantics():
    first = DeterministicTrack(energy_factor=1)
    second = DeterministicTrack(energy_factor=4)
    first.reset()
    second.reset()
    accumulated = 0
    for _ in range(30):
        a = first.step([10])
        b = second.step([1])
        np.testing.assert_array_equal(a[0][:-1], b[0][:-1])
        assert a[0][-1] == 0.2 and b[0][-1] == 0.8
        assert a[1] == b[1]
        accumulated += a[4]["step_energy_kwh"]
        assert a[4]["total_energy_kwh"] == pytest.approx(accumulated)
        assert a[4]["velocity"] == a[4]["velocity_km_h"]
    first.reset()
    assert first.state == VehicleState()


def test_finish_vs_time_limit():
    limited = gym.make("DeterministicTrack-v1", max_episode_steps=2)
    limited.reset()
    assert limited.step([0])[2:4] == (False, False)
    assert limited.step([0])[2:4] == (False, True)
    limited.close()
    env = DeterministicTrack()
    env.reset()
    env.state = VehicleState(position_m=999.9, velocity_m_s=10)
    assert env.step([0])[2:4] == (True, False)
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0])


def test_dynamics_at_velocity_limits():
    car = VehicleModel()
    state = VehicleState(velocity_m_s=36.99)
    next_state, power, energy = advance(state, 1, car, SimulationConfig())
    assert next_state.velocity_m_s == 37
    assert next_state.position_m == pytest.approx(
        36.99 * 0.1 + 0.5 * next_state.acceleration_m_s2 * 0.1**2
    )
    assert energy == power * 0.1 / 3600
    next_state, _, _ = advance(VehicleState(), -1, car, SimulationConfig())
    assert next_state.velocity_m_s == 0


@pytest.mark.parametrize("length", [50, 100, 1000])
def test_generated_tracks_are_valid(length):
    generate = StochasticTrackGenerator(length)
    for seed in range(30):
        track = generate(np.random.default_rng(seed))
        assert track.positions_m[0] == 0
        assert (np.diff(track.positions_m) > 0).all()
        assert (track.positions_m < length).all()
        assert (track.limits_m_s >= 20 / 3.6).all()
        assert (track.limits_m_s <= 100 / 3.6).all()
        assert (abs(np.diff(track.limits_m_s)) <= 40 / 3.6 + 1e-10).all()


def test_registration_is_idempotent_and_headless():
    register_envs()
    register_envs()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gym_longicontrol; "
            "e=gym_longicontrol.DeterministicTrack(); e.reset(); e.step([0]); "
            "assert not {'torch','matplotlib','sklearn','gym'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_config_immutable_and_validated():
    with pytest.raises(ValueError):
        SimulationConfig(dt_s=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        SimulationConfig().dt_s = 0.2


def test_rgb_rendering_does_not_change_dynamics():
    pytest.importorskip("matplotlib")
    env = DeterministicTrack(render_mode="rgb_array")
    env.reset()
    state = env.state
    frame = env.render()
    assert frame.shape == (450, 1000, 3) and frame.dtype == np.uint8
    assert env.state == state
    assert np.array_equal(frame, env.render())
    assert len(env._renderer.history) == 1
    env.step([1])
    assert not np.array_equal(frame, env.render())
    env.close()
    env.close()
    env.reset()
    assert env.render().shape == frame.shape
    env.close()


def test_rgb_layout_is_stable_across_steps_and_resets():
    pytest.importorskip("matplotlib")
    env = DeterministicTrack(render_mode="rgb_array")
    try:
        env.reset(seed=42)
        initial_frame = env.render()
        for action in (0.5, -1.0, 0.3):
            env.step([action])
            state = env.state
            frame = env.render()
            history_length = len(env._renderer.history)
            for _ in range(3):
                np.testing.assert_array_equal(frame, env.render())
            assert env.state == state
            assert len(env._renderer.history) == history_length
        env.reset(seed=42)
        np.testing.assert_array_equal(initial_frame, env.render())
    finally:
        env.close()
