"""The MO environment contract must not require the optional training stack."""

import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

from gym_longicontrol import register_envs
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.envs.multi_objective import REWARD_NAMES


@pytest.mark.parametrize("name", ["DeterministicTrack", "StochasticTrack"])
def test_vector_reward_matches_scalar_components_and_dynamics(name):
    scalar = gym.make(f"{name}-v1")
    vector = gym.make(f"MO{name}-v1")
    try:
        assert vector.spec.disable_env_checker
        assert not scalar.spec.disable_env_checker
        assert vector.unwrapped.reward_dim == 4
        assert vector.unwrapped.reward_names == REWARD_NAMES
        a, a_info = scalar.reset(seed=42)
        b, b_info = vector.reset(seed=42)
        np.testing.assert_array_equal(a, b)
        assert a_info == b_info
        for action in np.linspace(-1, 1, 100):
            a = scalar.step([action])
            b = vector.step([action])
            np.testing.assert_array_equal(a[0], b[0])
            assert a[2:] == b[2:]
            assert b[1].shape == (4,) and b[1].dtype == np.float64
            assert np.isfinite(b[1]).all()
            assert vector.unwrapped.reward_space.contains(b[1])
            np.testing.assert_array_equal(
                b[1], [b[4]["reward_components"][name] for name in REWARD_NAMES]
            )
            assert a[1] == pytest.approx(scalar.unwrapped.reward_weights @ b[1])
    finally:
        scalar.close()
        vector.close()


@pytest.mark.parametrize("name", ["MODeterministicTrack", "MOStochasticTrack"])
def test_seed_reset_termination_and_truncation(name):
    env = gym.make(f"{name}-v1", max_episode_steps=2)
    try:
        env.reset(seed=7)
        original = env.step([0.5])
        assert env.step([0.5])[2:4] == (False, True)
        env.reset(seed=7)
        repeated = env.step([0.5])
        np.testing.assert_array_equal(original[0], repeated[0])
        np.testing.assert_array_equal(original[1], repeated[1])
        assert original[2:] == repeated[2:]
        env.reset(seed=7)
        env.unwrapped.state = VehicleState(position_m=999.9, velocity_m_s=10)
        assert env.step([0])[2:4] == (True, False)
        with pytest.raises(gym.error.ResetNeeded):
            env.step([0])
    finally:
        env.close()


def test_context_is_not_a_preference_and_reward_array_is_independent():
    first = gym.make("MODeterministicTrack-v1", energy_factor=1)
    second = gym.make("MODeterministicTrack-v1", energy_factor=4)
    try:
        first.reset(seed=1)
        second.reset(seed=1)
        a, b = first.step([0.5]), second.step([0.5])
        assert a[0][-1] != b[0][-1]
        np.testing.assert_array_equal(a[1], b[1])
        saved = a[1].copy()
        first.step([0.1])
        np.testing.assert_array_equal(a[1], saved)
        a[1][0] = 123
        assert a[4]["reward_components"]["forward"] != 123
    finally:
        first.close()
        second.close()


def test_scalar_weights_rejected_in_mo_env():
    with pytest.raises(TypeError, match="unweighted"):
        gym.make("MOStochasticTrack-v1", reward_weights=(1, 1, 1, 1))


def test_mo_registration_is_idempotent_and_headless():
    register_envs()
    register_envs()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gymnasium as gym; import gym_longicontrol; "
            "env=gym.make('MOStochasticTrack-v1'); env.reset(seed=1); env.step([0]); "
            "assert not {'torch','morl_baselines','mo_gymnasium','matplotlib','gym'} "
            "& set(sys.modules); env.close()",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
