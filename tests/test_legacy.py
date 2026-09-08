import importlib.util
from pathlib import Path

import gymnasium
import numpy as np
import pytest

import gym_longicontrol.envs
from gym_longicontrol.compat import make_legacy_env
from gym_longicontrol.domain.vehicle import NumpyPowerModel


def test_four_value_adapter_and_time_limit():
    with pytest.warns(DeprecationWarning):
        env = make_legacy_env(max_episode_steps=1)
    assert env.seed(4) == [4]
    assert env.reset().shape == (8,)
    observation, reward, done, info = env.step([0])
    assert done and info["TimeLimit.truncated"]
    assert observation.shape == (8,) and np.isfinite(reward)
    env.close()


def test_gym_v0_registration():
    gym = pytest.importorskip("gym")
    from gym_longicontrol.compat import register_legacy_envs

    register_legacy_envs()
    register_legacy_envs()
    with pytest.warns(DeprecationWarning):
        env = gym.make("DeterministicTrack-v0")
    env.seed(2)
    observation = env.reset()
    assert env.observation_space.contains(observation)
    assert len(env.step(np.array([0.5]))) == 4
    env.close()


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_rollout_matches_original_sources(monkeypatch):
    pytest.importorskip("gym")
    root = Path(__file__).resolve().parents[1]
    legacy = root / "legacy/v0/gym_longicontrol/envs"
    car = _load_module("legacy_car", legacy / "car.py")
    monkeypatch.setattr(car, "get_power_estimator", lambda _: NumpyPowerModel())
    monkeypatch.setattr(gym_longicontrol.envs, "car", car, raising=False)
    original = _load_module("legacy_track", legacy / "deterministic_track.py")
    old = original.DeterministicTrack()
    modern = gym_longicontrol.DeterministicTrack()
    old_state, (new_state, _) = old.reset(), modern.reset()
    np.testing.assert_allclose(old_state, new_state, rtol=0, atol=1e-12)
    # Actions cover acceleration, coasting, braking, and multiple speed-limit crossings.
    actions = np.r_[
        np.full(350, 0.3), np.zeros(100), np.full(80, -0.5), np.full(1000, 0.2)
    ]
    for action in actions:
        a = old.step(np.array([action]))
        b = modern.step([action])
        np.testing.assert_allclose(a[0], b[0], rtol=0, atol=1e-10)
        assert a[1] == pytest.approx(b[1], abs=1e-10)
        assert a[2] == b[2] and not b[3]
        for key, value in a[3].items():
            assert value == pytest.approx(b[4][key], abs=1e-10)
        if a[2]:
            break
    assert old.position > 750
    modern.close()


def test_legacy_render_selects_mode():
    pytest.importorskip("matplotlib")
    with pytest.warns(DeprecationWarning):
        env = make_legacy_env()
    env.reset()
    assert env.render(mode="rgb_array").shape == (450, 1000, 3)
    env.close()


def test_step_requires_reset():
    env = gym_longicontrol.DeterministicTrack()
    with pytest.raises(gymnasium.error.ResetNeeded):
        env.step([0])
