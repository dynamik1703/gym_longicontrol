"""Run with the isolated interpreter after installing the built wheel."""

import sys
from importlib.metadata import version
from importlib.resources import files

import gymnasium as gym

import gym_longicontrol

assert version("gym-longicontrol") == "1.0.0"
assert "site-packages" in gym_longicontrol.__file__
assets = files("gym_longicontrol").joinpath("assets")
assert assets.joinpath("vehicle/BMW_electric_i3_2014.npz").is_file()
assert assets.joinpath("vehicle/BMW_electric_i3_2014.json").is_file()
assert assets.joinpath("track/img/car_80x40.png").is_file()
for name in (
    "DeterministicTrack-v1",
    "StochasticTrack-v1",
    "MODeterministicTrack-v1",
    "MOStochasticTrack-v1",
):
    env = gym.make(name)
    try:
        observation, _ = env.reset(seed=2)
        assert env.observation_space.contains(observation)
        observation, reward, terminated, truncated, _ = env.step([0.5])
        if name.startswith("MO"):
            assert env.unwrapped.reward_space.contains(reward)
            assert env.unwrapped.reward_dim == 4
        else:
            assert isinstance(reward, float)
        assert env.observation_space.contains(observation)
        assert not terminated and not truncated
    finally:
        env.close()
assert not {
    "torch",
    "matplotlib",
    "sklearn",
    "gym",
    "morl_baselines",
    "mo_gymnasium",
} & set(sys.modules)
print("Installed wheel: scalar/MO environments and packaged assets passed.")
