"""Explicit bridge for applications still using Gym's pre-0.24 API."""

import importlib.metadata
import warnings

import gymnasium


class LegacyAPIWrapper:
    """Four-value step/reset adapter, usable without installing old Gym."""

    def __init__(self, env):
        self.env = env
        self._next_seed = None

    def seed(self, seed=None):
        self._next_seed = seed
        self.env.action_space.seed(seed)
        return [seed]

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", self._next_seed)
        self._next_seed = None
        observation, _ = self.env.reset(seed=seed, **kwargs)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if truncated:
            info["TimeLimit.truncated"] = not terminated
        return observation, reward, terminated or truncated, info

    def render(self, mode="human"):
        core = self.env.unwrapped
        if core.render_mode != mode:
            core.close()
            core.render_mode = mode
        return core.render()

    def close(self):
        self.env.close()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_legacy_env(env_id="DeterministicTrack-v0", **kwargs):
    warnings.warn(
        "LongiControl's v0 API is deprecated; migrate to Gymnasium and v1 IDs. "
        "The adapter preserves API shape, not the historical seeded track sequence.",
        DeprecationWarning,
        stacklevel=2,
    )
    modern_id = env_id.removesuffix("-v0") + "-v1"
    return LegacyAPIWrapper(gymnasium.make(modern_id, **kwargs))


def _make_gym_env(env_id, **kwargs):
    import gym

    class GymAdapter(LegacyAPIWrapper, gym.Env):
        @property
        def unwrapped(self):
            # Gym must own this adapter's spec, not the inner Gymnasium spec.
            return self

        def __init__(self):
            super().__init__(make_legacy_env(env_id, **kwargs).env)
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
            self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(8,), dtype=float)
            self.metadata = {"render.modes": ["human", "rgb_array"]}
            self.reward_range = (-float("inf"), float("inf"))

    return GymAdapter()


def register_legacy_envs():
    """Register v0 IDs with Gym 0.23; install the ``legacy`` extra first."""
    version = importlib.metadata.version("gym")
    if version != "0.23.1":
        raise RuntimeError("The legacy bridge supports gym==0.23.1; use [legacy]")
    from gym.envs.registration import register, registry

    for name in ("DeterministicTrack", "StochasticTrack"):
        env_id = f"{name}-v0"
        if env_id not in registry.env_specs:
            # The inner Gymnasium TimeLimit is converted by LegacyAPIWrapper.
            register(
                id=env_id,
                entry_point="gym_longicontrol.compat:_make_gym_env",
                kwargs={"env_id": env_id},
            )
