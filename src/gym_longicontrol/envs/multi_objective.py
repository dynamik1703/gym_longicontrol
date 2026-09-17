"""Vector-reward variants compatible with the MO-Gymnasium environment API."""

import numpy as np
from gymnasium import spaces

from .longicontrol import DeterministicTrack, StochasticTrack

REWARD_NAMES = ("forward", "energy", "jerk", "shock")


class _VectorReward:
    """Share scalar dynamics, but expose unweighted rewards on the base env.

    MORL-Baselines reads ``env.unwrapped.reward_space`` and ``reward_dim``.
    A wrapper around a scalar environment would not satisfy that contract.
    """

    reward_names = REWARD_NAMES
    reward_dim = len(REWARD_NAMES)

    def __init__(self, **kwargs):
        if "reward_weights" in kwargs:
            raise TypeError(
                "MO environments return unweighted rewards; apply preference "
                "weights in the agent or mo_gymnasium.wrappers.LinearReward"
            )
        super().__init__(**kwargs)
        # Conservative bounds also support caller-supplied vehicles and tracks.
        # Energy may be positive during regenerative braking.
        self.reward_space = spaces.Box(
            -np.inf, np.inf, shape=(self.reward_dim,), dtype=np.float64
        )

    def step(self, action):
        observation, _, terminated, truncated, info = super().step(action)
        reward = np.array(
            [info["reward_components"][name] for name in self.reward_names],
            dtype=self.reward_space.dtype,
        )
        return observation, reward, terminated, truncated, info


class MODeterministicTrack(_VectorReward, DeterministicTrack):
    """DeterministicTrack-v1 dynamics with four unweighted reward objectives."""


class MOStochasticTrack(_VectorReward, StochasticTrack):
    """StochasticTrack-v1 dynamics with four unweighted reward objectives."""
