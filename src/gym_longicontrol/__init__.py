"""Gymnasium environments for longitudinal vehicle control."""

import sys

from .envs import (
    DeterministicTrack,
    LongiControlEnv,
    MODeterministicTrack,
    MOStochasticTrack,
    StochasticTrack,
)
from .registration import register_envs

__version__ = "1.0.0"
__all__ = [
    "DeterministicTrack",
    "StochasticTrack",
    "LongiControlEnv",
    "MODeterministicTrack",
    "MOStochasticTrack",
    "register_envs",
]

register_envs()

# Support the historical gym.make("gym_longicontrol:...-v0") import order.
# Do not import optional Gym during ordinary Gymnasium use.
if getattr(sys.modules.get("gym"), "__version__", None) == "0.23.1":
    from .compat import register_legacy_envs

    register_legacy_envs()
