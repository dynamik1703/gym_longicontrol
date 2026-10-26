"""Unchanged V3 controller selection and causal laws."""

from benchmarks.bankers_viability_shield_v3.controller_adapters import *  # noqa: F403
from benchmarks.bankers_viability_shield_v3.controller_adapters import (
    select as frozen_select,
)


def select(method, proposal, mask, legacy):
    if method == "V4_SYNTHETIC_MINIMAL":
        method = "V3_SYNTHETIC_MINIMAL"
    return frozen_select(method, proposal, mask, legacy)
