"""Frozen physical control laws, V3 logging scope only."""

from benchmarks.bankers_viability_shield_v2.inherited_shield_adapter import (
    ACTION_GRID,
    BANKERS,
    CONFIG,
    METHODS,
    CausalMemory,
    MeteredVehicle,
    _ExternalExecutionAdapter,
    inherited_test,
)

from .safe_action_mask import closest_certified

DIAGNOSTIC_STRIDE = 100
__all__ = [
    "ACTION_GRID",
    "BANKERS",
    "CONFIG",
    "METHODS",
    "CausalMemory",
    "MeteredVehicle",
    "_ExternalExecutionAdapter",
    "inherited_test",
    "select",
    "DIAGNOSTIC_STRIDE",
]


def select(method, proposal, mask, legacy):
    if method == "CAUSAL60":
        return None, proposal
    if method == "LEGACY_NATIVE_POST_GREEDY":
        return None, 1.0 if legacy["accepted"] else proposal
    if method == "V3_SYNTHETIC_MINIMAL":
        return 0, -1.0
    index = (
        max(i for i, v in enumerate(mask) if v)
        if method == "BANKERS_GREEDY"
        else closest_certified(mask, proposal)
    )
    return index, ACTION_GRID[index]
