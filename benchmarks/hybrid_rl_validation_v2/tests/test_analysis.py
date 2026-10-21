from __future__ import annotations

import pytest

from benchmarks.hybrid_rl_validation_v2.analysis import (
    _influence_gate,
    _paired_comparison,
)


def test_paired_energy_uses_only_the_registered_jointly_feasible_subset():
    episodes = {
        "CAUSAL60": {
            71_000_001: {"feasible": True, "energy_kwh": 1.0},
            71_000_002: {"feasible": True, "energy_kwh": 2.0},
        },
        "TIME_AWARE": {
            71_000_001: {"feasible": True, "energy_kwh": 0.9},
            71_000_002: {"feasible": False, "energy_kwh": 0.1},
        },
        "H11": {
            71_000_001: {"feasible": True, "energy_kwh": 0.8},
            71_000_002: {"feasible": True, "energy_kwh": 1.0},
        },
    }
    comparison = _paired_comparison(
        episodes,
        baseline="TIME_AWARE",
        treatment="H11",
        required_feasible=("CAUSAL60", "TIME_AWARE", "H11"),
        label="synthetic",
    )
    assert comparison["paired_track_count"] == 1
    assert comparison["saving_percent"]["mean"] == pytest.approx(100.0 / 9.0)
    assert comparison["per_track"][0]["numeric_seed"] == 71_000_001


def test_frozen_influence_gate_uses_aggregate_actions_and_episode_count():
    summary = {
        "shield": {
            "override_fraction": 0.25,
            "native_action_influence_fraction": 0.5,
            "influenced_episode_count": 36,
        }
    }
    assert _influence_gate(summary)["passed"]
    summary["shield"]["native_action_influence_fraction"] = 0.09
    assert not _influence_gate(summary)["passed"]
