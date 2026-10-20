"""Accounting tests for the frozen SAMEINFO_VALIDATION_V1 artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.same_information_scalar_vs_constrained.config import load_configuration
from benchmarks.same_information_scalar_vs_constrained.interpretation import classify

HERE = (
    Path(__file__).parents[2] / "benchmarks" / "same_information_scalar_vs_constrained"
)
RESULTS = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
VALIDATION_SEEDS = (
    9_500_001,
    9_500_002,
    9_500_003,
    9_500_005,
    9_500_006,
    9_500_007,
    9_500_008,
    9_500_011,
    9_500_012,
    9_500_014,
    9_500_015,
    9_500_016,
    9_500_017,
    9_500_018,
    9_500_019,
    9_500_020,
    9_500_021,
    9_500_022,
)
POLICIES = ("C11", "C29", "C47", "S11", "S29", "S47")


def test_validation_has_exactly_108_unique_policy_track_pairs():
    episodes = RESULTS["complete_compact_episode_metrics"]
    keys = [(row["policy"], row["evaluation_seed"]) for row in episodes]
    assert len(episodes) == len(keys) == len(set(keys)) == 108
    assert Counter(row["policy"] for row in episodes) == Counter(
        {policy: 18 for policy in POLICIES}
    )
    for policy in POLICIES:
        assert (
            tuple(row["evaluation_seed"] for row in episodes if row["policy"] == policy)
            == VALIDATION_SEEDS
        )
    integrity = RESULTS["execution_integrity"]
    assert integrity["validation_open_count"] == 1
    assert integrity["each_policy_track_pair_exactly_once"] is True
    assert integrity["interrupted_finalizer_recovery"]["duplicate_evaluations"] == 0


def test_episode_metrics_and_exclusive_failure_modes_are_consistent():
    episodes = RESULTS["complete_compact_episode_metrics"]
    required = {
        "policy",
        "method",
        "training_seed",
        "validation_task",
        "feasible",
        "completed",
        "deadline_compliant",
        "speed_compliant",
        "travel_time_s",
        "maximum_speed_violation_m_s",
        "integrated_speed_violation_m",
        "traction_energy_kwh",
        "regenerative_energy_kwh",
        "signed_net_energy_kwh",
        "exclusive_failure_mode",
    }
    for row in episodes:
        assert required <= row.keys()
        assert row["validation_task"] == row["evaluation_seed"]
        assert row["deadline_compliant"] == (
            row["completed"] and row["travel_time_s"] <= 140.0
        )
        assert row["speed_compliant"] == (row["maximum_speed_violation_m_s"] <= 0.0)
        assert row["signed_net_energy_kwh"] == pytest.approx(
            row["traction_energy_kwh"] - row["regenerative_energy_kwh"]
        )
    for policy in POLICIES:
        observed = Counter(
            row["exclusive_failure_mode"] for row in episodes if row["policy"] == policy
        )
        assert (
            dict(sorted(observed.items()))
            == RESULTS["per_policy"][policy]["exclusive_failure_modes"]
        )


def test_primary_secondary_and_paired_counts_rebuild_from_episodes():
    episodes = RESULTS["complete_compact_episode_metrics"]
    field_by_endpoint = {
        "requirement_satisfaction": "feasible",
        "completion": "completed",
        "deadline_compliance": "deadline_compliant",
        "speed_compliance": "speed_compliant",
    }
    for policy in POLICIES:
        rows = [row for row in episodes if row["policy"] == policy]
        for endpoint, field in field_by_endpoint.items():
            assert (
                sum(bool(row[field]) for row in rows)
                == RESULTS["per_policy"][policy]["counts"][endpoint]
            )

    paired = RESULTS["paired_differences_constrained_minus_scalar"]
    for seed in (11, 29, 47):
        for endpoint in field_by_endpoint:
            expected = (
                RESULTS["per_policy"][f"C{seed}"]["counts"][endpoint]
                - RESULTS["per_policy"][f"S{seed}"]["counts"][endpoint]
            )
            assert (
                paired["per_seed"][str(seed)][endpoint]["count_difference"] == expected
            )

    deltas = tuple(
        paired["per_seed"][str(seed)]["requirement_satisfaction"]["count_difference"]
        for seed in (11, 29, 47)
    )
    pooled = paired["pooled"]["requirement_satisfaction"]["count_difference"]
    assert deltas == (8, 7, 8)
    assert pooled == 23
    assert classify(deltas, pooled) == "CONSTRAINED_ROBUST_ADVANTAGE"
    assert RESULTS["classification"]["value"] == classify(deltas, pooled)


def test_feasible_only_energy_excludes_all_infeasible_episodes():
    episodes = RESULTS["complete_compact_episode_metrics"]
    for method in ("constrained", "scalar"):
        expected = [
            row["signed_net_energy_kwh"]
            for row in episodes
            if row["method"] == method and row["feasible"]
        ]
        observed = RESULTS["feasible_only_energy"][method]
        assert observed["count"] == len(expected)
        assert observed["values_kwh"] == expected
    assert RESULTS["feasible_only_energy"]["constrained"]["count"] == 23
    assert RESULTS["feasible_only_energy"]["scalar"] == {
        "count": 0,
        "maximum_kwh": None,
        "mean_kwh": None,
        "median_kwh": None,
        "minimum_kwh": None,
        "values_kwh": [],
    }


def test_hashes_counters_authorization_and_paper_remain_frozen():
    configuration = load_configuration()
    assert configuration.configuration_sha256 == (
        "96c587c9b14c0238257be60983bc9db7951fdabe2cebd867d5b9b13627e8c1f2"
    )
    assert configuration.raw["authorization"]["validation_authorized"] is False
    integrity = RESULTS["execution_integrity"]
    assert integrity["all_actor_hashes_unchanged"] is True
    assert integrity["all_checkpoint_hashes_unchanged"] is True
    assert integrity["all_training_counters_unchanged"] is True
    assert integrity["no_training_during_validation"] is True
    assert integrity["no_constrained_retraining"] is True
    assert integrity["no_scalar_retraining"] is True
    assert integrity["paper"] == {
        "episode_count": 0,
        "state": "SEALED_UNINSPECTED",
    }
    assert RESULTS["validation_split"]["ordered_sha256"] == (
        "d8a658e96b0090f8e0cdcdfdec3b291329045dc3dac9541e0603f34d916bf10a"
    )
    assert all(
        row["expected_sha256"] == row["observed_sha256"] and row["verified"]
        for row in RESULTS["provenance"]["checkpoint_integrity"].values()
    )
