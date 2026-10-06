import json
from pathlib import Path

import pytest

from benchmarks.goal_conditioned.config import (
    configuration_sha256,
    load_configuration,
)

ROOT = Path("benchmarks/goal_conditioned")


def _read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_completed_result_matches_the_frozen_primary_comparison():
    results = _read("results.json")
    assert results["configuration_sha256"] == configuration_sha256(
        load_configuration()
    )
    comparison = results["primary_comparison"]
    assert comparison == {
        "episodes_per_condition": 27,
        "estimand": "RSR(sac-her) - RSR(sac-no-her)",
        "interpretation": (
            "HER produced positive virtual samples, but canonical Validation "
            "success did not improve."
        ),
        "paired_seed_track_contingency": {
            "both_succeed": 0,
            "her_alone_succeeds": 0,
            "neither_succeeds": 27,
            "no_her_alone_succeeds": 0,
        },
        "preregistered_outcome": "B",
        "rsr_difference": 0.0,
        "sac_her_successes": 0,
        "sac_no_her_successes": 0,
    }
    for condition in ("sac-no-her", "sac-her"):
        overall = results["final_validation"][condition]["overall"]
        assert overall["success_count"] == 0
        assert overall["completion_count"] == 0
        assert overall["speed_compliant_count"] == 27
        assert overall["feasible_energy_count"] == 0
        assert overall["mean_feasible_energy_kwh"] is None


def test_replay_and_real_training_counters_remain_distinct():
    results = _read("results.json")
    no_her = results["replay_diagnostics"]["sac-no-her"]["overall"]
    her = results["replay_diagnostics"]["sac-her"]["overall"]
    assert no_her["virtual_rows"] == 0
    assert no_her["positive_real_reward_rows"] == 0
    assert her["virtual_rows"] == 182_498_400
    assert her["eligible_virtual_rows"] == 179_437_694
    assert her["relabeled_virtual_rows"] == 179_397_817
    assert her["fallback_virtual_rows"] == 3_060_706
    assert her["positive_virtual_reward_rows"] == 149_320
    assert her["rates"]["positive_virtual_reward_rate"] == pytest.approx(
        0.000818198954073022
    )
    training = results["training_outcomes"]
    assert training["sac-no-her"]["overall"][
        "canonical_successful_episode_count"
    ] == 0
    assert training["sac-her"]["overall"][
        "canonical_successful_episode_count"
    ] == 1
    assert training["sac-her"]["overall"]["first_canonical_success_step"] == 96_533


def test_execution_and_compact_episodes_keep_paper_tracks_sealed():
    manifest = _read("execution_manifest.json")
    assert manifest["status"] == "COMPLETE"
    assert manifest["validation_opened"] is True
    assert manifest["paper_test_tracks_opened"] is False
    assert manifest["totals"] == {
        "development_evaluation_transitions": 577_306,
        "training_transitions": 1_800_000,
        "validation_evaluation_transitions": 97_200,
    }
    assert len(manifest["policies"]) == 6
    assert all(
        policy["training_transitions"] == 300_000
        and policy["gradient_updates"] == 298_200
        and policy["validation_status"] == "COMPLETE"
        for policy in manifest["policies"].values()
    )

    compact = _read("validation_episodes.json")
    assert compact["episode_count"] == len(compact["episodes"]) == 54
    assert {row["evaluation_seed"] for row in compact["episodes"]} == set(
        range(3000, 3009)
    )
    assert not {row["evaluation_seed"] for row in compact["episodes"]} & set(
        range(4000, 4018)
    )
