"""Terminal route-clean gate tests for the repaired seed-11 factorial."""

import json
from pathlib import Path

from benchmarks.constrained_v2_seed11_factorial_v2.preflight import (
    build_report,
    file_sha256,
)

STUDY = Path("benchmarks/constrained_v2_seed11_factorial_v2")


def test_stream_coverage_bound_covers_every_physically_possible_run():
    coverage = build_report()["coverage"]
    assert coverage["minimum_physical_episode_steps"] == 347
    assert coverage["maximum_completed_episodes"] == 864
    assert coverage["maximum_exposed_tasks_including_final_active_episode"] == 865


def test_exact_stream_prefixes_match_completed_reproducibility_audit():
    checks = build_report()["stream_anchor_checks"]
    assert checks["historical_first_100_match"] is True
    assert checks["current_first_100_match"] is True
    assert build_report()["streams"]["T_HIST"]["episode_count"] == 865
    assert build_report()["streams"]["T_CURRENT"]["episode_count"] == 865


def test_frozen_diagnostic_9000005_fails_route_clean_gate():
    report = build_report()
    assert len(report["diagnostic_route_clean_checks"]) == 18
    assert len(report["collisions"]) == 1
    collision = report["collisions"][0]
    assert collision["numeric_seed"] == 9_000_005
    assert collision["numeric_seed_clean"] is True
    assert collision["route_sha256_absent_from_both_streams"] is False
    assert collision["full_task_sha256_absent_from_both_streams"] is False
    assert collision["matches"]["T_HIST"] == {
        "numeric_seed": [],
        "route": [],
        "full_task": [],
    }
    current = collision["matches"]["T_CURRENT"]
    assert current["numeric_seed"] == []
    assert current["route"] == current["full_task"]
    assert current["route"] == [
        {
            "training_episode_index": 559,
            "training_reset_seed": 1_141_367_263,
            "training_rng_draw_index": 559,
        }
    ]
    assert report["status"] == "EXECUTION_GATE_FAILED"
    assert report["main_training_authorized"] is False


def test_terminal_manifest_records_zero_execution_and_intact_seals():
    report = build_report()
    assert set(report["execution"].values()) <= {0, False}
    assert report["seals"] == {
        "legacy_development": "NOT_EVALUATED",
        "legacy_validation": "NOT_EVALUATED",
        "paper_4000_4017": "SEALED_UNINSPECTED",
    }
    manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "EXECUTION_GATE_FAILED"
    assert all(
        row["state"] == "NOT_STARTED" and row["attempts"] == []
        for row in manifest["conditions"].values()
    )
    assert manifest["paper"] == {
        "state": "SEALED_UNINSPECTED",
        "episode_count": 0,
    }


def test_materialized_results_and_freeze_are_current():
    materialized = json.loads((STUDY / "results.json").read_text(encoding="utf-8"))
    assert materialized == build_report()
    freeze = json.loads((STUDY / "freeze.json").read_text(encoding="utf-8"))
    assert freeze["terminal_state"] == "EXECUTION_GATE_FAILED"
    assert freeze["main_training_started"] is False
    for name, expected in freeze["files"].items():
        assert file_sha256(STUDY / name) == expected
