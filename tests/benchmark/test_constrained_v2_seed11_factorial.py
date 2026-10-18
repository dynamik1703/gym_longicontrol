"""Execution-gate tests for the seed-11 track-by-collector factorial."""

import json
from pathlib import Path

from benchmarks.constrained_v2_seed11_factorial.stream_preflight import (
    build_report,
    file_sha256,
)

STUDY = Path("benchmarks/constrained_v2_seed11_factorial")


def test_exact_streams_match_audit_and_current_archive():
    report = build_report()
    assert report["audit_source"]["historical_first_100_match"] is True
    assert report["audit_source"]["current_first_100_match"] is True
    assert (
        report["archived_current_source"]["generated_reset_seed_prefix_match"]
        is True
    )
    assert report["checks"]["streams_differ_at_episode_zero"] is True
    assert report["checks"]["current_reserved_numeric_seed_overlap"] == []
    assert report["initial_learner"]["all_identical"] is True
    assert report["initial_learner"]["actor_matches_audit"] is True


def test_content_collision_fails_main_execution_gate():
    report = build_report()
    expected = {
        ("historical", 158, "development", 2006),
        ("current", 163, "development", 2006),
        ("current", 242, "development", 2006),
    }
    observed = {
        (
            stream,
            item["episode_index"],
            item["reserved_split"],
            item["reserved_track_seed"],
        )
        for stream, items in report["collisions"].items()
        for item in items
        if item["reserved_split"] == "development"
    }
    assert expected <= observed
    assert report["checks"]["minimum_episodes_at_300k_with_1800_step_cap"] == 167
    assert report["checks"]["reserved_content_fingerprint_protection_pass"] is False
    assert report["main_training_authorized"] is False
    assert report["status"] == "EXECUTION_GATE_FAILED"


def test_no_evaluation_or_training_was_run_after_gate_failure():
    execution = build_report()["execution"]
    assert execution["conditions_started"] == 0
    assert execution["real_transitions"] == 0
    assert execution["gradient_updates"] == 0
    assert execution["development_episodes"] == 0
    assert execution["validation_episodes"] == 0
    assert execution["paper_episodes"] == 0


def test_frozen_protocol_hashes_and_seals_are_intact():
    freeze = json.loads((STUDY / "freeze.json").read_text(encoding="utf-8"))
    assert file_sha256(STUDY / "canonical.json") == freeze[
        "factorial_configuration_sha256"
    ]
    assert file_sha256(STUDY / "DESIGN.md") == freeze["design_sha256"]
    assert file_sha256(STUDY / "PROTOCOL.md") == freeze["protocol_sha256"]
    manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["validation"] == {"state": "SEALED", "episode_count": 0}
    assert manifest["paper"] == {"state": "SEALED", "episode_count": 0}
