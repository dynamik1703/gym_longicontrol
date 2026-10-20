import json
from pathlib import Path

import pytest

from benchmarks.gcsl_canonical import split

ROOT = Path("benchmarks/gcsl_canonical")


def test_six_historical_overlaps_are_exact_and_permanently_reported():
    detected = split.detected_historical_overlaps()
    assert len(detected) == 6
    assert split.verify_historical_overlap_ledger() is True
    ledger = json.loads(
        (ROOT / "historical_overlap_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["collision_count"] == 6
    assert [
        (row["training_policy"], row["accepted_task_index"], row["reset_seed"])
        for row in ledger["collisions"]
    ] == [
        ("G11", 104, 105371547),
        ("G29", 235, 69106380),
        ("G29", 646, 697284428),
        ("G47", 531, 246512140),
        ("G47", 608, 1275503705),
        ("G47", 689, 2015894932),
    ]
    assert all(row["route_sha256"] for row in ledger["collisions"])
    assert all(row["full_task_sha256"] for row in ledger["collisions"])


def test_historical_overlaps_do_not_fail_current_gcsl_gate():
    ledger = split.historical_overlap_ledger()
    for collision in ledger["collisions"]:
        seed = collision["training_policy"][1:]
        row = split.canonical_streams()[seed]["accepted_tasks"][
            collision["accepted_task_index"]
        ]
        split.assert_training_row_clean(row)


def test_training_collision_with_diagnostic_fails():
    row = split.evaluation_sets()["diagnostic"]["accepted_tasks"][0]
    with pytest.raises(RuntimeError, match="GCSL_DIAGNOSTIC_V1:ROUTE"):
        split.assert_training_row_clean(
            {
                "reset_seed": 123_456_789,
                "route_sha256": row["route_sha256"],
                "full_task_sha256": "unrelated",
            }
        )


def test_training_collision_with_validation_fails():
    row = split.evaluation_sets()["validation"]["accepted_tasks"][0]
    with pytest.raises(RuntimeError, match="GCSL_VALIDATION_V1:FULL_TASK"):
        split.assert_training_row_clean(
            {
                "reset_seed": 123_456_789,
                "route_sha256": "unrelated",
                "full_task_sha256": row["full_task_sha256"],
            }
        )


def test_new_candidate_matching_historical_content_is_rejected():
    candidate = split._candidate(1007, start=1007, accepted=[], other=[])
    assert candidate["accepted"] is False
    assert "PROTECTED_OR_INTERNAL_ROUTE" in candidate["rejection_reasons"]
    assert "PROTECTED_OR_INTERNAL_FULL_TASK" in candidate["rejection_reasons"]


def test_paper_is_numeric_only_and_never_instantiated(monkeypatch):
    instantiated = False

    def forbidden(_seed):
        nonlocal instantiated
        instantiated = True
        raise AssertionError("Paper content was instantiated")

    monkeypatch.setattr(split, "track_from_seed", forbidden)
    with pytest.raises(RuntimeError, match="fail-closed"):
        split._candidate(4000, start=4000, accepted=[], other=[])
    assert instantiated is False


def test_canonical_streams_are_unchanged_paired_and_cover_reachable_horizon():
    expected = {
        "11": "08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c",
        "29": "ca8912845c1293e86fd326a7415901c9beb0c70ae6a32db5e78263fceb355312",
        "47": "2b1323d664f8ca470975ce4238f33a1883bf6d88eb43d035681b5c1520b15255",
    }
    assert split.reachable_horizon() == {
        "real_transition_budget": 300_000,
        "minimum_physically_possible_episode_length": 347,
        "max_completed_episodes": 864,
        "max_exposed_tasks": 865,
    }
    for seed, stream in split.canonical_streams().items():
        assert stream["seed_sequence"] == [int(seed), 0x43414E4F]
        assert stream["ordered_sha256"] == expected[seed]
        assert len(stream["accepted_tasks"]) == 865
        assert all(
            row["nominal_policy_seed"] == int(seed)
            and row["accepted_task_index"] == index
            for index, row in enumerate(stream["accepted_tasks"])
        )


def test_new_sets_are_frozen_unique_novel_and_current_study_route_clean():
    ledger = split.build_split_ledger()
    assert ledger["current_study_route_clean_verified"] is True
    assert all(ledger["checks"].values())
    assert ledger["evaluation_sha256"] == {
        "diagnostic": (
            "cbd97f1f8f8c6a4d9fcbea1b224cd959c61a8fb680116ababbe9e1889772766c"
        ),
        "validation": (
            "d808fcc361be35f0c585003278f200d4013cc03718b0ceb318fbe6d55195a159"
        ),
    }
    sets = split.evaluation_sets()
    assert len(sets["diagnostic"]["accepted_tasks"]) == 18
    assert len(sets["validation"]["accepted_tasks"]) == 18
    assert [
        row["numeric_seed"] for row in sets["diagnostic"]["rejected_candidates"]
    ] == [9600002, 9600008, 9600016, 9600018]
    assert [
        row["numeric_seed"] for row in sets["validation"]["rejected_candidates"]
    ] == [9700016]
    assert ledger["paper"] == {
        "state": "SEALED_UNINSPECTED",
        "numeric_exclusion_only": list(range(4000, 4018)),
        "content_fingerprints_constructed": 0,
    }
