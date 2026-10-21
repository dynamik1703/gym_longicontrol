from __future__ import annotations

import json

from benchmarks.content_split_audit.streams import PAPER_SEEDS
from benchmarks.hybrid_rl_validation_v2.constants import (
    SPLIT_LEDGER,
    V1_ORDERED_SHA256,
    V2_ID,
)
from benchmarks.hybrid_rl_validation_v2.split import build_split_ledger


def test_v2_split_regenerates_exactly_without_performance_evaluation():
    stored = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    assert build_split_ledger() == stored
    assert stored["id"] == V2_ID
    assert stored["state"] == "FROZEN_UNOPENED"
    assert stored["performance_episode_count"] == 0
    assert len(stored["accepted_tasks"]) == 36
    assert stored["route_clean_verified"]
    assert all(stored["checks"].values())


def test_v2_is_unique_and_all_v1_tasks_are_protected():
    stored = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    accepted = stored["accepted_tasks"]
    assert len({row["numeric_seed"] for row in accepted}) == 36
    assert len({row["route_sha256"] for row in accepted}) == 36
    assert len({row["full_task_sha256"] for row in accepted}) == 36
    assert stored["historical_v1"]["ordered_sha256"] == V1_ORDERED_SHA256
    assert stored["historical_v1"]["all_36_tasks_protected"]
    assert stored["protected_sources"]["hybrid_validation_v1_rows"] == 36


def test_all_candidates_and_rejection_reasons_are_persisted():
    stored = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    scanned = stored["scanned_candidates"]
    rejected = stored["rejected_candidates"]
    assert [row for row in scanned if not row["accepted"]] == rejected
    assert all(row["rejection_reasons"] for row in rejected)
    assert all("canonical_task_serialization" in row for row in scanned)


def test_paper_remains_numeric_only_and_unconstructed():
    paper = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))["paper"]
    assert paper == {
        "state": "SEALED_UNINSPECTED",
        "numeric_exclusion_only": list(PAPER_SEEDS),
        "content_fingerprints_constructed": 0,
        "episode_count": 0,
    }
