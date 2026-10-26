"""Regression tests use retained records only; never reset/rerun a scientific pair."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from benchmarks.bankers_viability_shield_v1.backup_policy import SafetyIntegrityError
from benchmarks.bankers_viability_shield_v1.certificate import digest
from benchmarks.bankers_viability_shield_v1.fixtures import fixtures
from benchmarks.bankers_viability_shield_v1.verification import replay_records

HERE = Path(__file__).resolve().parents[1]


def records():
    with gzip.open(HERE / "evidence/FIXTURE__constant_20.jsonl.gz", "rt") as stream:
        return [json.loads(line) for line in stream]


def test_json_roundtrip_representation_is_content_equivalent():
    original = {"envelope": [(0.0, 20 / 3.6), (150.0, 20 / 3.6)]}
    decoded = json.loads(json.dumps(original))
    assert original != decoded
    assert digest(original) == digest(decoded)


def test_retained_full_native_evidence_independent_replay():
    rows = records()
    result = replay_records(
        rows, fixtures()["constant_20"], "BANKERS_PRE", full_masks=True
    )
    assert result["steps"] == 1800
    assert result["metrics"]["max_speed_violation_m_s"] == 0
    assert not result["feasible"]


def test_missing_native_record_fails_closed():
    rows = records()
    del rows[0]
    with pytest.raises(SafetyIntegrityError, match="missing/duplicate"):
        replay_records(rows, fixtures()["constant_20"], "BANKERS_PRE", full_masks=False)


def test_missing_applicable_deadline_bound_fails_closed():
    rows = records()
    rows[0]["completion_bound_s"] = None
    with pytest.raises(SafetyIntegrityError, match="missing deadline bound"):
        replay_records(rows, fixtures()["constant_20"], "BANKERS_PRE", full_masks=False)


def test_tampered_executed_successor_fails_closed():
    rows = records()
    rows[0] = copy.deepcopy(rows[0])
    rows[0]["actual_next_state"]["velocity_m_s"] += 1e-10
    with pytest.raises(SafetyIntegrityError, match="actual successor"):
        replay_records(rows, fixtures()["constant_20"], "BANKERS_PRE", full_masks=False)


def test_recorded_integrity_stop_blocks_development():
    from benchmarks.bankers_viability_shield_v1.evaluation import develop

    with pytest.raises(SafetyIntegrityError, match="recorded integrity stop"):
        develop()
    assert not (HERE / "pair_ledger.json").exists()
    assert not (HERE / "execution_manifest.json").exists()
