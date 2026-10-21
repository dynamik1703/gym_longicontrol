from __future__ import annotations

import json

import pytest

from benchmarks.slacktrader_v1 import evaluation
from benchmarks.slacktrader_v1.execution import (
    RunLease,
    ValidationLedger,
    new_attempt_manifest,
    require_main_training_authorized,
)
from benchmarks.slacktrader_v1.integrity import verify_flags, verify_split_ledger
from benchmarks.slacktrader_v1.split import build_split_ledger


def test_route_full_task_and_paper_separation_regenerate_exactly() -> None:
    stored = verify_split_ledger()
    regenerated = build_split_ledger()
    assert regenerated == stored
    assert all(stored["checks"].values())
    assert stored["evaluation_sets"]["validation"]["performance_episode_count"] == 0
    assert stored["paper"] == {
        "state": "SEALED_UNINSPECTED",
        "numeric_exclusion_only": list(range(4000, 4018)),
        "content_fingerprints_constructed": 0,
        "episode_count": 0,
    }


def test_validation_paper_and_training_fail_closed_before_environment_reset(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        evaluation,
        "_environment",
        lambda: (_ for _ in ()).throw(AssertionError("environment was constructed")),
    )
    with pytest.raises(PermissionError, match="VALIDATION_FROZEN_UNOPENED"):
        evaluation.evaluate_validation()
    with pytest.raises(PermissionError, match="PAPER_TEST_SEALED"):
        evaluation.evaluate_price_seed(numeric_seed=4000, selector=None)  # type: ignore[arg-type]
    with pytest.raises(PermissionError, match="MAIN_TRAINING_NOT_AUTHORIZED"):
        require_main_training_authorized()
    flags = verify_flags()
    assert not any(flags.values())


def test_immutable_attempt_and_exclusive_single_writer_lease(tmp_path) -> None:
    attempt = new_attempt_manifest(11, tmp_path)
    assert attempt["state"] == "PREPARED_NOT_LAUNCHED"
    assert json.loads((tmp_path / "attempt.json").read_text())["attempt_id"] == (
        attempt["attempt_id"]
    )
    with pytest.raises(FileExistsError):
        new_attempt_manifest(11, tmp_path)
    with RunLease(tmp_path) as lease:
        lease.heartbeat(7)
        with pytest.raises(RuntimeError, match="duplicate"):
            with RunLease(tmp_path):
                pass


def test_validation_ledger_is_exactly_once_and_same_attempt_resumable(tmp_path) -> None:
    ledger = ValidationLedger(tmp_path / "validation-ledger.json")
    prepared = ledger.initialize(split_sha256="split", identities_sha256="arms")
    assert prepared["state"] == "PREPARED_UNOPENED"
    claimed = ledger.claim("ATTEMPT-A")
    assert claimed["transaction_count"] == 1
    assert ledger.claim("ATTEMPT-A") == claimed
    with pytest.raises(RuntimeError, match="already claimed"):
        ledger.claim("ATTEMPT-B")
    complete = ledger.complete("ATTEMPT-A", "result")
    assert complete["state"] == "COMPLETE"
    with pytest.raises(RuntimeError, match="already claimed"):
        ledger.claim("ATTEMPT-A")
