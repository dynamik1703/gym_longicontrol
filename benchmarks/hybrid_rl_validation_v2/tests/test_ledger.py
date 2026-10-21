from __future__ import annotations

import pytest

from benchmarks.hybrid_rl_validation_v2.constants import CONTROLLER_ORDER, V2_ID
from benchmarks.hybrid_rl_validation_v2.ledger import (
    aggregate_completed,
    atomic_json,
    build_pair_ledger,
    canonical_sha256,
    mark_completed,
    mark_started,
    validate_pair_ledger,
    validate_registration,
)


def synthetic_split():
    tasks = [
        {
            "numeric_seed": 50_000_000 + index,
            "route_sha256": f"route-{index:02d}",
            "full_task_sha256": f"full-{index:02d}",
        }
        for index in range(36)
    ]
    return {
        "id": V2_ID,
        "ordered_sha256": canonical_sha256(tasks),
        "accepted_tasks": tasks,
    }


def synthetic_controllers():
    return {controller_id: {"fixture": True} for controller_id in CONTROLLER_ORDER}


def test_registration_contains_180_unique_pending_pairs():
    ledger = build_pair_ledger(
        synthetic_split(), controller_identities=synthetic_controllers()
    )
    validate_pair_ledger(ledger)
    assert len(ledger["pairs"]) == 180
    assert {pair["state"] for pair in ledger["pairs"]} == {"PENDING"}
    identities = {
        (pair["identity"]["controller_id"], pair["identity"]["numeric_seed"])
        for pair in ledger["pairs"]
    }
    assert len(identities) == 180
    validate_registration(
        ledger,
        synthetic_split(),
        controller_identities=synthetic_controllers(),
    )


def test_registration_tampering_is_rejected():
    ledger = build_pair_ledger(
        synthetic_split(), controller_identities=synthetic_controllers()
    )
    ledger["pairs"][0]["identity"]["numeric_seed"] += 1
    ledger["pairs"][0]["pair_sha256"] = canonical_sha256(
        ledger["pairs"][0]["identity"]
    )
    with pytest.raises(RuntimeError, match="identities or order"):
        validate_registration(
            ledger,
            synthetic_split(),
            controller_identities=synthetic_controllers(),
        )


def test_completed_pair_can_never_start_again(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = build_pair_ledger(
        synthetic_split(), controller_identities=synthetic_controllers()
    )
    atomic_json(path, ledger)
    pair_hash = ledger["pairs"][0]["pair_sha256"]
    mark_started(path, pair_hash, started_at_utc="fixture-start")
    mark_completed(
        path,
        pair_hash,
        result={"feasible": True, "energy_kwh": 1.0},
        completed_at_utc="fixture-complete",
    )
    with pytest.raises(RuntimeError, match="exactly once"):
        mark_started(path, pair_hash, started_at_utc="forbidden-repeat")


def test_aggregation_only_recovery_invokes_no_evaluator(tmp_path):
    path = tmp_path / "completed.json"
    ledger = build_pair_ledger(
        synthetic_split(), controller_identities=synthetic_controllers()
    )
    for pair in ledger["pairs"]:
        result = {"identity": pair["identity"], "outcome": {"fixture": True}}
        pair["state"] = "COMPLETED"
        pair["started_at_utc"] = "fixture-start"
        pair["completed_at_utc"] = "fixture-complete"
        pair["result"] = result
        pair["result_sha256"] = canonical_sha256(result)
    ledger["performance_episode_count"] = 180
    ledger["state"] = "COMPLETED_SEALED"
    ledger["execution_opened"] = True
    atomic_json(path, ledger)
    recovery = aggregate_completed(path)
    assert recovery["pair_count"] == 180
    assert recovery["evaluation_calls"] == 0


def test_v1_ledger_is_permanently_rejected():
    ledger = build_pair_ledger(
        synthetic_split(), controller_identities=synthetic_controllers()
    )
    ledger["study"] = "HYBRID_RL_VALIDATION_V1"
    with pytest.raises(PermissionError, match="permanently blocked"):
        validate_pair_ledger(ledger)
