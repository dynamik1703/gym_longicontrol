from __future__ import annotations

import json

import pytest

from benchmarks.hybrid_rl_validation_v2.constants import (
    AUTHORIZATION,
    CONTROLLER_ORDER,
    PAIR_LEDGER,
    V2_ID,
)
from benchmarks.hybrid_rl_validation_v2.ledger import (
    atomic_json,
    build_pair_ledger,
    canonical_sha256,
)
from benchmarks.hybrid_rl_validation_v2.validation_runner import (
    execute_next_registered_pair,
)


def synthetic_split():
    tasks = [
        {
            "numeric_seed": 60_000_000 + index,
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


def authorization(*, allowed: bool):
    return {
        "study": V2_ID,
        "v1_state": "INCIDENT_ABORTED",
        "preparation_lock": not allowed,
        "v2_validation_authorized": allowed,
        "v2_validation_enabled": allowed,
    }


def test_real_v2_is_denied_before_evaluator_entry():
    calls = {"preflight": 0, "evaluator": 0}

    def preflight():
        calls["preflight"] += 1

    def evaluator(_identity, _context):
        calls["evaluator"] += 1
        return {}

    with pytest.raises(PermissionError, match="preparation lock"):
        execute_next_registered_pair(
            ledger_path=PAIR_LEDGER,
            authorization_path=AUTHORIZATION,
            preflight=preflight,
            evaluator=evaluator,
        )
    assert calls == {"preflight": 0, "evaluator": 0}
    assert json.loads(PAIR_LEDGER.read_text())["performance_episode_count"] == 0


def test_invalid_policy_fails_before_evaluator_entry(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    authorization_path = tmp_path / "authorization.json"
    atomic_json(
        ledger_path,
        build_pair_ledger(
            synthetic_split(), controller_identities=synthetic_controllers()
        ),
    )
    atomic_json(authorization_path, authorization(allowed=True))
    calls = {"evaluator": 0}

    def invalid_policy():
        raise RuntimeError("checkpoint unavailable")

    def evaluator(_identity, _context):
        calls["evaluator"] += 1
        return {}

    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        execute_next_registered_pair(
            ledger_path=ledger_path,
            authorization_path=authorization_path,
            preflight=invalid_policy,
            evaluator=evaluator,
        )
    assert calls["evaluator"] == 0
    assert json.loads(ledger_path.read_text())["pairs"][0]["state"] == "PENDING"


def test_started_is_durable_before_callback_and_failure_is_terminal(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    authorization_path = tmp_path / "authorization.json"
    atomic_json(
        ledger_path,
        build_pair_ledger(
            synthetic_split(), controller_identities=synthetic_controllers()
        ),
    )
    atomic_json(authorization_path, authorization(allowed=True))
    calls = {"evaluator": 0}

    def evaluator(identity, _context):
        calls["evaluator"] += 1
        persisted = json.loads(ledger_path.read_text())
        pair = persisted["pairs"][identity["pair_ordinal"]]
        assert pair["state"] == "STARTED"
        raise RuntimeError("injected after durable STARTED")

    with pytest.raises(RuntimeError, match="injected"):
        execute_next_registered_pair(
            ledger_path=ledger_path,
            authorization_path=authorization_path,
            preflight=lambda: {"valid": True},
            evaluator=evaluator,
        )
    persisted = json.loads(ledger_path.read_text())
    assert persisted["state"] == "INTEGRITY_STOPPED"
    assert persisted["pairs"][0]["state"] == "INTERRUPTED_UNRECOVERABLE"
    with pytest.raises(RuntimeError, match="integrity stop"):
        execute_next_registered_pair(
            ledger_path=ledger_path,
            authorization_path=authorization_path,
            preflight=lambda: {"valid": True},
            evaluator=evaluator,
        )
    assert calls["evaluator"] == 1
