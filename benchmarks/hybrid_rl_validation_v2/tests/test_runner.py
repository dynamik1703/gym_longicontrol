from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import pytest

from benchmarks.hybrid_rl_validation_v2.constants import (
    CONTROLLER_ORDER,
    V2_ID,
    V2_TRANSACTION_ID,
)
from benchmarks.hybrid_rl_validation_v2.lease import TransactionLease
from benchmarks.hybrid_rl_validation_v2.ledger import (
    atomic_json,
    build_pair_ledger,
    canonical_sha256,
)
from benchmarks.hybrid_rl_validation_v2.validation_runner import (
    execute_next_registered_pair,
    execute_registered_transaction,
    validate_and_enrich_outcome,
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


def synthetic_ledger():
    ledger = build_pair_ledger(
        synthetic_split(), controller_identities=synthetic_controllers()
    )
    ledger["run_transaction_id"] = V2_TRANSACTION_ID
    for pair in ledger["pairs"]:
        pair["identity"]["transaction_id"] = V2_TRANSACTION_ID
        pair["pair_sha256"] = canonical_sha256(pair["identity"])
    return ledger


def authorization(*, allowed: bool):
    return {
        "study": V2_ID,
        "v1_state": "INCIDENT_ABORTED",
        "preparation_lock": not allowed,
        "v2_validation_authorized": allowed,
        "v2_validation_enabled": allowed,
    }


def synthetic_outcome():
    return {
        "completed": True,
        "travel_time_s": 100.0,
        "energy_kwh": 0.2,
        "speed_violation_count": 0,
        "max_speed_violation_m_s": 0.0,
        "integrated_speed_violation_m": 0.0,
        "feasible": True,
        "feasible_energy_kwh": 0.2,
        "deadline_compliant": True,
        "speed_compliant": True,
        "macro_decision_count": 1,
        "target_action": [0.0],
        "target_speed_m_s": [10.0],
        "proposal_count": 1,
        "override_count": 0,
        "override_fraction": 0.0,
        "native_action_influence_count": 1,
        "native_action_influence_fraction": 1.0,
        "intervention_reasons": {"ACCEPTED": 1},
        "deadline_bound_falsifications": [],
    }


def write_fixture(directory: Path, *, allowed: bool = True):
    ledger_path = directory / "ledger.json"
    authorization_path = directory / "authorization.json"
    lease_path = directory / "lease.json"
    atomic_json(ledger_path, synthetic_ledger())
    atomic_json(authorization_path, authorization(allowed=allowed))
    return ledger_path, authorization_path, lease_path


def _competing_worker(
    ledger_name: str,
    authorization_name: str,
    lease_name: str,
    ready,
    start,
    results,
):
    ready.put("ready")
    start.wait()
    calls = 0

    def evaluator(_identity, _context):
        nonlocal calls
        calls += 1
        return synthetic_outcome()

    try:
        execute_registered_transaction(
            ledger_path=Path(ledger_name),
            authorization_path=Path(authorization_name),
            lease_path=Path(lease_name),
            preflight=lambda: {"fixture": True},
            evaluator=evaluator,
        )
    except RuntimeError as error:
        results.put(("blocked", calls, str(error)))
    else:
        results.put(("completed", calls, None))


def test_two_competing_processes_execute_each_pair_only_once(tmp_path):
    ledger_path, authorization_path, lease_path = write_fixture(tmp_path)
    context = multiprocessing.get_context("fork")
    ready = context.Queue()
    start = context.Event()
    results = context.Queue()
    arguments = (
        str(ledger_path),
        str(authorization_path),
        str(lease_path),
        ready,
        start,
        results,
    )
    processes = [
        context.Process(target=_competing_worker, args=arguments) for _ in range(2)
    ]
    for process in processes:
        process.start()
    assert ready.get(timeout=10) == "ready"
    assert ready.get(timeout=10) == "ready"
    start.set()
    observed = [results.get(timeout=30), results.get(timeout=30)]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    assert sorted((state, calls) for state, calls, _ in observed) == [
        ("blocked", 0),
        ("completed", 180),
    ]
    persisted = json.loads(ledger_path.read_text())
    assert persisted["performance_episode_count"] == 180
    assert {pair["state"] for pair in persisted["pairs"]} == {"COMPLETED"}


def test_losing_lease_owner_cannot_enter_evaluator(tmp_path):
    ledger_path, authorization_path, lease_path = write_fixture(tmp_path)
    calls = 0

    def evaluator(_identity, _context):
        nonlocal calls
        calls += 1
        return synthetic_outcome()

    with TransactionLease(lease_path, V2_TRANSACTION_ID):
        with pytest.raises(RuntimeError, match="stale takeover is forbidden"):
            with TransactionLease(lease_path, V2_TRANSACTION_ID) as losing_lease:
                execute_next_registered_pair(
                    ledger_path=ledger_path,
                    authorization_path=authorization_path,
                    lease=losing_lease,
                    context={},
                    evaluator=evaluator,
                )
    assert calls == 0


def test_direct_runner_entry_cannot_bypass_lease(tmp_path):
    ledger_path, authorization_path, _ = write_fixture(tmp_path)
    calls = 0

    def evaluator(_identity, _context):
        nonlocal calls
        calls += 1
        return synthetic_outcome()

    with pytest.raises(PermissionError, match="requires lease ownership"):
        execute_next_registered_pair(
            ledger_path=ledger_path,
            authorization_path=authorization_path,
            lease=None,
            context={},
            evaluator=evaluator,
        )
    assert calls == 0


def test_started_is_durable_before_callback_and_failure_is_terminal(tmp_path):
    ledger_path, authorization_path, lease_path = write_fixture(tmp_path)
    calls = 0

    def evaluator(identity, _context):
        nonlocal calls
        calls += 1
        persisted = json.loads(ledger_path.read_text())
        pair = persisted["pairs"][identity["pair_ordinal"]]
        assert pair["state"] == "STARTED"
        raise RuntimeError("injected after durable STARTED")

    with pytest.raises(RuntimeError, match="injected"):
        execute_registered_transaction(
            ledger_path=ledger_path,
            authorization_path=authorization_path,
            lease_path=lease_path,
            preflight=lambda: {"valid": True},
            evaluator=evaluator,
        )
    persisted = json.loads(ledger_path.read_text())
    assert persisted["state"] == "INTEGRITY_STOPPED"
    assert persisted["pairs"][0]["state"] == "INTERRUPTED_UNRECOVERABLE"
    assert json.loads(lease_path.read_text())["state"] == "ACTIVE"
    with pytest.raises(RuntimeError, match="stale takeover is forbidden"):
        execute_registered_transaction(
            ledger_path=ledger_path,
            authorization_path=authorization_path,
            lease_path=lease_path,
            preflight=lambda: {"valid": True},
            evaluator=evaluator,
        )
    assert calls == 1


def test_failed_preflight_leaves_every_pair_pending_and_no_lease(tmp_path):
    ledger_path, authorization_path, lease_path = write_fixture(tmp_path)

    def invalid_preflight():
        raise RuntimeError("checkpoint unavailable")

    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        execute_registered_transaction(
            ledger_path=ledger_path,
            authorization_path=authorization_path,
            lease_path=lease_path,
            preflight=invalid_preflight,
            evaluator=lambda _identity, _context: synthetic_outcome(),
        )
    persisted = json.loads(ledger_path.read_text())
    assert {pair["state"] for pair in persisted["pairs"]} == {"PENDING"}
    assert not lease_path.exists()


def test_stale_lease_is_never_silently_reclaimed(tmp_path):
    _, _, lease_path = write_fixture(tmp_path)
    with TransactionLease(lease_path, V2_TRANSACTION_ID):
        pass
    assert json.loads(lease_path.read_text())["state"] == "ACTIVE"
    with pytest.raises(RuntimeError, match="stale takeover is forbidden"):
        with TransactionLease(lease_path, V2_TRANSACTION_ID):
            pass
    assert json.loads(lease_path.read_text())["state"] == "ACTIVE"


def test_authorization_denial_precedes_preflight_or_evaluator(tmp_path):
    ledger_path, authorization_path, lease_path = write_fixture(tmp_path, allowed=False)
    calls = {"preflight": 0, "evaluator": 0}

    def preflight():
        calls["preflight"] += 1

    def evaluator(_identity, _context):
        calls["evaluator"] += 1
        return synthetic_outcome()

    with pytest.raises(PermissionError, match="preparation lock"):
        execute_registered_transaction(
            ledger_path=ledger_path,
            authorization_path=authorization_path,
            lease_path=lease_path,
            preflight=preflight,
            evaluator=evaluator,
        )
    assert calls == {"preflight": 0, "evaluator": 0}
    assert not lease_path.exists()


def test_result_validation_adds_required_provenance_without_changing_metrics():
    identity = synthetic_ledger()["pairs"][0]["identity"]
    raw = synthetic_outcome()
    enriched = validate_and_enrich_outcome(identity, raw)
    assert all(enriched[key] == value for key, value in raw.items())
    assert enriched["exclusive_failure_mode"] == "feasible"
    assert enriched["final_position_m"] == 1000.0
    assert enriched["controller_id"] == identity["controller_id"]
    assert enriched["numeric_seed"] == identity["numeric_seed"]
