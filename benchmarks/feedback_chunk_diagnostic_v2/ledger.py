"""Frozen pair registration and authoritative exactly-once state transitions."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from .constants import (
    C_CHECKPOINTS,
    C_CONTROLLERS,
    CONTROLLER_ORDER,
    H_CHECKPOINTS,
    H_CONTROLLERS,
    NONLEARNING_CONTROLLERS,
    PAIR_COUNT,
    PANEL_ID,
)
from .utils import atomic_json, canonical_sha256, load_json

PAIR_STATES = frozenset({"NOT_STARTED", "STARTED", "COMPLETED"})


def _controller_provenance(controller_id: str) -> dict[str, Any]:
    if controller_id in H_CONTROLLERS:
        checkpoint = H_CHECKPOINTS[controller_id]
        return {
            "worker_runtime": "H",
            "checkpoint_sha256": checkpoint["sha256"],
            "attempt_id": checkpoint["attempt_id"],
        }
    if controller_id in C_CHECKPOINTS:
        checkpoint = C_CHECKPOINTS[controller_id]
        return {
            "worker_runtime": "C",
            "checkpoint_sha256": checkpoint["sha256"],
            "attempt_id": checkpoint["attempt_id"],
        }
    if controller_id in NONLEARNING_CONTROLLERS:
        return {
            "worker_runtime": "C",
            "checkpoint_sha256": None,
            "attempt_id": None,
        }
    raise RuntimeError(f"unregistered controller: {controller_id}")


def build_pair_ledger(split: dict[str, Any]) -> dict[str, Any]:
    tasks = split["panel"]["accepted_tasks"]
    if len(tasks) != 36 or split["panel"]["id"] != PANEL_ID:
        raise RuntimeError("frozen Diagnostic V2 split is invalid")
    pairs = []
    ordinal = 0
    for controller_id in CONTROLLER_ORDER:
        provenance = _controller_provenance(controller_id)
        for task_ordinal, task in enumerate(tasks):
            identity = {
                "pair_ordinal": ordinal,
                "controller_id": controller_id,
                "task_ordinal": task_ordinal,
                "numeric_seed": int(task["numeric_seed"]),
                "route_sha256": task["route_sha256"],
                "full_task_sha256": task["full_task_sha256"],
                **provenance,
            }
            pair_id = canonical_sha256(identity)
            pairs.append(
                {
                    "pair_id": pair_id,
                    "identity": identity,
                    "state": "NOT_STARTED",
                    "ownership_token_sha256": None,
                    "started_at_utc": None,
                    "completed_at_utc": None,
                    "result_path": None,
                    "evidence_path": None,
                    "result_sha256": None,
                    "evidence_sha256": None,
                }
            )
            ordinal += 1
    frozen_schedule = [row["identity"] | {"pair_id": row["pair_id"]} for row in pairs]
    ledger = {
        "schema_version": 1,
        "study": "FEEDBACK_CHUNK_RUNTIME_ISOLATED_DIAGNOSTIC_V2",
        "panel_id": PANEL_ID,
        "pair_count": PAIR_COUNT,
        "schedule_sha256": canonical_sha256(frozen_schedule),
        "state": "FROZEN_UNAUTHORIZED",
        "performance_episode_count": 0,
        "validation_performance_episode_count": 0,
        "paper_performance_episode_count": 0,
        "pairs": pairs,
    }
    validate_pair_ledger(ledger)
    return ledger


def validate_pair_ledger(ledger: dict[str, Any]) -> dict[str, int]:
    pairs = ledger.get("pairs", [])
    if ledger.get("pair_count") != PAIR_COUNT or len(pairs) != PAIR_COUNT:
        raise RuntimeError("pair ledger cardinality changed")
    if [row["identity"]["pair_ordinal"] for row in pairs] != list(range(PAIR_COUNT)):
        raise RuntimeError("pair ledger order changed")
    if len({row["pair_id"] for row in pairs}) != PAIR_COUNT:
        raise RuntimeError("pair IDs are not unique")
    frozen_schedule = []
    for row in pairs:
        if row.get("state") not in PAIR_STATES:
            raise RuntimeError("invalid pair state")
        if row["pair_id"] != canonical_sha256(row["identity"]):
            raise RuntimeError("pair identity hash changed")
        controller = row["identity"]["controller_id"]
        expected_runtime = "H" if controller in H_CONTROLLERS else "C"
        if row["identity"]["worker_runtime"] != expected_runtime:
            raise RuntimeError("pair assigned to incompatible worker")
        if controller not in C_CONTROLLERS | H_CONTROLLERS:
            raise RuntimeError("unknown controller in pair ledger")
        if row["state"] == "NOT_STARTED" and any(
            row[field] is not None
            for field in (
                "ownership_token_sha256",
                "started_at_utc",
                "completed_at_utc",
                "result_sha256",
                "evidence_sha256",
            )
        ):
            raise RuntimeError("untouched pair contains execution evidence")
        if row["state"] == "STARTED" and (
            row["ownership_token_sha256"] is None
            or row["started_at_utc"] is None
            or row["completed_at_utc"] is not None
        ):
            raise RuntimeError("STARTED pair is incomplete")
        if row["state"] == "COMPLETED" and any(
            row[field] is None
            for field in (
                "ownership_token_sha256",
                "started_at_utc",
                "completed_at_utc",
                "result_path",
                "evidence_path",
                "result_sha256",
                "evidence_sha256",
            )
        ):
            raise RuntimeError("COMPLETED pair lacks immutable artifacts")
        frozen_schedule.append(row["identity"] | {"pair_id": row["pair_id"]})
    if canonical_sha256(frozen_schedule) != ledger.get("schedule_sha256"):
        raise RuntimeError("frozen pair schedule hash changed")
    counts = Counter(row["state"] for row in pairs)
    if ledger.get("performance_episode_count") != counts["COMPLETED"]:
        raise RuntimeError("performance episode accounting changed")
    if ledger.get("validation_performance_episode_count") != 0:
        raise RuntimeError("Validation boundary was opened")
    if ledger.get("paper_performance_episode_count") != 0:
        raise RuntimeError("Paper boundary was opened")
    return dict(counts)


def _load_pair(path: Any, pair_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = load_json(path)
    validate_pair_ledger(ledger)
    matches = [row for row in ledger["pairs"] if row["pair_id"] == pair_id]
    if len(matches) != 1:
        raise RuntimeError("pair is absent or duplicated")
    return ledger, matches[0]


def mark_started(
    path: Any, *, pair_id: str, ownership_token: str, started_at_utc: str
) -> dict[str, Any]:
    ledger, row = _load_pair(path, pair_id)
    if row["state"] != "NOT_STARTED":
        raise RuntimeError("exactly-once violation: pair is not fresh")
    row["state"] = "STARTED"
    row["ownership_token_sha256"] = canonical_sha256(ownership_token)
    row["started_at_utc"] = started_at_utc
    ledger["state"] = "RUNNING"
    validate_pair_ledger(ledger)
    atomic_json(path, ledger)
    persisted = load_json(path)
    validate_pair_ledger(persisted)
    return deepcopy(row)


def mark_completed(
    path: Any,
    *,
    pair_id: str,
    ownership_token: str,
    completed_at_utc: str,
    result_path: str,
    evidence_path: str,
    result_sha256: str,
    evidence_sha256: str,
) -> dict[str, Any]:
    ledger, row = _load_pair(path, pair_id)
    if row["state"] != "STARTED":
        raise RuntimeError("only a STARTED pair can complete")
    if row["ownership_token_sha256"] != canonical_sha256(ownership_token):
        raise PermissionError("pair ownership token changed")
    row.update(
        {
            "state": "COMPLETED",
            "completed_at_utc": completed_at_utc,
            "result_path": result_path,
            "evidence_path": evidence_path,
            "result_sha256": result_sha256,
            "evidence_sha256": evidence_sha256,
        }
    )
    ledger["performance_episode_count"] += 1
    if ledger["performance_episode_count"] == PAIR_COUNT:
        ledger["state"] = "COMPLETED"
    validate_pair_ledger(ledger)
    atomic_json(path, ledger)
    persisted = load_json(path)
    validate_pair_ledger(persisted)
    return deepcopy(row)
