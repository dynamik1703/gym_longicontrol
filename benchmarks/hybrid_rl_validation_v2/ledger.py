"""Durable exactly-once pair ledger for the unopened V2 panel."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .constants import CONTROLLER_ORDER, V2_ID, V2_STATE

PAIR_STATES = frozenset(
    {"PENDING", "STARTED", "COMPLETED", "INTERRUPTED_UNRECOVERABLE"}
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Atomically replace JSON and fsync both file and containing directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pair_ledger(
    split: Mapping[str, Any],
    *,
    controller_identities: Mapping[str, Any],
) -> dict[str, Any]:
    if split["id"] != V2_ID or len(split["accepted_tasks"]) != 36:
        raise ValueError("V2 pair registration requires the frozen 36-task split")
    if tuple(controller_identities) != CONTROLLER_ORDER:
        raise ValueError("controller identities or order changed")
    transaction_basis = {
        "study": V2_ID,
        "split_sha256": split["ordered_sha256"],
        "controllers": list(CONTROLLER_ORDER),
        "controller_identities": dict(controller_identities),
    }
    transaction_id = canonical_sha256(transaction_basis)
    pairs = []
    for controller_id in CONTROLLER_ORDER:
        for task_ordinal, row in enumerate(split["accepted_tasks"]):
            identity = {
                "transaction_id": transaction_id,
                "pair_ordinal": len(pairs),
                "task_ordinal": task_ordinal,
                "controller_id": controller_id,
                "numeric_seed": int(row["numeric_seed"]),
                "route_sha256": row["route_sha256"],
                "full_task_sha256": row["full_task_sha256"],
            }
            pairs.append(
                {
                    "identity": identity,
                    "pair_sha256": canonical_sha256(identity),
                    "state": "PENDING",
                    "started_at_utc": None,
                    "completed_at_utc": None,
                    "interruption": None,
                    "result": None,
                    "result_sha256": None,
                }
            )
    identities = {
        (pair["identity"]["controller_id"], pair["identity"]["numeric_seed"])
        for pair in pairs
    }
    if len(pairs) != 180 or len(identities) != 180:
        raise RuntimeError("V2 registration is not exactly 180 unique pairs")
    return {
        "schema_version": 1,
        "study": V2_ID,
        "state": V2_STATE,
        "run_transaction_id": transaction_id,
        "split_sha256": split["ordered_sha256"],
        "controller_order": list(CONTROLLER_ORDER),
        "controller_identities": dict(controller_identities),
        "planned_pair_count": 180,
        "performance_episode_count": 0,
        "registration_complete": True,
        "execution_opened": False,
        "pairs": pairs,
    }


def validate_pair_ledger(ledger: Mapping[str, Any]) -> None:
    if ledger.get("study") != V2_ID:
        raise PermissionError("V1 and non-V2 ledgers are permanently blocked")
    pairs = ledger.get("pairs", [])
    if len(pairs) != 180 or ledger.get("planned_pair_count") != 180:
        raise RuntimeError("V2 ledger pair count changed")
    pair_hashes: set[str] = set()
    identities: set[tuple[str, int]] = set()
    completed = 0
    for ordinal, pair in enumerate(pairs):
        identity = pair["identity"]
        if identity["pair_ordinal"] != ordinal:
            raise RuntimeError("V2 pair order changed")
        expected_pair_hash = canonical_sha256(identity)
        if pair["pair_sha256"] != expected_pair_hash:
            raise RuntimeError("V2 pair identity hash changed")
        if pair["state"] not in PAIR_STATES:
            raise RuntimeError("unknown V2 pair state")
        pair_hashes.add(pair["pair_sha256"])
        identities.add((identity["controller_id"], int(identity["numeric_seed"])))
        if pair["state"] == "COMPLETED":
            completed += 1
            if pair["result"] is None:
                raise RuntimeError("completed V2 pair has no result")
            if canonical_sha256(pair["result"]) != pair["result_sha256"]:
                raise RuntimeError("completed V2 result hash changed")
        elif pair["result"] is not None or pair["result_sha256"] is not None:
            raise RuntimeError("non-completed V2 pair contains a result")
    if len(pair_hashes) != 180 or len(identities) != 180:
        raise RuntimeError("duplicate V2 controller-task pair")
    if ledger.get("performance_episode_count") != completed:
        raise RuntimeError("V2 performance episode counter changed")


def validate_registration(
    ledger: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    controller_identities: Mapping[str, Any],
) -> None:
    """Bind every immutable pair identity to the frozen split and controllers."""

    expected = build_pair_ledger(
        split,
        controller_identities=controller_identities,
    )
    fields = (
        "study",
        "run_transaction_id",
        "split_sha256",
        "controller_order",
        "controller_identities",
        "planned_pair_count",
    )
    if any(ledger.get(field) != expected[field] for field in fields):
        raise RuntimeError("V2 registration provenance changed")
    observed_pairs = [
        (pair["identity"], pair["pair_sha256"]) for pair in ledger["pairs"]
    ]
    expected_pairs = [
        (pair["identity"], pair["pair_sha256"]) for pair in expected["pairs"]
    ]
    if observed_pairs != expected_pairs:
        raise RuntimeError("V2 planned pair identities or order changed")


def _pair_by_hash(ledger: dict[str, Any], pair_sha256: str) -> dict[str, Any]:
    matches = [pair for pair in ledger["pairs"] if pair["pair_sha256"] == pair_sha256]
    if len(matches) != 1:
        raise KeyError("pair is not registered exactly once")
    return matches[0]


def mark_started(
    path: Path,
    pair_sha256: str,
    *,
    started_at_utc: str,
) -> dict[str, Any]:
    ledger = load_json(path)
    validate_pair_ledger(ledger)
    if any(
        pair["state"] in {"STARTED", "INTERRUPTED_UNRECOVERABLE"}
        for pair in ledger["pairs"]
    ):
        raise RuntimeError("an opened or interrupted V2 pair requires investigation")
    pair = _pair_by_hash(ledger, pair_sha256)
    if pair["state"] != "PENDING":
        raise RuntimeError("V2 pair may start exactly once")
    pair["state"] = "STARTED"
    pair["started_at_utc"] = started_at_utc
    ledger["state"] = "OPENED_IN_PROGRESS"
    ledger["execution_opened"] = True
    atomic_json(path, ledger)
    persisted = load_json(path)
    persisted_pair = _pair_by_hash(persisted, pair_sha256)
    if persisted_pair["state"] != "STARTED":
        raise RuntimeError("STARTED was not durably persisted")
    return persisted_pair


def mark_completed(
    path: Path,
    pair_sha256: str,
    *,
    result: Mapping[str, Any],
    completed_at_utc: str,
) -> dict[str, Any]:
    if not _finite(result):
        raise FloatingPointError("V2 results must contain only finite values")
    ledger = load_json(path)
    validate_pair_ledger(ledger)
    pair = _pair_by_hash(ledger, pair_sha256)
    if pair["state"] != "STARTED":
        raise RuntimeError("only a durably STARTED V2 pair can complete")
    frozen_result = {"identity": pair["identity"], "outcome": dict(result)}
    pair["result"] = frozen_result
    pair["result_sha256"] = canonical_sha256(frozen_result)
    pair["state"] = "COMPLETED"
    pair["completed_at_utc"] = completed_at_utc
    ledger["performance_episode_count"] = sum(
        item["state"] == "COMPLETED" for item in ledger["pairs"]
    )
    if ledger["performance_episode_count"] == 180:
        ledger["state"] = "COMPLETED_SEALED"
    atomic_json(path, ledger)
    return pair


def mark_interrupted(
    path: Path,
    pair_sha256: str,
    *,
    reason: str,
    interrupted_at_utc: str,
) -> dict[str, Any]:
    ledger = load_json(path)
    validate_pair_ledger(ledger)
    pair = _pair_by_hash(ledger, pair_sha256)
    if pair["state"] != "STARTED":
        raise RuntimeError("only a STARTED V2 pair can be interrupted")
    pair["state"] = "INTERRUPTED_UNRECOVERABLE"
    pair["interruption"] = {
        "reason": reason,
        "interrupted_at_utc": interrupted_at_utc,
    }
    ledger["state"] = "INTEGRITY_STOPPED"
    atomic_json(path, ledger)
    return pair


def aggregate_completed(path: Path) -> dict[str, Any]:
    """Recover aggregation from durable results without evaluating any pair."""

    ledger = load_json(path)
    validate_pair_ledger(ledger)
    if ledger["state"] != "COMPLETED_SEALED":
        raise RuntimeError("aggregation requires 180 durably completed V2 pairs")
    results = [pair["result"] for pair in ledger["pairs"]]
    return {
        "study": V2_ID,
        "pair_count": len(results),
        "result_set_sha256": canonical_sha256(results),
        "evaluation_calls": 0,
    }


def _finite(value: Any) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True
