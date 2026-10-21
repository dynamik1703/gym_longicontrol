"""Fail-closed execution plane for a preparation-only study."""

from __future__ import annotations

import json
import os
import socket
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANONICAL = HERE / "canonical.json"
RECOVERY_INTERVAL_NATIVE_STEPS = 10_000
SCIENTIFIC_INTERVAL_NATIVE_STEPS = 50_000


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def authorization() -> dict[str, bool]:
    return dict(json.loads(CANONICAL.read_text(encoding="utf-8"))["authorization"])


def require_main_training_authorized() -> None:
    flags = authorization()
    if not flags.get("main_training_authorized", False) or not flags.get(
        "main_training_enabled", False
    ):
        raise PermissionError("SLACKTRADER_MAIN_TRAINING_NOT_AUTHORIZED")


def new_attempt_manifest(training_seed: int, output: Path) -> dict[str, Any]:
    """Create an immutable prepared Attempt ID without launching training."""

    if training_seed not in (11, 29, 47):
        raise ValueError("seed is outside the frozen training matrix")
    attempt = {
        "attempt_id": str(uuid.uuid4()),
        "training_seed": int(training_seed),
        "state": "PREPARED_NOT_LAUNCHED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "main_training_authorized": False,
        "main_training_enabled": False,
    }
    destination = output / "attempt.json"
    if destination.exists():
        raise FileExistsError("immutable Attempt manifest already exists")
    _atomic_json(destination, attempt)
    return attempt


class RunLease(AbstractContextManager["RunLease"]):
    """Exclusive single-writer lease with explicit heartbeat state."""

    def __init__(self, output: Path) -> None:
        self.path = Path(output) / "execution.lease"
        self._descriptor: int | None = None

    def __enter__(self) -> RunLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise RuntimeError("duplicate SlackTrader execution lease") from error
        self.heartbeat(0)
        return self

    def heartbeat(self, native_transition_count: int) -> None:
        if self._descriptor is None:
            raise RuntimeError("lease is not active")
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "native_transition_count": int(native_transition_count),
                "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        os.ftruncate(self._descriptor, 0)
        os.write(self._descriptor, payload)
        os.fsync(self._descriptor)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        self.path.unlink(missing_ok=True)


class ValidationLedger:
    """Persistent exactly-once state machine, prepared but never claimed in V1."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def initialize(
        self, *, split_sha256: str, identities_sha256: str
    ) -> dict[str, Any]:
        if self.path.exists():
            return self.read()
        payload = {
            "schema_version": 1,
            "state": "PREPARED_UNOPENED",
            "split_sha256": str(split_sha256),
            "identities_sha256": str(identities_sha256),
            "attempt_id": None,
            "transaction_count": 0,
            "result_sha256": None,
        }
        _atomic_json(self.path, payload)
        return payload

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def claim(self, attempt_id: str) -> dict[str, Any]:
        payload = self.read()
        if payload["state"] == "RUNNING" and payload["attempt_id"] == attempt_id:
            return payload
        if payload["state"] != "PREPARED_UNOPENED":
            raise RuntimeError("Validation exactly-once transaction already claimed")
        if int(payload["transaction_count"]) != 0:
            raise RuntimeError("Validation transaction count is inconsistent")
        payload.update(
            {
                "state": "RUNNING",
                "attempt_id": str(attempt_id),
                "transaction_count": 1,
            }
        )
        _atomic_json(self.path, payload)
        return payload

    def complete(self, attempt_id: str, result_sha256: str) -> dict[str, Any]:
        payload = self.read()
        if payload["state"] != "RUNNING" or payload["attempt_id"] != attempt_id:
            raise RuntimeError("Validation completion does not own transaction")
        payload.update(
            {"state": "COMPLETE", "result_sha256": str(result_sha256)}
        )
        _atomic_json(self.path, payload)
        return payload
