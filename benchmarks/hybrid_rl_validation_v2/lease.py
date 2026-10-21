"""Durable, non-reclaimable single-writer lease for Validation V2."""

from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .constants import V2_ID, V2_TRANSACTION_ID
from .ledger import atomic_json, load_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class TransactionLease:
    """An exclusive file that is never deleted or automatically reclaimed."""

    def __init__(self, path: Path, transaction_id: str) -> None:
        if transaction_id != V2_TRANSACTION_ID:
            raise PermissionError("replacement V2 transaction IDs are forbidden")
        self.path = path
        self.transaction_id = transaction_id
        self.owner_token = uuid.uuid4().hex
        self.pid = os.getpid()
        self.host = socket.gethostname()
        self.acquired_at_utc: str | None = None
        self._entered = False

    def __enter__(self) -> TransactionLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        acquired_at = utc_now()
        record = {
            "schema_version": 1,
            "study": V2_ID,
            "transaction_id": self.transaction_id,
            "state": "ACTIVE",
            "pid": self.pid,
            "host": self.host,
            "owner_token": self.owner_token,
            "acquired_at_utc": acquired_at,
            "acquisition": {
                "mechanism": "os.open(O_CREAT|O_EXCL)",
                "file_fsync": True,
                "directory_fsync": True,
                "automatic_stale_reclaim": False,
            },
            "completed_at_utc": None,
            "completed_pair_count": None,
        }
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            raise RuntimeError(
                "V2 transaction lease already exists; stale takeover is forbidden"
            ) from error
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(record, stream, allow_nan=False, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(self.path.parent)
        except BaseException:
            # The exclusive inode intentionally remains as a fail-closed record.
            raise
        self.acquired_at_utc = acquired_at
        self._entered = True
        self.assert_owned()
        return self

    def assert_owned(self) -> dict[str, Any]:
        if not self._entered:
            raise PermissionError("V2 evaluator requires an entered transaction lease")
        try:
            record = load_json(self.path)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise PermissionError("V2 lease record is missing or incomplete") from error
        expected = {
            "study": V2_ID,
            "transaction_id": self.transaction_id,
            "state": "ACTIVE",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "owner_token": self.owner_token,
            "acquired_at_utc": self.acquired_at_utc,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise PermissionError("current process does not own the active V2 lease")
        return record

    def seal(self, *, completed_pair_count: int) -> dict[str, Any]:
        record = self.assert_owned()
        if completed_pair_count != 180:
            raise RuntimeError("V2 lease may seal only after all 180 pairs complete")
        record["state"] = "COMPLETED_SEALED"
        record["completed_at_utc"] = utc_now()
        record["completed_pair_count"] = completed_pair_count
        atomic_json(self.path, record)
        persisted = load_json(self.path)
        if persisted != record:
            raise RuntimeError("sealed V2 lease did not persist exactly")
        self._entered = False
        return persisted

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        # Never unlink: an exception or process death leaves a permanent blocker.
        self._entered = False
        return False
