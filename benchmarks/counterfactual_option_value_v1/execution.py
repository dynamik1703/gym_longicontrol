"""Atomic files, exclusive ownership and immutable branch identities."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ExclusiveLease(AbstractContextManager["ExclusiveLease"]):
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    def __enter__(self) -> ExclusiveLease:
        try:
            self._descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise RuntimeError("duplicate scientific writer") from error
        self.heartbeat("CLAIMED")
        return self

    def heartbeat(self, identity: str) -> None:
        if self._descriptor is None:
            raise RuntimeError("lease is inactive")
        encoded = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "identity": identity,
                "utc": utc_now(),
            },
            sort_keys=True,
        ).encode("utf-8")
        os.lseek(self._descriptor, 0, os.SEEK_SET)
        os.ftruncate(self._descriptor, 0)
        os.write(self._descriptor, encoded)
        os.fsync(self._descriptor)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        self.path.unlink(missing_ok=True)
