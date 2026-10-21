"""Single-writer exactly-once execution state for OPTIONS_CAPABILITY_V1."""

from __future__ import annotations

import json
import os
import socket
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .integrity import HERE, PAIR_LEDGER
from .split import canonical_sha256

CONTROLLER_ORDER = (
    "FROZEN_MPC_V2",
    "RESET_FEEDBACK",
    "CONTINUE_FEEDBACK",
    "COMMIT_FEEDBACK",
)


def atomic_json(path: Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


class CapabilityLease(AbstractContextManager["CapabilityLease"]):
    def __init__(self, path: Path = HERE / "capability.lease") -> None:
        self.path = path
        self._descriptor: int | None = None

    def __enter__(self) -> CapabilityLease:
        try:
            self._descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError as error:
            raise RuntimeError("duplicate capability writer") from error
        self.heartbeat("CLAIMED")
        return self

    def heartbeat(self, pair_id: str) -> None:
        if self._descriptor is None:
            raise RuntimeError("capability lease is inactive")
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "pair_id": pair_id,
                "utc": datetime.now(timezone.utc).isoformat(),
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


def expected_pair_identities(seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": f"{controller}:{seed}",
            "controller_id": controller,
            "numeric_seed": int(seed),
        }
        for seed in seeds
        for controller in CONTROLLER_ORDER
    ]


def initialize_pair_ledger(seeds: list[int]) -> dict[str, Any]:
    identities = expected_pair_identities(seeds)
    payload = {
        "schema_version": 1,
        "panel_id": "OPTIONS_CAPABILITY_V1",
        "state": "PREPARED_UNOPENED",
        "identities_sha256": canonical_sha256(identities),
        "expected_pair_count": 72,
        "attempt_id": None,
        "pairs": {
            row["pair_id"]: {**row, "state": "PENDING", "record_path": None}
            for row in identities
        },
    }
    if len(identities) != 72 or len(payload["pairs"]) != 72:
        raise RuntimeError("capability identity matrix is not exactly 4 x 18")
    if PAIR_LEDGER.exists():
        observed = json.loads(PAIR_LEDGER.read_text(encoding="utf-8"))
        if observed["identities_sha256"] != payload["identities_sha256"]:
            raise RuntimeError("existing capability pair identities changed")
        return observed
    atomic_json(PAIR_LEDGER, payload)
    return payload


class PairLedger:
    def __init__(self, path: Path = PAIR_LEDGER) -> None:
        self.path = path

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def open(self) -> dict[str, Any]:
        payload = self.read()
        if payload["state"] == "COMPLETE":
            raise PermissionError("OPTIONS_CAPABILITY_V1 is already complete")
        if payload["state"] != "PREPARED_UNOPENED":
            raise RuntimeError("an incomplete prior capability attempt requires audit")
        payload["state"] = "RUNNING"
        payload["attempt_id"] = str(uuid.uuid4())
        payload["opened_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, payload)
        return payload

    def start_pair(self, pair_id: str) -> dict[str, Any]:
        payload = self.read()
        pair = payload["pairs"][pair_id]
        if pair["state"] == "COMPLETE":
            raise PermissionError("completed pair must not be rerun")
        if pair["state"] != "PENDING":
            raise RuntimeError("uncertain STARTED pair requires exact verified resume")
        pair["state"] = "STARTED"
        pair["started_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, payload)
        return pair

    def complete_pair(
        self, pair_id: str, record_path: Path, record_sha256: str
    ) -> None:
        payload = self.read()
        pair = payload["pairs"][pair_id]
        if pair["state"] != "STARTED":
            raise RuntimeError("pair completion lacks STARTED state")
        pair.update(
            {
                "state": "COMPLETE",
                "record_path": str(record_path.relative_to(HERE)),
                "record_sha256": record_sha256,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_json(self.path, payload)

    def finish(self) -> dict[str, Any]:
        payload = self.read()
        if len(payload["pairs"]) != 72 or any(
            row["state"] != "COMPLETE" for row in payload["pairs"].values()
        ):
            raise RuntimeError("cannot finish capability with incomplete pairs")
        payload["state"] = "COMPLETE"
        payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path, payload)
        return payload
