"""Persistent JSON-lines worker for one native checkpoint runtime."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import traceback
from typing import Any

from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed

from .checkpoint_audit import (
    load_c_agents,
    load_h_agents,
    native_runtime_manifest,
    production_identity,
)
from .constants import (
    C_CONTROLLERS,
    EXECUTION_MANIFEST,
    H_CONTROLLERS,
    PAIR_LEDGER,
    RUNTIME_MANIFESTS,
    SPLIT_LEDGER,
)
from .evaluation import evaluate_controller
from .fixtures import run_fixture_suite
from .integrity import verify_source_bundle
from .ledger import validate_pair_ledger
from .split import verify_task_row
from .utils import canonical_sha256, load_json


class NativeWorker:
    def __init__(self, runtime_id: str) -> None:
        if runtime_id not in ("C", "H"):
            raise ValueError("worker runtime must be C or H")
        self.runtime_id = runtime_id
        expected = load_json(RUNTIME_MANIFESTS)[runtime_id]
        observed = native_runtime_manifest()
        if observed != expected:
            raise RuntimeError(f"native runtime {runtime_id} manifest changed")
        if runtime_id == "C":
            self.checkpoint_metadata, self.c_agents = load_c_agents()
            self.h_agents: dict[str, Any] = {}
            self.allowed = C_CONTROLLERS
        else:
            self.checkpoint_metadata, self.h_agents = load_h_agents()
            self.c_agents = {}
            self.allowed = H_CONTROLLERS
        self.prepared: dict[str, dict[str, Any]] = {}
        self.executed: set[str] = set()

    def _frozen_pair(self, pair_id: str) -> dict[str, Any]:
        ledger = load_json(PAIR_LEDGER)
        validate_pair_ledger(ledger)
        rows = [row for row in ledger["pairs"] if row["pair_id"] == pair_id]
        if len(rows) != 1:
            raise PermissionError("pair is absent from the frozen schedule")
        return rows[0]

    def _verify_manifest(self, manifest_sha256: str) -> dict[str, Any]:
        manifest = load_json(EXECUTION_MANIFEST)
        if canonical_sha256(manifest) != manifest_sha256:
            raise RuntimeError("execution manifest identity changed")
        verify_source_bundle(manifest["scientific_source_bundle"])
        if (
            manifest["runtime_manifests"][self.runtime_id]["manifest_sha256"]
            != (native_runtime_manifest()["manifest_sha256"])
        ):
            raise RuntimeError("worker runtime is not the frozen runtime")
        return manifest

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        pair = request["pair"]
        pair_id = pair["pair_id"]
        if pair_id in self.executed or pair_id in self.prepared:
            raise RuntimeError("worker rejects an already prepared or executed pair")
        controller = pair["identity"]["controller_id"]
        if controller not in self.allowed:
            raise PermissionError("controller is incompatible with worker runtime")
        self._verify_manifest(request["execution_manifest_sha256"])
        frozen = self._frozen_pair(pair_id)
        if frozen != pair or frozen["state"] != "NOT_STARTED":
            raise RuntimeError("worker accepts only a fresh frozen pair")
        if frozen["identity"]["worker_runtime"] != self.runtime_id:
            raise PermissionError("pair runtime assignment changed")
        task = next(
            row
            for row in load_json(SPLIT_LEDGER)["panel"]["accepted_tasks"]
            if int(row["numeric_seed"]) == int(frozen["identity"]["numeric_seed"])
        )
        if (
            task["route_sha256"] != frozen["identity"]["route_sha256"]
            or task["full_task_sha256"] != frozen["identity"]["full_task_sha256"]
        ):
            raise RuntimeError("pair task fingerprint changed")
        verify_task_row(task)
        if controller in self.checkpoint_metadata:
            metadata = self.checkpoint_metadata[controller]
            if (
                metadata["checkpoint_sha256"] != frozen["identity"]["checkpoint_sha256"]
                or metadata["attempt_id"] != frozen["identity"]["attempt_id"]
            ):
                raise RuntimeError("loaded checkpoint provenance differs from pair")
        token = secrets.token_hex(32)
        self.prepared[pair_id] = {
            "token": token,
            "pair": pair,
            "execution_manifest_sha256": request["execution_manifest_sha256"],
        }
        return {
            "command": "PREPARED",
            "pair_id": pair_id,
            "ownership_token": token,
            "checkpoint_loaded_and_verified": (
                controller in self.checkpoint_metadata
                or controller in C_CONTROLLERS - set(self.checkpoint_metadata)
            ),
            "runtime_id": self.runtime_id,
        }

    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        pair_id = request["pair_id"]
        prepared = self.prepared.get(pair_id)
        if prepared is None or prepared["token"] != request["ownership_token"]:
            raise PermissionError("worker does not own this prepared pair")
        self._verify_manifest(prepared["execution_manifest_sha256"])
        frozen = self._frozen_pair(pair_id)
        if frozen["state"] != "STARTED":
            raise RuntimeError("master did not persist STARTED before reset")
        if frozen["ownership_token_sha256"] != canonical_sha256(
            request["ownership_token"]
        ):
            raise PermissionError("master STARTED token differs from worker token")
        identity = frozen["identity"]
        result = evaluate_controller(
            numeric_seed=int(identity["numeric_seed"]),
            controller_id=identity["controller_id"],
            c_agents=self.c_agents,
            h_agents=self.h_agents,
        )
        self.executed.add(pair_id)
        del self.prepared[pair_id]
        return {"command": "RESULT", "pair_id": pair_id, **result}

    def fingerprint(self, request: dict[str, Any]) -> dict[str, Any]:
        seed = int(request["numeric_seed"])
        if seed in PAPER_SEEDS:
            raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
        fingerprint = fingerprint_track(
            track_from_seed(seed), v2_semantics(), numeric_seed=seed
        )
        return {
            "command": "FINGERPRINT",
            "runtime_id": self.runtime_id,
            "numeric_seed": seed,
            "route_sha256": fingerprint.route_sha256,
            "full_task_sha256": fingerprint.full_task_sha256,
        }

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "READY":
            return {
                "command": "READY",
                "runtime_id": self.runtime_id,
                "runtime_manifest_sha256": native_runtime_manifest()["manifest_sha256"],
                "loaded_checkpoints": self.checkpoint_metadata,
            }
        if command == "PREPARE":
            return self.prepare(request)
        if command == "EXECUTE":
            return self.execute(request)
        if command == "FINGERPRINT":
            return self.fingerprint(request)
        if command == "SYNTHETIC_FIXTURES":
            return {
                "command": command,
                "runtime_id": self.runtime_id,
                **run_fixture_suite(),
            }
        if command == "PRODUCTION_IDENTITY":
            return {
                "command": command,
                "runtime_id": self.runtime_id,
                "identity": production_identity(),
            }
        if command == "SHUTDOWN":
            return {"command": "SHUTDOWN", "runtime_id": self.runtime_id}
        raise PermissionError(f"unknown worker command: {command}")


def _emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, choices=("C", "H"))
    arguments = parser.parse_args(argv)
    try:
        worker = NativeWorker(arguments.runtime)
    except BaseException as error:
        _emit(
            {
                "command": "FATAL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise SystemExit(2) from error
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = worker.dispatch(request)
            _emit(response)
            if request.get("command") == "SHUTDOWN":
                break
        except BaseException as error:
            _emit(
                {
                    "command": "ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )


if __name__ == "__main__":
    main()
