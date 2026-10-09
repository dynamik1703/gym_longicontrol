"""Frozen scheduling, authorization, provenance, and restart protection."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .checkpointing import file_sha256
from .config import (
    MODEL_CONDITIONS,
    RESERVED_TRACKS,
    load_configuration,
    training_track_seed,
)

FREEZE_PARENT = "cae33ff317757e24182955569b698ed29448683c"
DEFAULT_OUTPUT_ROOT = Path("runs/model-based-rl-v1")
RUN_STATES = frozenset({"NOT_STARTED", "RUNNING", "INTERRUPTED", "COMPLETED"})
SCIENTIFIC_FILES = (
    "benchmarks/model_based_rl/canonical.json",
    "benchmarks/model_based_rl/adapter.py",
    "benchmarks/model_based_rl/model_state.py",
    "benchmarks/model_based_rl/physics_model.py",
    "benchmarks/model_based_rl/learned_model.py",
    "benchmarks/model_based_rl/model_training.py",
    "benchmarks/model_based_rl/imagination.py",
    "benchmarks/model_based_rl/fsrl_adapter.py",
    "benchmarks/model_based_rl/model_disabled_parity.py",
    "benchmarks/model_based_rl/diagnostics.py",
    "benchmarks/model_based_rl/checkpointing.py",
    "benchmarks/model_based_rl/evaluation.py",
    "benchmarks/model_based_rl/execution.py",
    "benchmarks/model_based_rl/training.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def assert_freeze_ancestor(repository: str | Path = ".") -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE_PARENT, "HEAD"],
        cwd=repository,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("MBRL branch does not descend from the frozen CRL commit")


def scientific_hashes(repository: str | Path = ".") -> dict[str, str]:
    root = Path(repository)
    return {name: file_sha256(root / name) for name in SCIENTIFIC_FILES}


class TrainingTrackStream:
    """Dedicated, checkpointable training RNG excluding every research split."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4D42524C]))
        self.draw_count = 0

    def next_seed(self) -> int:
        value = training_track_seed(self.rng)
        self.draw_count += 1
        if value in RESERVED_TRACKS:
            raise AssertionError("Reserved track escaped exclusion")
        return value

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state, "draw_count": self.draw_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.draw_count = int(state["draw_count"])


def initial_manifest() -> dict[str, Any]:
    configuration = load_configuration()
    runs = {
        f"{condition}-seed-{seed}": {
            "condition": condition,
            "training_seed": seed,
            "state": "NOT_STARTED",
            "attempts": [],
            "real_transitions": 0,
            "synthetic_transitions": 0,
            "rl_gradient_updates": 0,
            "model_gradient_updates": 0,
            "final_checkpoint_sha256": None,
            "development_complete": False,
        }
        for condition in MODEL_CONDITIONS
        for seed in configuration.raw["training_seeds"]
    }
    return {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "configuration_sha256": configuration.configuration_sha256,
        "created_at": utc_now(),
        "runs": runs,
        "validation": {"state": "SEALED", "opened_at": None},
        "paper_test": {"state": "SEALED", "opened_at": None},
    }


def load_or_create_manifest(output_root: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(output_root) / "manifest.json"
    configuration = load_configuration()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["configuration_sha256"] != configuration.configuration_sha256:
            raise RuntimeError("Existing study manifest uses a different configuration")
        return path, payload
    payload = initial_manifest()
    atomic_json(path, payload)
    return path, payload


def validation_ready(manifest: dict[str, Any]) -> bool:
    runs = manifest["runs"].values()
    return all(
        run["state"] == "COMPLETED"
        and run["real_transitions"] == 300_000
        and run["rl_gradient_updates"] == 30_000
        and run["final_checkpoint_sha256"]
        and run["development_complete"]
        for run in runs
    ) and len(manifest["runs"]) == 6


def assert_validation_authorized(manifest: dict[str, Any]) -> None:
    configuration = load_configuration()
    if not configuration.raw["authorization"]["validation_authorized"]:
        raise PermissionError("Validation is not authorized by the frozen protocol")
    if not validation_ready(manifest):
        raise RuntimeError("All six final policies must be frozen before Validation")
    if manifest["validation"]["state"] != "SEALED":
        raise RuntimeError("Validation may be opened exactly once")


def assert_paper_test_blocked() -> None:
    raise PermissionError("Paper-test tracks 4000-4017 remain sealed")


class RunLease(AbstractContextManager):
    """Exclusive attempt record; crashes remain visible as INTERRUPTED attempts."""

    def __init__(
        self,
        output_root: str | Path,
        run_id: str,
        *,
        resume: bool,
        restart: bool = False,
    ):
        if resume and restart:
            raise ValueError("An attempt cannot be both resumed and freshly restarted")
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.resume = resume
        self.restart = restart
        self.attempt_id = str(uuid.uuid4())
        self.path: Path | None = None
        self.manifest_path: Path | None = None
        self.manifest: dict[str, Any] | None = None

    def __enter__(self):
        self.manifest_path, self.manifest = load_or_create_manifest(self.output_root)
        run = self.manifest["runs"][self.run_id]
        allowed = {"INTERRUPTED"} if self.resume or self.restart else {"NOT_STARTED"}
        if run["state"] not in allowed:
            if self.resume:
                operation = "resume"
            elif self.restart:
                operation = "restart"
            else:
                operation = "start"
            raise RuntimeError(
                f"Cannot {operation} {self.run_id} from {run['state']}"
            )
        if self.restart:
            authorization = run.get("restart_authorization")
            if not authorization or authorization.get("consumed_at") is not None:
                raise PermissionError(
                    "Fresh restart lacks unused explicit authorization"
                )
            counter_names = (
                "real_transitions",
                "synthetic_transitions",
                "rl_gradient_updates",
                "model_gradient_updates",
            )
            if any(int(run[name]) != 0 for name in counter_names):
                raise RuntimeError("Authorized fresh restart is limited to zero work")
            authorization["consumed_at"] = utc_now()
        run_directory = self.output_root / self.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        self.path = run_directory / ".run.lock"
        descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"host": socket.gethostname(), "pid": os.getpid()}, stream)
        run["state"] = "RUNNING"
        run["attempts"].append(
            {
                "attempt_id": self.attempt_id,
                "started_at": utc_now(),
                "resume": self.resume,
                "restart": self.restart,
                "state": "RUNNING",
            }
        )
        atomic_json(self.manifest_path, self.manifest)
        return self

    def complete(self, counters: dict[str, Any], checkpoint: str | Path) -> None:
        assert self.manifest is not None and self.manifest_path is not None
        run = self.manifest["runs"][self.run_id]
        if counters["real_transitions"] != 300_000:
            raise RuntimeError("A completed run must consume exactly 300k real steps")
        run.update(counters)
        run["final_checkpoint_sha256"] = file_sha256(checkpoint)
        run["state"] = "COMPLETED"
        run["attempts"][-1].update({"state": "COMPLETED", "ended_at": utc_now()})
        atomic_json(self.manifest_path, self.manifest)

    def __exit__(self, exc_type, exc, traceback):
        assert self.manifest is not None and self.manifest_path is not None
        run = self.manifest["runs"][self.run_id]
        if exc_type is not None and run["state"] == "RUNNING":
            run["state"] = "INTERRUPTED"
            run["attempts"][-1].update(
                {"state": "INTERRUPTED", "ended_at": utc_now(), "error": str(exc)}
            )
            atomic_json(self.manifest_path, self.manifest)
        if self.path is not None and self.path.exists():
            self.path.unlink()
        return False
