"""Atomic, provenance-bound exact-resume checkpoints for GCSL attempts."""

from __future__ import annotations

import importlib.metadata
import os
import pickle
import platform
import random
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .replay import TrajectoryReplay

CHECKPOINT_SCHEMA_VERSION = 2
REQUIRED_RUNTIME_FIELDS = frozenset(
    {
        "training_seed",
        "attempt_id",
        "transition_count",
        "update_count",
        "episode_id",
        "episode_step",
        "pending_update",
        "pending_reset",
        "observation",
        "current_outcome",
        "policy_action_rng_state",
        "replay_rng_state",
        "task_stream_state",
        "current_task",
        "training_exposures",
        "diagnostics",
        "training_outcomes",
        "diagnostic_checkpoints_completed",
        "first_canonical_support_transition",
        "first_canonical_success_transition",
        "walltime_state",
    }
)


@dataclass(frozen=True)
class LoadedCheckpoint:
    attempt_id: str
    learner_state: dict[str, Any]
    replay: TrajectoryReplay
    environment: Any
    runtime: dict[str, Any]
    rng_states: dict[str, Any]
    configuration_sha256: str
    source_hashes: dict[str, str]


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("gymnasium", "numpy", "torch", "gym-longicontrol"):
        versions[name] = importlib.metadata.version(name)
    return versions


def capture_rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy_legacy"])
    torch.set_rng_state(states["torch_cpu"])


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    missing = REQUIRED_RUNTIME_FIELDS - runtime.keys()
    if missing:
        raise ValueError(f"checkpoint runtime fields missing: {sorted(missing)}")
    for field in ("transition_count", "update_count", "episode_id", "episode_step"):
        if not isinstance(runtime[field], int) or runtime[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if not runtime["attempt_id"]:
        raise ValueError("checkpoint attempt ID is required")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path: str | Path,
    *,
    learner_state: Mapping[str, Any],
    replay: TrajectoryReplay,
    environment: Any,
    runtime: Mapping[str, Any],
    configuration_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    _validate_runtime(runtime)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "attempt_id": runtime["attempt_id"],
        "configuration_sha256": configuration_sha256,
        "source_hashes": dict(source_hashes),
        "runtime_versions": runtime_versions(),
        "learner_state": dict(learner_state),
        "replay": replay.to_state(),
        "environment_pickle": pickle.dumps(environment, protocol=5),
        "runtime": dict(runtime),
        "rng_states": capture_rng_states(),
    }
    destination = Path(path)
    started = time.perf_counter()
    _atomic_bytes(destination, pickle.dumps(payload, protocol=5))
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "path": str(destination),
        "sha256": file_sha256(destination),
        "size_bytes": destination.stat().st_size,
        "write_s": time.perf_counter() - started,
        "transition_count": int(runtime["transition_count"]),
        "update_count": int(runtime["update_count"]),
        "attempt_id": runtime["attempt_id"],
    }
    metadata_path = destination.with_name(f"{destination.name}.metadata.json")
    _atomic_bytes(
        metadata_path,
        (json_dumps(metadata) + "\n").encode("utf-8"),
    )
    return metadata


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_configuration_sha256: str,
    expected_source_hashes: Mapping[str, str],
    expected_attempt_id: str | None = None,
) -> LoadedCheckpoint:
    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256:
        raise RuntimeError("checkpoint SHA-256 mismatch")
    payload = pickle.loads(source.read_bytes())
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema changed")
    comparisons = {
        "configuration_sha256": expected_configuration_sha256,
        "source_hashes": dict(expected_source_hashes),
        "runtime_versions": runtime_versions(),
    }
    for field, expected in comparisons.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"checkpoint provenance changed: {field}")
    if expected_attempt_id is not None and payload["attempt_id"] != expected_attempt_id:
        raise RuntimeError("checkpoint Attempt ID changed")
    runtime = dict(payload["runtime"])
    _validate_runtime(runtime)
    if runtime["attempt_id"] != payload["attempt_id"]:
        raise RuntimeError("checkpoint Attempt ID is internally inconsistent")
    return LoadedCheckpoint(
        attempt_id=payload["attempt_id"],
        learner_state=payload["learner_state"],
        replay=TrajectoryReplay.from_state(payload["replay"]),
        environment=pickle.loads(payload["environment_pickle"]),
        runtime=runtime,
        rng_states=payload["rng_states"],
        configuration_sha256=payload["configuration_sha256"],
        source_hashes=dict(payload["source_hashes"]),
    )
