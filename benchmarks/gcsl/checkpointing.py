"""Atomic exact-continuation checkpoints for GCSL policy attempts."""

from __future__ import annotations

import importlib.metadata
import os
import pickle
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .replay import TrajectoryReplay

CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_RUNTIME_FIELDS = frozenset(
    {
        "transition_count",
        "update_cycle_count",
        "episode_id",
        "episode_step",
        "current_episode_ended",
        "observation",
        "current_outcome",
        "policy_action_rng_state",
        "replay_rng_state",
        "track_rng_state",
        "track_seeds_used",
        "diagnostics",
        "training_outcomes",
        "development_completed",
        "first_canonical_target_available_transition",
        "first_physical_canonical_success_transition",
    }
)


@dataclass(frozen=True)
class LoadedCheckpoint:
    learner_state: dict[str, Any]
    replay: TrajectoryReplay
    environment: Any
    runtime: dict[str, Any]
    configuration_sha256: str
    scientific_source_sha256: dict[str, str]
    execution_source_sha256: dict[str, str]


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    packages = ("gymnasium", "numpy", "torch")
    versions = {"python": platform.python_version()}
    for name in packages:
        versions[name] = importlib.metadata.version(name)
    return versions


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    missing = REQUIRED_RUNTIME_FIELDS - runtime.keys()
    if missing:
        raise ValueError(f"checkpoint runtime fields missing: {sorted(missing)}")
    for field in ("transition_count", "update_cycle_count"):
        if not isinstance(runtime[field], int) or runtime[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")


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
    environment,
    runtime: Mapping[str, Any],
    configuration_sha256: str,
    scientific_source_sha256: Mapping[str, str],
    execution_source_sha256: Mapping[str, str],
) -> str:
    _validate_runtime(runtime)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "configuration_sha256": configuration_sha256,
        "scientific_source_sha256": dict(scientific_source_sha256),
        "execution_source_sha256": dict(execution_source_sha256),
        "runtime_versions": runtime_versions(),
        "learner_state": dict(learner_state),
        "replay": replay.to_state(),
        "environment_pickle": pickle.dumps(environment, protocol=5),
        "runtime": dict(runtime),
    }
    destination = Path(path)
    _atomic_bytes(destination, pickle.dumps(payload, protocol=5))
    return file_sha256(destination)


def load_checkpoint(
    path: str | Path,
    *,
    expected_configuration_sha256: str,
    expected_scientific_source_sha256: Mapping[str, str],
    expected_execution_source_sha256: Mapping[str, str],
) -> LoadedCheckpoint:
    payload = pickle.loads(Path(path).read_bytes())
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema changed")
    comparisons = {
        "configuration_sha256": expected_configuration_sha256,
        "scientific_source_sha256": dict(expected_scientific_source_sha256),
        "execution_source_sha256": dict(expected_execution_source_sha256),
        "runtime_versions": runtime_versions(),
    }
    for field, expected in comparisons.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"checkpoint provenance changed: {field}")
    runtime = dict(payload["runtime"])
    _validate_runtime(runtime)
    return LoadedCheckpoint(
        learner_state=payload["learner_state"],
        replay=TrajectoryReplay.from_state(payload["replay"]),
        environment=pickle.loads(payload["environment_pickle"]),
        runtime=runtime,
        configuration_sha256=payload["configuration_sha256"],
        scientific_source_sha256=dict(payload["scientific_source_sha256"]),
        execution_source_sha256=dict(payload["execution_source_sha256"]),
    )


def checkpoint_size_bytes(path: str | Path) -> int:
    return Path(path).stat().st_size
