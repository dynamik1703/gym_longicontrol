"""Atomic, exact-continuation checkpoints for the projected-goal CRL runner."""

from __future__ import annotations

import importlib.metadata
import os
import pickle
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .replay import TransitionReplay

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
        "actor_rng_key",
        "update_rng_key",
        "future_rng_state",
        "track_rng_state",
        "track_seeds_used",
        "diagnostics",
        "training_outcomes",
        "development_completed",
        "first_canonical_positive_available_transition",
    }
)


@dataclass(frozen=True)
class LoadedCheckpoint:
    learner_state: Any
    replay: TransitionReplay
    environment: Any
    runtime: dict[str, Any]
    configuration_sha256: str
    scientific_source_sha256: dict[str, str]
    execution_source_sha256: dict[str, str]
    runtime_versions: dict[str, str]


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    packages = ("gymnasium", "numpy", "jax", "jaxlib", "flax", "optax")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    missing = REQUIRED_RUNTIME_FIELDS - runtime.keys()
    if missing:
        raise ValueError(f"checkpoint runtime fields are missing: {sorted(missing)}")
    transitions = runtime["transition_count"]
    updates = runtime["update_cycle_count"]
    if not isinstance(transitions, int) or transitions < 0:
        raise ValueError("transition_count must be a nonnegative integer")
    if not isinstance(updates, int) or updates < 0:
        raise ValueError("update_cycle_count must be a nonnegative integer")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
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


def save_checkpoint(
    path: str | Path,
    *,
    learner_state,
    replay: TransitionReplay,
    environment,
    runtime: Mapping[str, Any],
    configuration_sha256: str,
    scientific_source_sha256: Mapping[str, str],
    execution_source_sha256: Mapping[str, str],
) -> str:
    """Atomically persist every state needed for the next step and update."""

    from flax import serialization

    _validate_runtime(runtime)
    if len(configuration_sha256) != 64:
        raise ValueError("configuration_sha256 must be a SHA-256 hex digest")
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration_sha256": configuration_sha256,
        "scientific_source_sha256": dict(scientific_source_sha256),
        "execution_source_sha256": dict(execution_source_sha256),
        "runtime_versions": runtime_versions(),
        "learner_state": serialization.to_bytes(learner_state),
        "replay": replay.to_state(),
        "environment_pickle": pickle.dumps(environment, protocol=5),
        "runtime": dict(runtime),
    }
    destination = Path(path)
    _atomic_write(destination, pickle.dumps(payload, protocol=5))
    return file_sha256(destination)


def load_checkpoint(
    path: str | Path,
    *,
    learner_state_template,
    expected_configuration_sha256: str,
    expected_scientific_source_sha256: Mapping[str, str],
    expected_execution_source_sha256: Mapping[str, str],
) -> LoadedCheckpoint:
    """Load only a checkpoint matching the frozen science and local runtime."""

    from flax import serialization

    payload = pickle.loads(Path(path).read_bytes())
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("checkpoint schema version changed")
    if payload.get("configuration_sha256") != expected_configuration_sha256:
        raise RuntimeError("checkpoint scientific configuration hash changed")
    if payload.get("scientific_source_sha256") != dict(
        expected_scientific_source_sha256
    ):
        raise RuntimeError("checkpoint scientific source hashes changed")
    if payload.get("execution_source_sha256") != dict(expected_execution_source_sha256):
        raise RuntimeError("checkpoint execution source hashes changed")
    if payload.get("runtime_versions") != runtime_versions():
        raise RuntimeError("checkpoint dependency/runtime versions changed")
    runtime = dict(payload["runtime"])
    _validate_runtime(runtime)
    return LoadedCheckpoint(
        learner_state=serialization.from_bytes(
            learner_state_template, payload["learner_state"]
        ),
        replay=TransitionReplay.from_state(payload["replay"]),
        environment=pickle.loads(payload["environment_pickle"]),
        runtime=runtime,
        configuration_sha256=payload["configuration_sha256"],
        scientific_source_sha256=dict(payload["scientific_source_sha256"]),
        execution_source_sha256=dict(payload["execution_source_sha256"]),
        runtime_versions=dict(payload["runtime_versions"]),
    )
