"""Atomic provenance-bound checkpoints for exact mid-block recovery."""

from __future__ import annotations

import importlib.metadata
import json
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

from .replay import MacroReplay

CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_RUNTIME_FIELDS = frozenset(
    {
        "training_seed",
        "attempt_id",
        "native_transition_count",
        "macro_transition_count",
        "gradient_update_count",
        "episode_id",
        "episode_native_step",
        "pending_reset",
        "observation",
        "held_strategy_action",
        "remaining_native_steps_in_block",
        "macro_accumulator",
        "task_stream",
        "policy_generator_state",
        "update_generator_state",
        "warmup_rng_state",
        "replay_rng_state",
    }
)


@dataclass(frozen=True)
class LoadedCheckpoint:
    attempt_id: str
    agent_state: dict[str, Any]
    replay: MacroReplay
    environment: Any
    controller: Any
    runtime: dict[str, Any]
    global_rng_states: dict[str, Any]
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
    for distribution in ("gymnasium", "numpy", "torch", "gym-longicontrol"):
        versions[distribution] = importlib.metadata.version(distribution)
    return versions


def capture_global_rng_states() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }


def restore_global_rng_states(states: Mapping[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy_legacy"])
    torch.set_rng_state(states["torch_cpu"])


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    missing = REQUIRED_RUNTIME_FIELDS - runtime.keys()
    if missing:
        raise ValueError(f"checkpoint runtime fields missing: {sorted(missing)}")
    if not runtime["attempt_id"]:
        raise ValueError("checkpoint Attempt ID is required")
    for field in (
        "native_transition_count",
        "macro_transition_count",
        "gradient_update_count",
        "episode_id",
        "episode_native_step",
        "remaining_native_steps_in_block",
    ):
        if not isinstance(runtime[field], int) or runtime[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")


def _atomic_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_checkpoint(
    path: str | Path,
    *,
    agent_state: Mapping[str, Any],
    replay: MacroReplay,
    environment: Any,
    controller: Any,
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
        "agent_state": dict(agent_state),
        "replay": replay.state_dict(),
        "environment_pickle": pickle.dumps(environment, protocol=5),
        "controller_pickle": pickle.dumps(controller, protocol=5),
        "runtime": dict(runtime),
        "global_rng_states": capture_global_rng_states(),
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
        "native_transition_count": int(runtime["native_transition_count"]),
        "macro_transition_count": int(runtime["macro_transition_count"]),
        "gradient_update_count": int(runtime["gradient_update_count"]),
        "attempt_id": runtime["attempt_id"],
    }
    _atomic_bytes(
        destination.with_name(f"{destination.name}.metadata.json"),
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return metadata


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None,
    expected_configuration_sha256: str,
    expected_source_hashes: Mapping[str, str],
    expected_attempt_id: str | None,
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
        agent_state=payload["agent_state"],
        replay=MacroReplay.from_state_dict(payload["replay"]),
        environment=pickle.loads(payload["environment_pickle"]),
        controller=pickle.loads(payload["controller_pickle"]),
        runtime=runtime,
        global_rng_states=payload["global_rng_states"],
        configuration_sha256=payload["configuration_sha256"],
        source_hashes=dict(payload["source_hashes"]),
    )
