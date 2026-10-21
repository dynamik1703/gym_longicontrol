"""Verified atomic full-state checkpoints for same-attempt recovery."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .controller import TemporalOptionController

CHECKPOINT_SCHEMA_VERSION = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def capture_checkpoint(
    *,
    environment: Any,
    controller: TemporalOptionController,
    observation: np.ndarray,
    held_strategy_action: float | None,
    decision_clock: int,
    partial_transition: dict[str, Any],
    counters: dict[str, int],
) -> dict[str, Any]:
    """Capture environment, causal/option memory, clocks, accumulators and RNGs."""

    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "environment_pickle": pickle.dumps(environment, protocol=5),
        "controller": controller.snapshot(),
        "observation": np.asarray(observation, dtype=np.float64),
        "held_strategy_action": held_strategy_action,
        "decision_clock": int(decision_clock),
        "partial_transition": dict(partial_transition),
        "counters": {key: int(value) for key, value in counters.items()},
        "numpy_legacy_rng_state": np.random.get_state(),
    }


def restore_checkpoint(
    payload: dict[str, Any], controller: TemporalOptionController
) -> tuple[Any, np.ndarray, float | None, int, dict[str, Any], dict[str, int]]:
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported temporal-option checkpoint schema")
    environment = pickle.loads(payload["environment_pickle"])
    controller.restore(payload["controller"])
    np.random.set_state(payload["numpy_legacy_rng_state"])
    return (
        environment,
        np.asarray(payload["observation"], dtype=np.float64),
        payload["held_strategy_action"],
        int(payload["decision_clock"]),
        dict(payload["partial_transition"]),
        {key: int(value) for key, value in payload["counters"].items()},
    )


def write_verified_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = pickle.dumps(payload, protocol=5)
    digest = sha256_bytes(encoded)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    metadata = {
        "schema_version": 1,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sha256": digest,
        "size_bytes": len(encoded),
    }
    sidecar = destination.with_name(f"{destination.name}.metadata.json")
    sidecar_temp = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
    sidecar_temp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(sidecar_temp, sidecar)
    return metadata


def load_verified_checkpoint(path: Path) -> dict[str, Any]:
    source = Path(path)
    metadata = json.loads(
        source.with_name(f"{source.name}.metadata.json").read_text(encoding="utf-8")
    )
    encoded = source.read_bytes()
    if sha256_bytes(encoded) != metadata["sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    payload = pickle.loads(encoded)
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported temporal-option checkpoint schema")
    return payload
