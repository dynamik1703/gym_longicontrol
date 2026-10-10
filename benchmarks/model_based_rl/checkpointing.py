"""Atomic, integrity-checked full-state checkpoints for exact resume."""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_COMMON_KEYS = frozenset(
    {
        "policy",
        "policy_optimizers",
        "lagrangian_states",
        "real_rl_replay",
        "real_model_replay",
        "synthetic_replay",
        "environment_state",
        "track_stream_state",
        "counters",
        "diagnostics",
        "hashes",
        "rng_states",
        "model_condition",
    }
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_states(
    *, model_rng: np.random.Generator, synthetic_rng: np.random.Generator
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "model": model_rng.bit_generator.state,
        "synthetic": synthetic_rng.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_states(
    states: dict[str, Any],
    *,
    model_rng: np.random.Generator,
    synthetic_rng: np.random.Generator,
) -> None:
    import torch

    random.setstate(states["python"])
    np.random.set_state(states["numpy_legacy"])
    model_rng.bit_generator.state = states["model"]
    synthetic_rng.bit_generator.state = states["synthetic"]
    torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    missing = REQUIRED_COMMON_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing required state: {sorted(missing)}")
    condition = payload["model_condition"]
    if condition == "learned":
        required = {"learned_model", "model_optimizer_states", "normalization"}
        missing_learned = required - payload.keys()
        if missing_learned:
            raise ValueError(
                f"Learned checkpoint is incomplete: {sorted(missing_learned)}"
            )
    elif condition != "physics":
        raise ValueError("Unknown checkpoint model condition")


def atomic_torch_save_with_metrics(
    path: str | Path, payload: dict[str, Any]
) -> dict[str, Any]:
    import torch

    validate_checkpoint_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    started = time.perf_counter()
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            serialized_at = time.perf_counter()
            os.fsync(stream.fileno())
            fsynced_at = time.perf_counter()
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    committed_at = time.perf_counter()
    return {
        "sha256": file_sha256(destination),
        "size_bytes": destination.stat().st_size,
        "serialization_s": serialized_at - started,
        "fsync_s": fsynced_at - serialized_at,
        "atomic_commit_s": committed_at - fsynced_at,
        "write_total_s": committed_at - started,
    }


def atomic_torch_save(path: str | Path, payload: dict[str, Any]) -> str:
    return str(atomic_torch_save_with_metrics(path, payload)["sha256"])


def load_checkpoint(path: str | Path, *, expected_sha256: str | None = None):
    import torch

    source = Path(path)
    actual = file_sha256(source)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError("Checkpoint SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    validate_checkpoint_payload(payload)
    return payload
