"""Atomic, verified, exact-resume checkpoints for the V3 factorial."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import CONDITION_NAMES

CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_KEYS = frozenset(
    {
        "checkpoint_schema_version",
        "condition",
        "attempt_id",
        "configuration_sha256",
        "source_hashes",
        "runtime_versions",
        "policy",
        "policy_optimizers",
        "lagrangian_states",
        "replay_buffer",
        "environment_state",
        "collector_state",
        "counters",
        "episode_diagnostics",
        "diagnostic_checkpoints_completed",
        "rng_states",
        "previous_completed_episodes",
        "pid_update_count",
        "current_episode_start_actor_sha256",
        "current_episode_start_transition",
        "walltime_state",
    }
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_states() -> dict[str, Any]:
    import torch

    state = {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(states: dict[str, Any]) -> None:
    import torch

    random.setstate(states["python"])
    np.random.set_state(states["numpy_legacy"])
    torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def policy_optimizer_state(policy: Any) -> dict[str, Any]:
    state = {
        "actor": policy.actor_optim.state_dict(),
        "critics": policy.critics_optim.state_dict(),
    }
    if getattr(policy, "_is_auto_alpha", False):
        state["alpha"] = policy._alpha_optim.state_dict()  # noqa: SLF001
        state["log_alpha"] = policy._log_alpha.detach().cpu()  # noqa: SLF001
    return state


def restore_policy(policy: Any, payload: dict[str, Any]) -> None:
    policy.load_state_dict(payload["policy"])
    optimizers = payload["policy_optimizers"]
    policy.actor_optim.load_state_dict(optimizers["actor"])
    policy.critics_optim.load_state_dict(optimizers["critics"])
    for optimizer, state in zip(
        policy.lag_optims, payload["lagrangian_states"], strict=True
    ):
        optimizer.load_state_dict(state)
    if getattr(policy, "_is_auto_alpha", False):
        policy._alpha_optim.load_state_dict(optimizers["alpha"])  # noqa: SLF001
        policy._log_alpha.data.copy_(optimizers["log_alpha"])  # noqa: SLF001
        policy._alpha = policy._log_alpha.detach().exp()  # noqa: SLF001


def collector_state(collector: Any) -> dict[str, Any]:
    return {
        "data": copy.deepcopy(collector.data),
        "collect_step": int(collector.collect_step),
        "collect_episode": int(collector.collect_episode),
        "collect_time": float(collector.collect_time),
    }


def restore_collector(
    collector: Any,
    *,
    state: dict[str, Any],
    recorder: Any,
    replay_buffer: Any,
) -> None:
    if len(collector.env.workers) != 1:
        raise RuntimeError("Exact resume requires one synchronous environment")
    collector.env.workers[0].env = recorder
    collector.buffer = replay_buffer
    collector.data = state["data"]
    collector.collect_step = int(state["collect_step"])
    collector.collect_episode = int(state["collect_episode"])
    collector.collect_time = float(state["collect_time"])


def validate_payload(payload: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Checkpoint is incomplete: {sorted(missing)}")
    if payload["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported V3 checkpoint schema")
    if payload["condition"] not in CONDITION_NAMES:
        raise ValueError("Checkpoint condition is outside the frozen matrix")
    counters = payload["counters"]
    if counters["real_transitions"] < 0 or counters["rl_gradient_updates"] < 0:
        raise ValueError("Checkpoint counters cannot be negative")


def atomic_save(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    validate_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    started = time.perf_counter()
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "path": str(destination),
        "sha256": file_sha256(destination),
        "size_bytes": destination.stat().st_size,
        "write_total_s": time.perf_counter() - started,
        "transition_count": int(payload["counters"]["real_transitions"]),
        "attempt_id": payload["attempt_id"],
    }


def load_verified(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_configuration_sha256: str | None = None,
) -> dict[str, Any]:
    import torch

    source = Path(path)
    actual = file_sha256(source)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError("Checkpoint SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    validate_payload(payload)
    if (
        expected_configuration_sha256 is not None
        and payload["configuration_sha256"] != expected_configuration_sha256
    ):
        raise RuntimeError("Checkpoint configuration changed")
    return payload


def write_verified_checkpoint(
    path: str | Path, payload: dict[str, Any]
) -> dict[str, Any]:
    metadata = atomic_save(path, payload)
    restored = load_verified(
        path,
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256=payload["configuration_sha256"],
    )
    if restored["counters"] != payload["counters"]:
        raise RuntimeError("Checkpoint counters changed during restore verification")
    metadata_path = Path(path).with_name(f"{Path(path).name}.metadata.json")
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, metadata_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return metadata
