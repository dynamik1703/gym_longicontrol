"""Atomic schema-versioned checkpoints for exact MVE resume."""

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

CHECKPOINT_SCHEMA_VERSION = 1
REQUIRED_KEYS = frozenset(
    {
        "checkpoint_schema_version",
        "condition",
        "training_seed",
        "attempt_id",
        "configuration_sha256",
        "source_hashes",
        "runtime_versions",
        "policy",
        "policy_optimizers",
        "lagrangian_states",
        "real_replay",
        "physics_sidecar",
        "environment_state",
        "collector_state",
        "track_stream_state",
        "counters",
        "diagnostics",
        "target_accumulator",
        "anticipation_probes",
        "rng_states",
        "pid_real_episode_updates",
        "previous_completed_episodes",
        "development_checkpoints_completed",
    }
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_states(rollout_rng: np.random.Generator) -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "model_rollout": copy.deepcopy(rollout_rng.bit_generator.state),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(
    states: dict[str, Any], rollout_rng: np.random.Generator
) -> None:
    import torch

    random.setstate(states["python"])
    np.random.set_state(states["numpy_legacy"])
    rollout_rng.bit_generator.state = states["model_rollout"]
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
    state = payload["policy_optimizers"]
    policy.actor_optim.load_state_dict(state["actor"])
    policy.critics_optim.load_state_dict(state["critics"])
    for optimizer, optimizer_state in zip(
        policy.lag_optims, payload["lagrangian_states"]
    ):
        optimizer.load_state_dict(optimizer_state)
    if getattr(policy, "_is_auto_alpha", False):
        policy._alpha_optim.load_state_dict(state["alpha"])  # noqa: SLF001
        policy._log_alpha.data.copy_(state["log_alpha"])  # noqa: SLF001
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
        raise ValueError("Unsupported checkpoint schema")
    if payload["condition"] not in {"matched_real_only", "physics_mve"}:
        raise ValueError("Unknown study condition")
    counters = payload["counters"]
    if counters.get("synthetic_transitions", 0) != 0:
        raise ValueError("Primary MVE checkpoints may contain no synthetic replay")


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
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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
        raise RuntimeError("Checkpoint configuration differs")
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
        raise RuntimeError("Checkpoint counters changed during verified restore")
    metadata_path = Path(path).with_name(f"{Path(path).name}.metadata.json")
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, metadata_path)
    return metadata
