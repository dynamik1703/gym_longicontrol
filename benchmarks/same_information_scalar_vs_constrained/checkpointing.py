"""Atomic exact-resume checkpoints for scalar SAC (intentionally no PID state)."""

from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_v2_collector_replication.checkpointing import file_sha256

SCHEMA_VERSION = 1
REQUIRED = frozenset(
    {
        "checkpoint_schema_version",
        "condition",
        "attempt_id",
        "configuration_sha256",
        "source_hashes",
        "runtime_versions",
        "policy",
        "policy_optimizers",
        "replay_buffer",
        "environment_state",
        "collector_state",
        "counters",
        "diagnostic_checkpoints_completed",
        "rng_states",
        "walltime_state",
    }
)


def capture_rng_states() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy_legacy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    import torch

    random.setstate(states["python"])
    np.random.set_state(states["numpy_legacy"])
    torch.set_rng_state(states["torch_cpu"])


def policy_optimizer_state(policy: Any) -> dict[str, Any]:
    if policy.lag_optims:
        raise RuntimeError("Scalar checkpoint cannot contain PID state")
    return {
        "actor": policy.actor_optim.state_dict(),
        "critics": policy.critics_optim.state_dict(),
        "alpha": policy._alpha_optim.state_dict(),  # noqa: SLF001
        "log_alpha": policy._log_alpha.detach().cpu(),  # noqa: SLF001
    }


def restore_policy(policy: Any, payload: dict[str, Any]) -> None:
    if policy.lag_optims:
        raise RuntimeError("Scalar policy unexpectedly contains PID state")
    policy.load_state_dict(payload["policy"])
    state = payload["policy_optimizers"]
    policy.actor_optim.load_state_dict(state["actor"])
    policy.critics_optim.load_state_dict(state["critics"])
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
    collector: Any, *, state: dict[str, Any], recorder: Any, replay: Any
):
    if len(collector.env.workers) != 1:
        raise RuntimeError("Exact resume requires one synchronous environment")
    collector.env.workers[0].env = recorder
    collector.buffer = replay
    collector.data = state["data"]
    collector.collect_step = state["collect_step"]
    collector.collect_episode = state["collect_episode"]
    collector.collect_time = state["collect_time"]


def validate_payload(payload: dict[str, Any]) -> None:
    missing = REQUIRED - payload.keys()
    if missing:
        raise ValueError(f"Incomplete scalar checkpoint: {sorted(missing)}")
    if payload["checkpoint_schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported scalar checkpoint schema")
    if "lagrangian_states" in payload or "pid_state" in payload:
        raise ValueError("Scalar checkpoint must not contain PID/Lagrange state")


def write_verified(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
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
    finally:
        if temporary.exists():
            temporary.unlink()
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "path": str(destination),
        "sha256": file_sha256(destination),
        "size_bytes": destination.stat().st_size,
        "write_s": time.perf_counter() - started,
        "transition_count": int(payload["counters"]["real_transitions"]),
        "attempt_id": payload["attempt_id"],
    }
    restored = load_verified(
        destination,
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256=payload["configuration_sha256"],
    )
    if restored["counters"] != payload["counters"]:
        raise RuntimeError("Checkpoint round trip changed counters")
    meta_path = destination.with_name(f"{destination.name}.metadata.json")
    meta_temporary = meta_path.with_name(f".{meta_path.name}.tmp")
    try:
        with meta_temporary.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(meta_temporary, meta_path)
    finally:
        if meta_temporary.exists():
            meta_temporary.unlink()
    return metadata


def load_verified(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_configuration_sha256: str | None = None,
) -> dict[str, Any]:
    import torch

    source = Path(path)
    if expected_sha256 is not None and file_sha256(source) != expected_sha256:
        raise RuntimeError("Scalar checkpoint SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    validate_payload(payload)
    if (
        expected_configuration_sha256 is not None
        and payload["configuration_sha256"] != expected_configuration_sha256
    ):
        raise RuntimeError("Scalar checkpoint configuration changed")
    return payload
