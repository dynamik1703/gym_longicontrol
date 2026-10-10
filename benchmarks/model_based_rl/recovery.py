"""Execution-only heartbeat, verified checkpoint, and exact-restore support."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

from .checkpointing import (
    atomic_torch_save_with_metrics,
    file_sha256,
    load_checkpoint,
    restore_rng_states,
)
from .execution import atomic_json, utc_now

CHECKPOINT_SCHEMA_VERSION = 2
RECOVERY_INTERVAL = 10_000
HEARTBEAT_INTERVAL = 250
SCIENCE_FREEZE_PATH = Path(__file__).with_name("scientific_freeze.json")
EXECUTION_FILES = (
    "benchmarks/model_based_rl/checkpointing.py",
    "benchmarks/model_based_rl/execution.py",
    "benchmarks/model_based_rl/training.py",
    "benchmarks/model_based_rl/runner.py",
    "benchmarks/model_based_rl/recovery.py",
    "benchmarks/model_based_rl/recovery_admin.py",
    "benchmarks/model_based_rl/detached_launcher.py",
    "benchmarks/model_based_rl/matrix_worker.py",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, Any]:
    import numpy as np

    try:
        import torch
    except ImportError:  # pragma: no cover - required only for main execution
        torch = None
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": None if torch is None else torch.__version__,
        "gymnasium": _package_version("gymnasium"),
        "tianshou": _package_version("tianshou"),
        "fsrl_revision": "e056fc9498d5d037869533da7cf976acf462f918",
    }


def load_scientific_freeze() -> dict[str, Any]:
    return json.loads(SCIENCE_FREEZE_PATH.read_text(encoding="utf-8"))


def verify_scientific_freeze(repository: str | Path = ".") -> dict[str, str]:
    root = Path(repository)
    expected = load_scientific_freeze()["files"]
    actual = {name: file_sha256(root / name) for name in expected}
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"Frozen MBRL science changed: {mismatches}")
    return actual


def execution_hashes(repository: str | Path = ".") -> dict[str, str]:
    root = Path(repository)
    return {
        name: file_sha256(root / name)
        for name in EXECUTION_FILES
        if (root / name).is_file()
    }


def collector_state(collector: Any) -> dict[str, Any]:
    """Return only pickle-safe state; DummyVectorEnv contains local lambdas."""

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
        raise RuntimeError("Exact restore expects one synchronous environment worker")
    collector.env.workers[0].env = recorder
    collector.buffer = replay_buffer
    collector.data = state["data"]
    collector.collect_step = int(state["collect_step"])
    collector.collect_episode = int(state["collect_episode"])
    collector.collect_time = float(state["collect_time"])


def restore_policy_state(policy: Any, payload: dict[str, Any]) -> None:
    policy.load_state_dict(payload["policy"])
    optimizers = payload["policy_optimizers"]
    policy.actor_optim.load_state_dict(optimizers["actor"])
    policy.critics_optim.load_state_dict(optimizers["critics"])
    for optimizer, state in zip(policy.lag_optims, payload["lagrangian_states"]):
        optimizer.load_state_dict(state)
    if getattr(policy, "_is_auto_alpha", False):
        policy._alpha_optim.load_state_dict(optimizers["alpha"])  # noqa: SLF001
        policy._log_alpha.data.copy_(optimizers["log_alpha"])  # noqa: SLF001
        policy._alpha = policy._log_alpha.detach().exp()  # noqa: SLF001


def restore_runtime_rng(
    payload: dict[str, Any], *, model_rng: Any, synthetic_rng: Any
) -> None:
    restore_rng_states(
        payload["rng_states"], model_rng=model_rng, synthetic_rng=synthetic_rng
    )


def heartbeat_payload(
    *,
    condition: str,
    seed: int,
    attempt_id: str,
    counters: dict[str, Any],
    recorder: Any,
    synthetic: Any,
    status: str,
    last_recovery_checkpoint: str | None,
    last_scientific_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "condition": condition,
        "seed": seed,
        "attempt_id": attempt_id,
        "pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "session_id": os.getsid(0),
        "host": socket.gethostname(),
        "real_transition_count": int(counters["real_transitions"]),
        "rl_update_count": int(counters["rl_gradient_updates"]),
        "model_refresh_count": int(counters["model_refreshes"]),
        "model_update_count": int(counters["model_gradient_updates"]),
        "synthetic_generated_count": int(counters["synthetic_transitions"]),
        "synthetic_sampled_count": int(synthetic.sampled_total),
        "current_episode_id": int(recorder.completed_episodes + 1),
        "current_episode_step": int(recorder.episode_step),
        "last_completed_episode": int(recorder.completed_episodes),
        "last_recovery_checkpoint": last_recovery_checkpoint,
        "last_scientific_checkpoint": last_scientific_checkpoint,
        "status": status,
    }


def write_heartbeat(run_directory: str | Path, payload: dict[str, Any]) -> Path:
    return atomic_json(Path(run_directory) / "heartbeat.json", payload)


def recovery_checkpoint_due(real_transitions: int) -> bool:
    return real_transitions > 0 and real_transitions % RECOVERY_INTERVAL == 0


def _metadata_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.name}.metadata.json")


def write_verified_checkpoint(
    checkpoint: str | Path,
    payload: dict[str, Any],
    *,
    kind: str,
) -> dict[str, Any]:
    destination = Path(checkpoint)
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Recovery checkpoint schema version is missing or invalid")
    write_metrics = atomic_torch_save_with_metrics(destination, payload)
    restore_started = time.perf_counter()
    restored = load_checkpoint(destination, expected_sha256=write_metrics["sha256"])
    restore_s = time.perf_counter() - restore_started
    if restored["counters"] != payload["counters"]:
        raise RuntimeError("Verified checkpoint counters changed during round trip")
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "path": str(destination),
        "sha256": write_metrics["sha256"],
        "size_bytes": write_metrics["size_bytes"],
        "transition_count": int(payload["counters"]["real_transitions"]),
        "attempt_id": payload["attempt_id"],
        "timestamp_utc": utc_now(),
        "serialization_s": write_metrics["serialization_s"],
        "fsync_s": write_metrics["fsync_s"],
        "atomic_commit_s": write_metrics["atomic_commit_s"],
        "write_total_s": write_metrics["write_total_s"],
        "verified_restore_s": restore_s,
    }
    atomic_json(_metadata_path(destination), metadata)
    return metadata


def verify_recovery_checkpoint(
    checkpoint: str | Path,
    *,
    configuration_sha256: str,
    repository: str | Path = ".",
    expected_attempt_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(checkpoint)
    metadata = json.loads(_metadata_path(source).read_text(encoding="utf-8"))
    payload = load_checkpoint(source, expected_sha256=metadata["sha256"])
    if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("Checkpoint schema does not support exact resume")
    if payload["hashes"]["configuration"] != configuration_sha256:
        raise RuntimeError("Checkpoint scientific configuration differs")
    if payload["hashes"]["scientific_sources"] != verify_scientific_freeze(repository):
        raise RuntimeError("Checkpoint scientific-source provenance differs")
    if payload["hashes"]["execution_sources"] != execution_hashes(repository):
        raise RuntimeError("Checkpoint execution-source provenance differs")
    if payload["runtime_versions"] != runtime_versions():
        raise RuntimeError("Checkpoint runtime/dependency provenance differs")
    if expected_attempt_id is not None and payload["attempt_id"] != expected_attempt_id:
        raise RuntimeError("Checkpoint belongs to a different attempt")
    if int(payload["counters"]["real_transitions"]) != int(
        metadata["transition_count"]
    ):
        raise RuntimeError("Checkpoint metadata transition count differs")
    return payload, metadata


def publish_latest_recovery(
    run_directory: str | Path, metadata: dict[str, Any]
) -> Path:
    return atomic_json(Path(run_directory) / "recovery" / "latest.json", metadata)


def apply_retention(run_directory: str | Path, *, keep: int = 2) -> None:
    recovery_root = Path(run_directory) / "recovery"
    checkpoints = sorted(
        recovery_root.glob("step-*/checkpoint.pt"),
        key=lambda path: int(path.parent.name.split("-")[-1]),
    )
    for checkpoint in checkpoints[:-keep]:
        metadata = _metadata_path(checkpoint)
        if metadata.exists():
            metadata.unlink()
        checkpoint.unlink()
        try:
            checkpoint.parent.rmdir()
        except OSError:
            pass
