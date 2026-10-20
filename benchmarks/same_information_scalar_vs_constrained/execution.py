"""Hardened TRANSITIONAL execution for the future Scalar arm."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_rl.adapter import alpha_value, logger_snapshot
from benchmarks.constrained_v2_collector_replication.checkpointing import file_sha256
from benchmarks.constrained_v2_repro_audit.harness import _module_sha256

from .agent import build_scalar_agent
from .checkpointing import (
    SCHEMA_VERSION,
    capture_rng_states,
    collector_state,
    load_verified,
    policy_optimizer_state,
    restore_collector,
    restore_policy,
    restore_rng_states,
    write_verified,
)
from .config import CHECKPOINTS, TRAINING_SEEDS, load_configuration
from .evaluation import evaluate_diagnostic
from .initialization import verify_initialization
from .streams import build_training_environment

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("runs/same-information-scalar-vs-constrained")
HEARTBEAT_INTERVAL = 250
SCIENTIFIC_FILES = (
    "canonical.json",
    "config.py",
    "scalarization.py",
    "signals.py",
    "agent.py",
    "split.py",
    "streams.py",
    "evaluation.py",
    "checkpointing.py",
    "execution.py",
    "runner.py",
    "reference.py",
    "initialization.py",
    "interpretation.py",
    "finalization.py",
    "resource_preflight.py",
    "preflight.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def runtime_versions() -> dict[str, Any]:
    import torch

    def version(name: str):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": version("gymnasium"),
        "tianshou": version("tianshou"),
        "fsrl": version("fsrl"),
        "fsrl_revision": "e056fc9498d5d037869533da7cf976acf462f918",
    }


def source_hashes() -> dict[str, str]:
    return {name: file_sha256(HERE / name) for name in SCIENTIFIC_FILES}


def assert_source_ancestor() -> None:
    expected = load_configuration().raw["lineage"]["preparation_base_commit"]
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected, "HEAD"],
        cwd=HERE,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Required preparation base is not an ancestor of HEAD")


def condition_name(seed: int) -> str:
    if seed not in TRAINING_SEEDS:
        raise ValueError("Unknown Scalar seed")
    return f"S{seed}_sameinfo_scalar_seed{seed}"


def initial_manifest() -> dict[str, Any]:
    configuration = load_configuration()
    return {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "configuration_sha256": configuration.configuration_sha256,
        "created_at": utc_now(),
        "condition_order": [condition_name(seed) for seed in TRAINING_SEEDS],
        "runs": {
            condition_name(seed): {
                "state": "NOT_STARTED",
                "training_seed": seed,
                "collector": "TRANSITIONAL",
                "attempts": [],
                "last_verified_checkpoint": None,
                "diagnostic_checkpoints_completed": [],
            }
            for seed in TRAINING_SEEDS
        },
        "reference_diagnostic": {"state": "NOT_EVALUATED", "episode_count": 0},
        "validation": {"state": "FROZEN_UNOPENED", "episode_count": 0},
        "paper": {"state": "SEALED_UNINSPECTED", "episode_count": 0},
    }


def load_or_create_manifest(output_root: Path) -> tuple[Path, dict[str, Any]]:
    path = output_root / "manifest.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = initial_manifest()
        atomic_json(path, payload)
    if payload["configuration_sha256"] != load_configuration().configuration_sha256:
        raise RuntimeError("Execution manifest configuration changed")
    return path, payload


class RunLease:
    def __init__(self, output_root: Path, run_id: str, *, resume: bool):
        self.output_root = output_root
        self.run_id = run_id
        self.resume = resume
        self.manifest_path: Path | None = None
        self.manifest: dict[str, Any] | None = None
        self.attempt: dict[str, Any] | None = None
        self.attempt_id = ""
        self.lock = output_root / run_id / "RUNNING.lock"

    def __enter__(self):
        self.manifest_path, self.manifest = load_or_create_manifest(self.output_root)
        run = self.manifest["runs"][self.run_id]
        if self.lock.exists():
            raise RuntimeError(f"Run lock exists: {self.lock}")
        if self.resume:
            if run["state"] != "INTERRUPTED" or not run["attempts"]:
                raise RuntimeError("Only an interrupted same Attempt may resume")
            self.attempt = run["attempts"][-1]
            self.attempt_id = self.attempt["attempt_id"]
            self.attempt["state"] = "RUNNING"
            self.attempt.setdefault("resume_events", []).append({"at": utc_now()})
        else:
            if run["state"] != "NOT_STARTED":
                raise RuntimeError("Fresh restart is unauthorized")
            self.attempt_id = str(uuid.uuid4())
            self.attempt = {
                "attempt_id": self.attempt_id,
                "attempt_number": 1,
                "state": "RUNNING",
                "started_at": utc_now(),
                "resume_events": [],
            }
            run["attempts"].append(self.attempt)
        run["state"] = "RUNNING"
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        self.lock.write_text(self.attempt_id + "\n", encoding="utf-8")
        atomic_json(self.manifest_path, self.manifest)
        return self

    def record(self, metadata: dict[str, Any], diagnostics: set[int]) -> None:
        assert self.manifest is not None and self.manifest_path is not None
        run = self.manifest["runs"][self.run_id]
        run["last_verified_checkpoint"] = metadata
        run["diagnostic_checkpoints_completed"] = sorted(diagnostics)
        self.attempt["last_verified_checkpoint"] = metadata
        atomic_json(self.manifest_path, self.manifest)

    def complete(
        self, checkpoint: Path, counters: dict[str, int], diagnostics: set[int]
    ):
        assert self.manifest is not None and self.manifest_path is not None
        if counters != {"real_transitions": 300_000, "rl_gradient_updates": 30_000}:
            raise RuntimeError("Final budgets are not exact")
        if diagnostics != set(CHECKPOINTS):
            raise RuntimeError("Scientific checkpoint schedule is incomplete")
        run = self.manifest["runs"][self.run_id]
        run.update(counters)
        run["final_checkpoint_sha256"] = file_sha256(checkpoint)
        run["state"] = "COMPLETED"
        self.attempt.update({"state": "COMPLETED", "ended_at": utc_now()})
        atomic_json(self.manifest_path, self.manifest)

    def __exit__(self, exc_type, exc, _traceback):
        if self.manifest is not None and self.manifest_path is not None:
            run = self.manifest["runs"][self.run_id]
            if exc_type is not None and run["state"] == "RUNNING":
                run["state"] = "INTERRUPTED"
                self.attempt.update(
                    {"state": "INTERRUPTED", "ended_at": utc_now(), "error": str(exc)}
                )
                atomic_json(self.manifest_path, self.manifest)
        if self.lock.exists():
            self.lock.unlink()
        return False


def _losses(logger: Any) -> dict[str, float]:
    if not tuple(logger.logger_keys):
        return {}
    return logger_snapshot(logger)


def _checkpoint_payload(
    *,
    run_id: str,
    attempt_id: str,
    policy: Any,
    collector: Any,
    recorder: Any,
    counters: dict[str, int],
    diagnostics: set[int],
    walltime_state: dict[str, float],
) -> dict[str, Any]:
    configuration = load_configuration()
    return {
        "checkpoint_schema_version": SCHEMA_VERSION,
        "condition": run_id,
        "attempt_id": attempt_id,
        "configuration_sha256": configuration.configuration_sha256,
        "source_hashes": source_hashes(),
        "runtime_versions": runtime_versions(),
        "policy": policy.state_dict(),
        "policy_optimizers": policy_optimizer_state(policy),
        "replay_buffer": collector.buffer,
        "environment_state": recorder,
        "collector_state": collector_state(collector),
        "counters": dict(counters),
        "diagnostic_checkpoints_completed": sorted(diagnostics),
        "rng_states": capture_rng_states(),
        "walltime_state": walltime_state,
    }


def _heartbeat(
    run_id: str, seed: int, attempt: str, counters: dict[str, int], status: str
):
    return {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "condition": run_id,
        "training_seed": seed,
        "collector": "TRANSITIONAL",
        "attempt_id": attempt,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "status": status,
        **counters,
    }


def run_policy(
    *,
    training_seed: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cpu",
    threads: int = 4,
    resume: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Path:
    configuration = load_configuration()
    from .config import require_main_training_authorized

    require_main_training_authorized(configuration)
    assert_source_ancestor()
    if resume != (checkpoint_path is not None):
        raise ValueError("Exact resume requires one explicit checkpoint")
    initialization = verify_initialization(
        configuration, device=device, threads=threads
    )
    if not initialization["passed"]:
        raise RuntimeError("Paired initialization gate failed")
    from tianshou.data import Collector, ReplayBuffer

    seed = int(training_seed)
    run_id = condition_name(seed)
    root = Path(output_root)
    with RunLease(root, run_id, resume=resume) as lease:
        recorder = build_training_environment(
            configuration,
            training_seed=seed,
            max_simulator_steps=configuration.raw["real_transition_budget"],
        )
        agent, logger = build_scalar_agent(
            recorder, training_seed=seed, device=device, threads=threads
        )
        policy = agent.policy
        policy.train()
        collector = Collector(
            policy,
            recorder,
            ReplayBuffer(configuration.raw["algorithm"]["replay_capacity"]),
            exploration_noise=True,
        )
        counters = {"real_transitions": 0, "rl_gradient_updates": 0}
        diagnostics: set[int] = set()
        walltime_state = {"completed_segments_s": 0.0}
        if resume:
            metadata_path = Path(checkpoint_path).with_name(
                "checkpoint.pt.metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload = load_verified(
                checkpoint_path,
                expected_sha256=metadata["sha256"],
                expected_configuration_sha256=configuration.configuration_sha256,
            )
            if (
                payload["attempt_id"] != lease.attempt_id
                or payload["condition"] != run_id
            ):
                raise RuntimeError("Checkpoint belongs to another run/Attempt")
            if payload["source_hashes"] != source_hashes():
                raise RuntimeError("Scientific source hashes changed")
            restore_policy(policy, payload)
            restored = payload["environment_state"]
            recorder.close()
            restore_collector(
                collector,
                state=payload["collector_state"],
                recorder=restored,
                replay=payload["replay_buffer"],
            )
            recorder = restored
            counters = dict(payload["counters"])
            diagnostics = set(payload["diagnostic_checkpoints_completed"])
            walltime_state = dict(payload["walltime_state"])
            restore_rng_states(payload["rng_states"])
        run_directory = root / run_id
        atomic_json(
            run_directory / "initialization.json",
            {"training_seed": seed, **initialization["by_seed"][str(seed)]},
        )
        segment_started = time.perf_counter()
        final_checkpoint: Path | None = None
        atomic_json(
            run_directory / "heartbeat.json",
            _heartbeat(run_id, seed, lease.attempt_id, counters, "RUNNING"),
        )
        while (
            counters["real_transitions"] < configuration.raw["real_transition_budget"]
        ):
            recorder.set_actor_version(counters["rl_gradient_updates"])
            stats = collector.collect(n_step=1)
            if int(stats["n/st"]) != 1:
                raise RuntimeError("Transition accounting changed")
            counters["real_transitions"] += 1
            target = math.floor(
                counters["real_transitions"]
                * configuration.raw["update_per_transition"]
                + 1e-12
            )
            while (
                counters["rl_gradient_updates"] < target
                and len(collector.buffer)
                >= configuration.raw["algorithm"]["batch_size"]
            ):
                policy.update(
                    configuration.raw["algorithm"]["batch_size"], collector.buffer
                )
                counters["rl_gradient_updates"] += 1
            real_count = counters["real_transitions"]
            scientific = real_count in CHECKPOINTS
            recovery = (
                real_count % configuration.raw["recovery_checkpoint_interval"] == 0
            )
            if scientific:
                result = evaluate_diagnostic(
                    policy, configuration, real_transitions=real_count
                )
                result["training_state"] = {
                    **counters,
                    "alpha": alpha_value(policy),
                    "actor_sha256": _module_sha256(policy.actor),
                    "learner_losses": _losses(logger),
                }
                directory = run_directory / f"step-{real_count:06d}"
                atomic_json(directory / "diagnostic.json", result)
                diagnostics.add(real_count)
            elif recovery:
                directory = run_directory / "recovery"
            if scientific or recovery:
                walltime = {
                    "completed_segments_s": walltime_state["completed_segments_s"]
                    + time.perf_counter()
                    - segment_started
                }
                checkpoint = directory / "checkpoint.pt"
                metadata = write_verified(
                    checkpoint,
                    _checkpoint_payload(
                        run_id=run_id,
                        attempt_id=lease.attempt_id,
                        policy=policy,
                        collector=collector,
                        recorder=recorder,
                        counters=counters,
                        diagnostics=diagnostics,
                        walltime_state=walltime,
                    ),
                )
                atomic_json(run_directory / "recovery" / "latest.json", metadata)
                lease.record(metadata, diagnostics)
                if scientific:
                    final_checkpoint = checkpoint
            if real_count % HEARTBEAT_INTERVAL == 0 or scientific or recovery:
                atomic_json(
                    run_directory / "heartbeat.json",
                    _heartbeat(run_id, seed, lease.attempt_id, counters, "RUNNING"),
                )
        if counters["rl_gradient_updates"] != 30_000 or final_checkpoint is None:
            raise RuntimeError("Scalar run did not reach exact final budgets")
        lease.complete(final_checkpoint, counters, diagnostics)
        atomic_json(
            run_directory / "heartbeat.json",
            _heartbeat(run_id, seed, lease.attempt_id, counters, "COMPLETED"),
        )
        return final_checkpoint


def run_matrix(**kwargs) -> list[Path]:
    configuration = load_configuration()
    from .config import require_main_training_authorized

    require_main_training_authorized(configuration)
    output_root = Path(kwargs.pop("output_root", DEFAULT_OUTPUT_ROOT))
    results = []
    for seed in TRAINING_SEEDS:
        _path, manifest = load_or_create_manifest(output_root)
        row = manifest["runs"][condition_name(seed)]
        if row["state"] == "COMPLETED":
            continue
        if row["state"] == "NOT_STARTED":
            results.append(
                run_policy(training_seed=seed, output_root=output_root, **kwargs)
            )
        elif row["state"] == "INTERRUPTED" and row["last_verified_checkpoint"]:
            results.append(
                run_policy(
                    training_seed=seed,
                    output_root=output_root,
                    resume=True,
                    checkpoint_path=row["last_verified_checkpoint"]["path"],
                    **kwargs,
                )
            )
        else:
            raise RuntimeError("FRESH_RESTART_REQUIRED")
    return results
