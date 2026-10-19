"""Hardened serial execution of the paired six-policy collector study."""

from __future__ import annotations

import copy
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
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_rl.adapter import (
    alpha_value,
    logger_snapshot,
    multiplier_values,
)
from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_v2_repro_audit.harness import _module_sha256

from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_states,
    collector_state,
    file_sha256,
    load_verified,
    policy_optimizer_state,
    restore_collector,
    restore_policy,
    restore_rng_states,
    write_verified_checkpoint,
)
from .config import CONDITION_NAMES, Condition, load_configuration
from .evaluation import evaluate_diagnostic, evaluate_validation
from .initialization import policy_initial_hashes
from .streams import (
    TrainingRecorder,
    actor_stationarity,
    build_training_environment,
    exposure_summary,
)

DEFAULT_OUTPUT_ROOT = Path("runs/constrained-v2-collector-replication")
HEARTBEAT_INTERVAL = 250
SCIENTIFIC_FILES = (
    "canonical.json",
    "config.py",
    "split.py",
    "streams.py",
    "initialization.py",
    "evaluation.py",
    "checkpointing.py",
    "execution.py",
    "runner.py",
    "resource_preflight.py",
    "preflight.py",
    "finalize.py",
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

    def version(name: str) -> str | None:
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


def source_hashes(repository: str | Path = ".") -> dict[str, str]:
    root = Path(repository) / "benchmarks/constrained_v2_collector_replication"
    return {name: file_sha256(root / name) for name in SCIENTIFIC_FILES}


def assert_source_ancestor(repository: str | Path = ".") -> None:
    source = load_configuration().raw["source_commit"]
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source, "HEAD"],
        cwd=repository,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "Study does not descend from the requested V3 terminal commit"
        )


def preparation_status(repository: str | Path = ".") -> dict[str, Any]:
    path = (
        Path(repository)
        / "benchmarks/constrained_v2_collector_replication/preparation_status.json"
    )
    if not path.exists():
        raise PermissionError(
            "Collector-replication preparation has not been materialized"
        )
    status = json.loads(path.read_text(encoding="utf-8"))
    configuration = load_configuration()
    if status.get("configuration_sha256") != configuration.configuration_sha256:
        raise RuntimeError("Preparation used a different frozen configuration")
    if status.get("scientific_source_hashes") != source_hashes(repository):
        raise RuntimeError("Scientific sources changed after preparation")
    if status.get("ready_for_main_training") is not True:
        raise PermissionError("Mandatory pre-training gates did not all pass")
    return status


def initial_manifest() -> dict[str, Any]:
    configuration = load_configuration()
    return {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "configuration_sha256": configuration.configuration_sha256,
        "source_commit": configuration.raw["source_commit"],
        "created_at": utc_now(),
        "condition_order": list(CONDITION_NAMES),
        "runs": {
            name: {
                "state": "NOT_STARTED",
                "attempts": [],
                "real_transitions": 0,
                "rl_gradient_updates": 0,
                "completed_training_episodes": 0,
                "pid_update_count": 0,
                "diagnostic_checkpoints_completed": [],
                "final_checkpoint_sha256": None,
            }
            for name in CONDITION_NAMES
        },
        "diagnostic": {"state": "FROZEN", "episode_count": 0},
        "validation": {
            "state": "FROZEN_UNOPENED",
            "episode_count": 0,
            "opened_at": None,
            "completed_conditions": [],
        },
        "legacy_development": {"state": "NOT_EVALUATED", "episode_count": 0},
        "legacy_validation": {"state": "NOT_EVALUATED", "episode_count": 0},
        "paper": {"state": "SEALED_UNINSPECTED", "episode_count": 0},
    }


def load_or_create_manifest(output_root: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(output_root) / "manifest.json"
    configuration = load_configuration()
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["configuration_sha256"] != configuration.configuration_sha256:
            raise RuntimeError("Existing manifest uses another protocol")
        return path, manifest
    manifest = initial_manifest()
    atomic_json(path, manifest)
    return path, manifest


class RunLease(AbstractContextManager):
    def __init__(self, output_root: str | Path, run_id: str, *, resume: bool):
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.resume = resume
        self.attempt_id = str(uuid.uuid4())
        self.manifest_path: Path | None = None
        self.manifest: dict[str, Any] | None = None
        self.lock_path: Path | None = None
        self.attempt_index = -1

    def __enter__(self):
        self.manifest_path, self.manifest = load_or_create_manifest(self.output_root)
        run = self.manifest["runs"][self.run_id]
        required = "INTERRUPTED" if self.resume else "NOT_STARTED"
        if run["state"] != required:
            raise RuntimeError(f"Cannot start {self.run_id} from {run['state']}")
        if self.resume:
            if not run.get("last_verified_checkpoint") or not run["attempts"]:
                raise PermissionError("Exact resume requires a verified checkpoint")
            attempt = run["attempts"][-1]
            self.attempt_id = attempt["attempt_id"]
            attempt.setdefault("resume_events", []).append(
                {"at": utc_now(), "pid": os.getpid()}
            )
            attempt["state"] = "RUNNING"
            self.attempt_index = len(run["attempts"]) - 1
        else:
            attempt = {
                "attempt_number": 1,
                "attempt_id": self.attempt_id,
                "state": "RUNNING",
                "started_at": utc_now(),
                "resume_events": [],
            }
            run["attempts"].append(attempt)
            self.attempt_index = len(run["attempts"]) - 1
        directory = self.output_root / self.run_id
        directory.mkdir(parents=True, exist_ok=True)
        self.lock_path = directory / ".run.lock"
        descriptor = os.open(
            self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
        )
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                {
                    "attempt_id": self.attempt_id,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "started_at": utc_now(),
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        run["state"] = "RUNNING"
        atomic_json(self.manifest_path, self.manifest)
        return self

    @property
    def attempt(self) -> dict[str, Any]:
        assert self.manifest is not None
        return self.manifest["runs"][self.run_id]["attempts"][self.attempt_index]

    def record_checkpoint(
        self,
        counters: dict[str, int],
        metadata: dict[str, Any],
        *,
        episode_count: int,
        pid_update_count: int,
        diagnostic_completed: set[int],
    ) -> None:
        assert self.manifest is not None and self.manifest_path is not None
        run = self.manifest["runs"][self.run_id]
        run.update({key: int(value) for key, value in counters.items()})
        run["completed_training_episodes"] = int(episode_count)
        run["pid_update_count"] = int(pid_update_count)
        run["diagnostic_checkpoints_completed"] = sorted(diagnostic_completed)
        run["last_verified_checkpoint"] = metadata
        self.attempt["last_verified_checkpoint"] = metadata
        atomic_json(self.manifest_path, self.manifest)

    def complete(
        self,
        counters: dict[str, int],
        final_checkpoint: Path,
        *,
        episode_count: int,
        pid_update_count: int,
        diagnostic_completed: set[int],
    ) -> None:
        assert self.manifest is not None and self.manifest_path is not None
        if counters != {"real_transitions": 300_000, "rl_gradient_updates": 30_000}:
            raise RuntimeError("A completed policy must have exact frozen budgets")
        expected_diagnostics = set(load_configuration().raw["diagnostic_checkpoints"])
        if diagnostic_completed != expected_diagnostics:
            raise RuntimeError("Diagnostic checkpoint schedule is incomplete")
        run = self.manifest["runs"][self.run_id]
        run.update({key: int(value) for key, value in counters.items()})
        run["completed_training_episodes"] = int(episode_count)
        run["pid_update_count"] = int(pid_update_count)
        run["diagnostic_checkpoints_completed"] = sorted(diagnostic_completed)
        run["final_checkpoint_sha256"] = file_sha256(final_checkpoint)
        run["state"] = "COMPLETED"
        self.attempt.update({"state": "COMPLETED", "ended_at": utc_now()})
        self.manifest["diagnostic"]["episode_count"] = sum(
            18 * len(row["diagnostic_checkpoints_completed"])
            for row in self.manifest["runs"].values()
        )
        if all(row["state"] == "COMPLETED" for row in self.manifest["runs"].values()):
            self.manifest["diagnostic"]["state"] = "COMPLETED"
        atomic_json(self.manifest_path, self.manifest)

    def __exit__(self, exc_type, exc, traceback):
        assert self.manifest is not None and self.manifest_path is not None
        run = self.manifest["runs"][self.run_id]
        if exc_type is not None and run["state"] == "RUNNING":
            run["state"] = "INTERRUPTED"
            self.attempt.update(
                {"state": "INTERRUPTED", "ended_at": utc_now(), "error": str(exc)}
            )
            atomic_json(self.manifest_path, self.manifest)
        if self.lock_path is not None and self.lock_path.exists():
            self.lock_path.unlink()
        return False


def _heartbeat(
    *,
    condition: Condition,
    attempt_id: str,
    counters: dict[str, int],
    recorder: TrainingRecorder,
    pid_update_count: int,
    status: str,
    last_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "condition": condition.name,
        "training_seed": condition.training_seed,
        "track_stream": f"CANONICAL_{condition.training_seed}",
        "collector": condition.collector,
        "attempt_id": attempt_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        **{key: int(value) for key, value in counters.items()},
        "completed_training_episodes": recorder.completed_episodes,
        "pid_update_count": pid_update_count,
        "current_episode_step": recorder.episode_step,
        "last_verified_checkpoint": last_checkpoint,
        "status": status,
    }


def _losses(logger: Any) -> dict[str, float]:
    if not tuple(logger.logger_keys):
        return {}
    return logger_snapshot(logger)


def apply_collector_schedule(
    *,
    collector_name: str,
    completed_costs: np.ndarray | None,
    policy: Any,
    replay_buffer: Any,
    counters: dict[str, int],
    target_updates: int,
    batch_size: int,
) -> dict[str, Any]:
    """Apply the frozen PID/update order after one inserted transition."""

    if collector_name not in {"EPISODIC", "TRANSITIONAL"}:
        raise ValueError("Unknown collector treatment")
    events = ["TRANSITION_INSERTED"]
    replay_size_before_updates = len(replay_buffer)
    update_count_before = counters["rl_gradient_updates"]
    completed = completed_costs is not None
    lambda_before = multiplier_values(policy) if completed else None
    updates_before_pid = counters["rl_gradient_updates"] if completed else None
    if completed:
        policy.pre_update_fn(stats_train={"cost": completed_costs})
        events.append("PID_UPDATED")
    lambda_after = multiplier_values(policy) if completed else None
    updates_after_pid = counters["rl_gradient_updates"] if completed else None
    permit_updates = collector_name == "TRANSITIONAL" or completed
    if permit_updates:
        while (
            counters["rl_gradient_updates"] < target_updates
            and len(replay_buffer) >= batch_size
        ):
            policy.update(batch_size, replay_buffer)
            counters["rl_gradient_updates"] += 1
            events.append("RL_UPDATE")
    if completed:
        policy.post_update_fn(stats_train={"cost": completed_costs})
        events.append("POST_UPDATE_HOOK")
    return {
        "events": events,
        "lambda_before": lambda_before,
        "lambda_after": lambda_after,
        "updates_before_pid": updates_before_pid,
        "updates_after_pid": updates_after_pid,
        "updates_after_episode": counters["rl_gradient_updates"],
        "updates_executed": counters["rl_gradient_updates"] - update_count_before,
        "replay_size_before_updates": replay_size_before_updates,
        "replay_size_after_updates": len(replay_buffer),
        "pid_updated": completed,
    }


def _episode_row(
    *,
    condition: Condition,
    recorder: TrainingRecorder,
    transition_count: int,
    completed: Any,
    exposure: dict[str, Any],
    lambda_before: tuple[float, ...],
    lambda_after: tuple[float, ...],
    updates_before_pid: int,
    updates_after_pid: int,
    updates_after_episode: int,
    alpha: float,
    actor_start: str,
    actor_end: str,
    actor_after_processing: str,
    losses: dict[str, float],
    episode_update_events: list[dict[str, int]],
    replay_size_at_terminal: int,
) -> dict[str, Any]:
    metrics = asdict(completed.metrics)
    return {
        "condition": condition.name,
        "nominal_seed": condition.training_seed,
        "track_stream": f"CANONICAL_{condition.training_seed}",
        "collector": condition.collector,
        "episode_index": recorder.completed_episodes - 1,
        "real_transition_count": transition_count,
        "track": exposure,
        "episode_duration_steps": completed.simulator_steps,
        "episode_duration_s": metrics["travel_time_s"],
        "completed": metrics["completed"],
        "budget_truncated": completed.budget_truncated,
        "speed_episodic_cost": completed.costs[0],
        "deadline_episodic_cost": completed.costs[1],
        "lambda_speed_before": lambda_before[0],
        "lambda_speed_after": lambda_after[0],
        "lambda_deadline_before": lambda_before[1],
        "lambda_deadline_after": lambda_after[1],
        "rl_updates_before_pid": updates_before_pid,
        "rl_updates_after_pid": updates_after_pid,
        "rl_updates_after_episode_processing": updates_after_episode,
        "alpha": alpha,
        "actor_sha256_at_episode_start": actor_start,
        "actor_sha256_at_episode_terminal_before_updates": actor_end,
        "actor_sha256_after_episode_processing": actor_after_processing,
        "actor_stationarity": actor_stationarity(
            recorder.last_completed_actor_versions
        ),
        "update_timing": {
            "events": episode_update_events,
            "update_events": len(episode_update_events),
            "updates_executed": sum(
                row["updates_executed"] for row in episode_update_events
            ),
            "replay_size_at_terminal": replay_size_at_terminal,
        },
        "metrics": metrics,
        "learner_losses": losses,
    }


def _checkpoint_payload(
    *,
    condition: Condition,
    attempt_id: str,
    policy: Any,
    collector: Any,
    recorder: TrainingRecorder,
    counters: dict[str, int],
    episode_diagnostics: list[dict[str, Any]],
    diagnostic_completed: set[int],
    previous_completed_episodes: int,
    pid_update_count: int,
    current_episode_start_actor_sha256: str,
    current_episode_start_transition: int,
    current_episode_update_events: list[dict[str, int]],
    walltime_state: dict[str, float],
) -> dict[str, Any]:
    configuration = load_configuration()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "condition": condition.name,
        "attempt_id": attempt_id,
        "configuration_sha256": configuration.configuration_sha256,
        "source_hashes": source_hashes(),
        "runtime_versions": runtime_versions(),
        "policy": policy.state_dict(),
        "policy_optimizers": policy_optimizer_state(policy),
        "lagrangian_states": [item.state_dict() for item in policy.lag_optims],
        "replay_buffer": collector.buffer,
        "environment_state": copy.deepcopy(recorder),
        "collector_state": collector_state(collector),
        "counters": dict(counters),
        "episode_diagnostics": episode_diagnostics,
        "diagnostic_checkpoints_completed": sorted(diagnostic_completed),
        "rng_states": capture_rng_states(),
        "previous_completed_episodes": previous_completed_episodes,
        "pid_update_count": pid_update_count,
        "current_episode_start_actor_sha256": (current_episode_start_actor_sha256),
        "current_episode_start_transition": current_episode_start_transition,
        "current_episode_update_events": current_episode_update_events,
        "walltime_state": walltime_state,
    }


def _condition_by_name(name: str) -> Condition:
    for condition in load_configuration().conditions:
        if condition.name == name:
            return condition
    raise ValueError(f"Unknown condition: {name}")


def run_policy(
    *,
    condition_name: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cpu",
    threads: int = 4,
    resume: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Path:
    preparation_status()
    configuration = load_configuration()
    condition = _condition_by_name(condition_name)
    if resume != (checkpoint_path is not None):
        raise ValueError("Exact resume requires one explicit checkpoint")

    import tianshou
    from tianshou.data import Collector, ReplayBuffer

    if tianshou.__version__ != configuration.v2.algorithm.tianshou_version:
        raise RuntimeError("Pinned Tianshou version changed")
    output_root = Path(output_root)
    with RunLease(output_root, condition.name, resume=resume) as lease:
        recorder = build_training_environment(
            configuration,
            training_seed=condition.training_seed,
            max_simulator_steps=configuration.raw["real_transition_budget"],
        )
        agent, logger = build_agent(
            configuration.v2,
            recorder,
            training_seed=condition.training_seed,
            device=device,
            threads=threads,
        )
        policy = agent.policy
        initialized = policy_initial_hashes(
            policy, replay_capacity=configuration.v2.algorithm.buffer_size
        )
        frozen_initialization = json.loads(
            (Path(__file__).resolve().parent / "initialization.json").read_text(
                encoding="utf-8"
            )
        )["by_condition"][condition.name]
        if initialized != frozen_initialization:
            raise RuntimeError(
                "Main-run learner/replay initialization differs from freeze"
            )
        policy.train()
        collector = Collector(
            policy,
            recorder,
            ReplayBuffer(configuration.v2.algorithm.buffer_size),
            exploration_noise=True,
        )
        counters = {"real_transitions": 0, "rl_gradient_updates": 0}
        episode_diagnostics: list[dict[str, Any]] = []
        diagnostic_completed: set[int] = set()
        previous_completed = 0
        pid_update_count = 0
        current_episode_start_actor = _module_sha256(policy.actor)
        current_episode_start_transition = 0
        current_episode_update_events: list[dict[str, int]] = []
        walltime_state = {"completed_segments_s": 0.0}
        last_checkpoint: str | None = None

        if resume:
            assert checkpoint_path is not None
            metadata_path = Path(checkpoint_path).with_name(
                f"{Path(checkpoint_path).name}.metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload = load_verified(
                checkpoint_path,
                expected_sha256=metadata["sha256"],
                expected_configuration_sha256=configuration.configuration_sha256,
            )
            if payload["attempt_id"] != lease.attempt_id:
                raise RuntimeError("Checkpoint belongs to another Attempt")
            if payload["condition"] != condition.name:
                raise RuntimeError("Checkpoint belongs to another condition")
            if payload["source_hashes"] != source_hashes():
                raise RuntimeError("Checkpoint scientific sources differ")
            if payload["runtime_versions"] != runtime_versions():
                raise RuntimeError("Checkpoint runtime provenance differs")
            restore_policy(policy, payload)
            restored_recorder = payload["environment_state"]
            recorder.close()
            restore_collector(
                collector,
                state=payload["collector_state"],
                recorder=restored_recorder,
                replay_buffer=payload["replay_buffer"],
            )
            recorder = restored_recorder
            counters = dict(payload["counters"])
            episode_diagnostics = payload["episode_diagnostics"]
            diagnostic_completed = set(payload["diagnostic_checkpoints_completed"])
            previous_completed = int(payload["previous_completed_episodes"])
            pid_update_count = int(payload["pid_update_count"])
            current_episode_start_actor = payload["current_episode_start_actor_sha256"]
            current_episode_start_transition = int(
                payload["current_episode_start_transition"]
            )
            current_episode_update_events = list(
                payload["current_episode_update_events"]
            )
            walltime_state = dict(payload["walltime_state"])
            restore_rng_states(payload["rng_states"])
            last_checkpoint = str(checkpoint_path)

        run_directory = output_root / condition.name
        atomic_json(run_directory / "initialization.json", initialized)
        segment_started = time.perf_counter()
        atomic_json(
            run_directory / "heartbeat.json",
            _heartbeat(
                condition=condition,
                attempt_id=lease.attempt_id,
                counters=counters,
                recorder=recorder,
                pid_update_count=pid_update_count,
                status="RUNNING",
                last_checkpoint=last_checkpoint,
            ),
        )
        checkpoints = set(configuration.raw["diagnostic_checkpoints"])
        final_checkpoint: Path | None = None

        while (
            counters["real_transitions"] < configuration.raw["real_transition_budget"]
        ):
            recorder.set_actor_version(counters["rl_gradient_updates"])
            stats = collector.collect(n_step=1)
            if int(stats["n/st"]) != 1:
                raise RuntimeError("Native transition accounting changed")
            counters["real_transitions"] += 1
            real_count = counters["real_transitions"]
            completed_now = recorder.completed_episodes != previous_completed
            completed = None
            exposure = None
            lambda_before: tuple[float, ...] | None = None
            lambda_after: tuple[float, ...] | None = None
            updates_before_pid: int | None = None
            updates_after_pid: int | None = None
            actor_terminal: str | None = None

            if completed_now:
                if recorder.completed_episodes != previous_completed + 1:
                    raise RuntimeError("Exactly one episode may finish per transition")
                completed = recorder.last_completed_episode
                exposure = recorder.last_completed_exposure
                if completed is None or exposure is None:
                    raise RuntimeError("Completed episode metadata is incomplete")
                actor_terminal = _module_sha256(policy.actor)
                costs = np.asarray(completed.costs, dtype=np.float64)

            target_updates = math.floor(
                real_count * configuration.raw["update_per_transition"] + 1e-12
            )
            schedule = apply_collector_schedule(
                collector_name=condition.collector,
                completed_costs=costs if completed_now else None,
                policy=policy,
                replay_buffer=collector.buffer,
                counters=counters,
                target_updates=target_updates,
                batch_size=configuration.v2.algorithm.batch_size,
            )
            if schedule["updates_executed"]:
                current_episode_update_events.append(
                    {
                        "real_transition_count": real_count,
                        "replay_size": schedule["replay_size_before_updates"],
                        "updates_executed": schedule["updates_executed"],
                        "episode_terminal": int(completed_now),
                    }
                )
            if schedule["pid_updated"]:
                pid_update_count += 1
                lambda_before = schedule["lambda_before"]
                lambda_after = schedule["lambda_after"]
                updates_before_pid = schedule["updates_before_pid"]
                updates_after_pid = schedule["updates_after_pid"]

            if completed_now:
                assert completed is not None
                assert exposure is not None
                assert lambda_before is not None and lambda_after is not None
                assert updates_before_pid is not None and updates_after_pid is not None
                assert actor_terminal is not None
                actor_after = _module_sha256(policy.actor)
                episode_diagnostics.append(
                    _episode_row(
                        condition=condition,
                        recorder=recorder,
                        transition_count=real_count,
                        completed=completed,
                        exposure=exposure,
                        lambda_before=lambda_before,
                        lambda_after=lambda_after,
                        updates_before_pid=updates_before_pid,
                        updates_after_pid=updates_after_pid,
                        updates_after_episode=schedule["updates_after_episode"],
                        alpha=alpha_value(policy),
                        actor_start=current_episode_start_actor,
                        actor_end=actor_terminal,
                        actor_after_processing=actor_after,
                        losses=_losses(logger),
                        episode_update_events=current_episode_update_events,
                        replay_size_at_terminal=len(collector.buffer),
                    )
                )
                atomic_json(
                    run_directory / "training-episode-diagnostics.json",
                    episode_diagnostics,
                )
                previous_completed = recorder.completed_episodes
                current_episode_start_actor = actor_after
                current_episode_start_transition = real_count
                current_episode_update_events = []

            scientific = real_count in checkpoints
            recovery = (
                real_count % configuration.raw["recovery_checkpoint_interval"] == 0
            )
            if scientific:
                diagnostic = evaluate_diagnostic(
                    policy, configuration, real_transitions=real_count
                )
                checkpoint_state = {
                    "real_transitions": real_count,
                    "completed_training_episodes": recorder.completed_episodes,
                    "pid_update_count": pid_update_count,
                    "lambda_speed": multiplier_values(policy)[0],
                    "lambda_deadline": multiplier_values(policy)[1],
                    "alpha": alpha_value(policy),
                    "learner_losses": _losses(logger),
                    "exposure": exposure_summary(recorder),
                }
                diagnostic["training_state"] = checkpoint_state
                directory = run_directory / f"step-{real_count:06d}"
                atomic_json(directory / "diagnostic.json", diagnostic)
                diagnostic_completed.add(real_count)
            elif recovery:
                directory = run_directory / "recovery"

            if scientific or recovery:
                walltime = {
                    "completed_segments_s": walltime_state["completed_segments_s"]
                    + (time.perf_counter() - segment_started)
                }
                checkpoint = directory / "checkpoint.pt"
                payload = _checkpoint_payload(
                    condition=condition,
                    attempt_id=lease.attempt_id,
                    policy=policy,
                    collector=collector,
                    recorder=recorder,
                    counters=counters,
                    episode_diagnostics=episode_diagnostics,
                    diagnostic_completed=diagnostic_completed,
                    previous_completed_episodes=previous_completed,
                    pid_update_count=pid_update_count,
                    current_episode_start_actor_sha256=current_episode_start_actor,
                    current_episode_start_transition=current_episode_start_transition,
                    current_episode_update_events=current_episode_update_events,
                    walltime_state=walltime,
                )
                metadata = write_verified_checkpoint(checkpoint, payload)
                last_checkpoint = str(checkpoint)
                atomic_json(run_directory / "recovery/latest.json", metadata)
                lease.record_checkpoint(
                    counters,
                    metadata,
                    episode_count=recorder.completed_episodes,
                    pid_update_count=pid_update_count,
                    diagnostic_completed=diagnostic_completed,
                )
                if scientific:
                    final_checkpoint = checkpoint

            if real_count % HEARTBEAT_INTERVAL == 0 or scientific or recovery:
                atomic_json(
                    run_directory / "heartbeat.json",
                    _heartbeat(
                        condition=condition,
                        attempt_id=lease.attempt_id,
                        counters=counters,
                        recorder=recorder,
                        pid_update_count=pid_update_count,
                        status="RUNNING",
                        last_checkpoint=last_checkpoint,
                    ),
                )

        if counters["rl_gradient_updates"] != 30_000:
            raise RuntimeError("Final optimizer budget is not exactly 30,000")
        if pid_update_count != recorder.completed_episodes:
            raise RuntimeError("Every completed real episode must update PID once")
        if final_checkpoint is None:
            raise RuntimeError("Final scientific checkpoint is absent")
        lease.complete(
            counters,
            final_checkpoint,
            episode_count=recorder.completed_episodes,
            pid_update_count=pid_update_count,
            diagnostic_completed=diagnostic_completed,
        )
        atomic_json(
            run_directory / "heartbeat.json",
            _heartbeat(
                condition=condition,
                attempt_id=lease.attempt_id,
                counters=counters,
                recorder=recorder,
                pid_update_count=pid_update_count,
                status="COMPLETED",
                last_checkpoint=last_checkpoint,
            ),
        )
        return final_checkpoint


def run_matrix(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cpu",
    threads: int = 4,
) -> list[Path]:
    preparation_status()
    output_root = Path(output_root)
    results = []
    for condition in load_configuration().conditions:
        _path, manifest = load_or_create_manifest(output_root)
        state = manifest["runs"][condition.name]["state"]
        if state == "COMPLETED":
            continue
        if state == "NOT_STARTED":
            results.append(
                run_policy(
                    condition_name=condition.name,
                    output_root=output_root,
                    device=device,
                    threads=threads,
                )
            )
            continue
        if state == "INTERRUPTED":
            metadata = manifest["runs"][condition.name].get("last_verified_checkpoint")
            if not metadata:
                raise RuntimeError("FRESH_RESTART_REQUIRED")
            results.append(
                run_policy(
                    condition_name=condition.name,
                    output_root=output_root,
                    device=device,
                    threads=threads,
                    resume=True,
                    checkpoint_path=metadata["path"],
                )
            )
            continue
        raise RuntimeError(f"Cannot automatically recover {condition.name}: {state}")
    run_validation(output_root=output_root, device=device, threads=threads)
    from .finalize import finalize_results

    finalize_results(output_root)
    return results


def run_validation(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cpu",
    threads: int = 4,
) -> None:
    """Open Validation once, only after all six final checkpoints verify."""

    preparation_status()
    configuration = load_configuration()
    output_root = Path(output_root)
    manifest_path, manifest = load_or_create_manifest(output_root)
    checkpoint_payloads: dict[str, dict[str, Any]] = {}
    for condition in configuration.conditions:
        run = manifest["runs"][condition.name]
        if run["state"] != "COMPLETED":
            raise PermissionError(
                "Validation remains sealed until all six policies complete"
            )
        metadata = run.get("last_verified_checkpoint")
        if not metadata or int(metadata["transition_count"]) != 300_000:
            raise RuntimeError("A final 300k verified checkpoint is missing")
        if file_sha256(metadata["path"]) != run["final_checkpoint_sha256"]:
            raise RuntimeError("A frozen final checkpoint hash changed")
        checkpoint_payloads[condition.name] = load_verified(
            metadata["path"],
            expected_sha256=run["final_checkpoint_sha256"],
            expected_configuration_sha256=configuration.configuration_sha256,
        )

    validation = manifest["validation"]
    if validation["state"] == "COMPLETED":
        if validation["episode_count"] != 108:
            raise RuntimeError("Completed Validation count changed")
        return
    if validation["state"] == "FROZEN_UNOPENED":
        validation.update({"state": "OPEN", "opened_at": utc_now()})
        atomic_json(manifest_path, manifest)
    elif validation["state"] != "OPEN":
        raise RuntimeError("Unknown Validation lifecycle state")

    for condition in configuration.conditions:
        destination = output_root / condition.name / "validation.json"
        if condition.name in validation["completed_conditions"]:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing.get("episode_count") != 18:
                raise RuntimeError("Recorded Validation artifact is incomplete")
            continue
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if (
                existing.get("evaluation_id") != "COLLECTOR_VALIDATION_V1"
                or existing.get("episode_count") != 18
            ):
                raise RuntimeError("Untracked Validation artifact cannot be trusted")
            validation["completed_conditions"].append(condition.name)
            validation["episode_count"] += 18
            atomic_json(manifest_path, manifest)
            continue

        environment = build_training_environment(
            configuration,
            training_seed=condition.training_seed,
            max_simulator_steps=1,
        )
        try:
            agent, _logger = build_agent(
                configuration.v2,
                environment,
                training_seed=condition.training_seed,
                device=device,
                threads=threads,
            )
            restore_policy(agent.policy, checkpoint_payloads[condition.name])
            result = evaluate_validation(agent.policy, configuration)
        finally:
            environment.close()
        result["condition"] = condition.name
        result["collector"] = condition.collector
        result["nominal_seed"] = condition.training_seed
        result["final_checkpoint_sha256"] = manifest["runs"][condition.name][
            "final_checkpoint_sha256"
        ]
        atomic_json(destination, result)
        validation["completed_conditions"].append(condition.name)
        validation["episode_count"] += int(result["episode_count"])
        atomic_json(manifest_path, manifest)

    if (
        tuple(validation["completed_conditions"]) != CONDITION_NAMES
        or validation["episode_count"] != 108
    ):
        raise RuntimeError("Validation matrix is not exactly 6 x 18")
    validation.update({"state": "COMPLETED", "completed_at": utc_now()})
    atomic_json(manifest_path, manifest)
