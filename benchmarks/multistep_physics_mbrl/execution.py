"""Hardened six-policy runner, provenance, recovery, and launch preparation."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import platform
import plistlib
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

from benchmarks.constrained_rl.adapter import logger_snapshot
from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.scalar_sac.experiment import _base_environment

from .adapter import (
    RealEpisodePIDGate,
    RealPhysicsReplay,
    RealTransitionRecorder,
    TrainingSeedWrapper,
)
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
from .config import CONDITIONS, SOURCE_COMMIT, TRAINING_SEEDS, load_configuration
from .diagnostics import (
    TargetAccumulator,
    anticipatory_speed_diagnostic,
    downward_limit_probe_metadata,
)
from .evaluation import evaluate_development
from .value_expansion import real_replay_update

DEFAULT_OUTPUT_ROOT = Path("runs/multistep-physics-mbrl-v1")
HEARTBEAT_INTERVAL = 250
SCIENTIFIC_FILES = (
    "canonical.json",
    "DESIGN.md",
    "PROTOCOL.md",
    "config.py",
    "adapter.py",
    "physics_rollout.py",
    "value_expansion.py",
    "diagnostics.py",
    "evaluation.py",
    "checkpointing.py",
    "execution.py",
    "runner.py",
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


def assert_source_ancestor(repository: str | Path = ".") -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, "HEAD"],
        cwd=repository,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError("Study does not descend from the completed H=1 commit")


def source_hashes(repository: str | Path = ".") -> dict[str, str]:
    root = Path(repository) / "benchmarks" / "multistep_physics_mbrl"
    return {name: file_sha256(root / name) for name in SCIENTIFIC_FILES}


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


class TrainingTrackStream:
    """Checkpointable RNG that rejects every research split."""

    def __init__(self, training_seed: int):
        self.rng = np.random.default_rng(
            np.random.SeedSequence([training_seed, 0x4D564531])
        )
        self.draw_count = 0
        self.excluded = frozenset(load_configuration().raw["track_splits"]["excluded"])

    def next_seed(self) -> int:
        while True:
            value = int(self.rng.integers(0, 2**31 - 1))
            if value not in self.excluded:
                self.draw_count += 1
                return value

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "draw_count": self.draw_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]
        self.draw_count = int(state["draw_count"])


def initial_manifest() -> dict[str, Any]:
    configuration = load_configuration()
    runs = {
        f"{condition}-seed-{seed}": {
            "condition": condition,
            "training_seed": seed,
            "state": "NOT_STARTED",
            "attempts": [],
            "real_transitions": 0,
            "synthetic_transitions": 0,
            "model_target_transitions": 0,
            "rl_gradient_updates": 0,
            "final_checkpoint_sha256": None,
            "development_complete": False,
        }
        for condition in CONDITIONS
        for seed in TRAINING_SEEDS
    }
    return {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "configuration_sha256": configuration.configuration_sha256,
        "source_commit": SOURCE_COMMIT,
        "created_at": utc_now(),
        "runs": runs,
        "validation": {"state": "SEALED", "opened_at": None},
        "paper_test": {"state": "SEALED", "opened_at": None},
    }


def load_or_create_manifest(output_root: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(output_root) / "manifest.json"
    configuration = load_configuration()
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["configuration_sha256"] != configuration.configuration_sha256:
            raise RuntimeError("Existing manifest uses a different protocol")
        return path, manifest
    manifest = initial_manifest()
    atomic_json(path, manifest)
    return path, manifest


class RunLease(AbstractContextManager):
    """Reject duplicate starts and preserve interrupted-attempt provenance."""

    def __init__(
        self, output_root: str | Path, run_id: str, *, resume: bool = False
    ) -> None:
        self.output_root = Path(output_root)
        self.run_id = run_id
        self.resume = resume
        self.attempt_id = str(uuid.uuid4())
        self.lock_path: Path | None = None
        self.manifest_path: Path | None = None
        self.manifest: dict[str, Any] | None = None
        self.attempt_index = -1

    def __enter__(self):
        self.manifest_path, self.manifest = load_or_create_manifest(self.output_root)
        run = self.manifest["runs"][self.run_id]
        allowed = "INTERRUPTED" if self.resume else "NOT_STARTED"
        if run["state"] != allowed:
            raise RuntimeError(f"Cannot start {self.run_id} from {run['state']}")
        if self.resume:
            if not run["attempts"] or not run.get("last_verified_checkpoint"):
                raise PermissionError("Resume requires a verified recovery checkpoint")
            attempt = run["attempts"][-1]
            self.attempt_id = attempt["attempt_id"]
            attempt.setdefault("resume_events", []).append(
                {"at": utc_now(), "pid": os.getpid()}
            )
            attempt["state"] = "RUNNING"
            self.attempt_index = len(run["attempts"]) - 1
        else:
            attempt = {
                "attempt_number": len(run["attempts"]) + 1,
                "attempt_id": self.attempt_id,
                "state": "RUNNING",
                "started_at": utc_now(),
            }
            run["attempts"].append(attempt)
            self.attempt_index = len(run["attempts"]) - 1
        run_directory = self.output_root / self.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        self.lock_path = run_directory / ".run.lock"
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
        self, counters: dict[str, int], metadata: dict[str, Any]
    ) -> None:
        assert self.manifest is not None and self.manifest_path is not None
        run = self.manifest["runs"][self.run_id]
        run.update({key: int(value) for key, value in counters.items()})
        run["last_verified_checkpoint"] = metadata
        self.attempt["last_verified_checkpoint"] = metadata
        atomic_json(self.manifest_path, self.manifest)

    def complete(self, counters: dict[str, int], final_checkpoint: Path) -> None:
        assert self.manifest is not None and self.manifest_path is not None
        run = self.manifest["runs"][self.run_id]
        if counters["real_transitions"] != 300_000:
            raise RuntimeError(
                "Completed policy must use exactly 300k real transitions"
            )
        if counters["rl_gradient_updates"] != 30_000:
            raise RuntimeError("Completed policy must use exactly 30k RL updates")
        if counters["synthetic_transitions"] != 0:
            raise RuntimeError("Synthetic replay appeared in the primary study")
        run.update({key: int(value) for key, value in counters.items()})
        run["development_complete"] = True
        run["final_checkpoint_sha256"] = file_sha256(final_checkpoint)
        run["state"] = "COMPLETED"
        self.attempt.update({"state": "COMPLETED", "ended_at": utc_now()})
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


def heartbeat_payload(
    *,
    condition: str,
    training_seed: int,
    attempt_id: str,
    counters: dict[str, int],
    recorder: RealTransitionRecorder,
    status: str,
    last_checkpoint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "condition": condition,
        "training_seed": training_seed,
        "attempt_id": attempt_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "session_id": os.getsid(0),
        **{key: int(value) for key, value in counters.items()},
        "completed_real_episodes": recorder.completed_episodes,
        "current_episode_step": recorder.episode_step,
        "last_verified_checkpoint": last_checkpoint,
        "status": status,
    }


def _checkpoint_payload(
    *,
    condition: str,
    training_seed: int,
    attempt_id: str,
    policy: Any,
    collector: Any,
    recorder: RealTransitionRecorder,
    sidecar: RealPhysicsReplay,
    track_stream: TrainingTrackStream,
    counters: dict[str, int],
    diagnostics: list[dict[str, Any]],
    target_accumulator: TargetAccumulator,
    anticipation_probes: list[Any],
    rollout_rng: np.random.Generator,
    pid_gate: RealEpisodePIDGate,
    previous_completed_episodes: int,
    development_checkpoints_completed: set[int],
) -> dict[str, Any]:
    configuration = load_configuration()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "condition": condition,
        "training_seed": training_seed,
        "attempt_id": attempt_id,
        "configuration_sha256": configuration.configuration_sha256,
        "source_hashes": source_hashes(),
        "runtime_versions": runtime_versions(),
        "policy": policy.state_dict(),
        "policy_optimizers": policy_optimizer_state(policy),
        "lagrangian_states": [item.state_dict() for item in policy.lag_optims],
        "real_replay": collector.buffer,
        "physics_sidecar": sidecar,
        "environment_state": copy.deepcopy(recorder),
        "collector_state": collector_state(collector),
        "track_stream_state": track_stream.state_dict(),
        "counters": dict(counters),
        "diagnostics": diagnostics,
        "target_accumulator": target_accumulator,
        "anticipation_probes": anticipation_probes,
        "rng_states": capture_rng_states(rollout_rng),
        "pid_real_episode_updates": pid_gate.real_episode_updates,
        "previous_completed_episodes": previous_completed_episodes,
        "development_checkpoints_completed": sorted(
            development_checkpoints_completed
        ),
    }


def run_policy(
    *,
    condition: str,
    training_seed: int,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cpu",
    threads: int = 4,
    resume: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Execute one policy only after a separate authorization changes the freeze."""

    configuration = load_configuration()
    authorization = configuration.raw["authorization"]
    if not (
        authorization["main_training_authorized"]
        and authorization["main_training_enabled"]
    ):
        raise PermissionError(
            "Main six-policy training is implemented but not authorized"
        )
    if condition not in CONDITIONS or training_seed not in TRAINING_SEEDS:
        raise ValueError("Condition or training seed is outside the frozen matrix")
    if resume != (checkpoint_path is not None):
        raise ValueError("Exact resume requires one explicit checkpoint")

    import tianshou
    from tianshou.data import Collector, ReplayBuffer

    if tianshou.__version__ != configuration.v2.algorithm.tianshou_version:
        raise RuntimeError("Pinned Tianshou version changed")
    run_id = f"{condition}-seed-{training_seed}"
    with RunLease(output_root, run_id, resume=resume) as lease:
        track_stream = TrainingTrackStream(training_seed)
        seeded = TrainingSeedWrapper(_base_environment(configuration.v2), track_stream)
        dense = DenseDeadlineTaskWrapper(
            seeded,
            task=configuration.v2.task,
            energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
            deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
            max_simulator_steps=configuration.v2.simulator_step_budget,
        )
        recorder = RealTransitionRecorder(dense)
        agent, _logger = build_agent(
            configuration.v2,
            recorder,
            training_seed=training_seed,
            device=device,
            threads=threads,
        )
        policy = agent.policy
        policy.train()
        collector = Collector(
            policy,
            recorder,
            ReplayBuffer(configuration.v2.algorithm.buffer_size),
            exploration_noise=True,
        )
        base = recorder.unwrapped
        sidecar = RealPhysicsReplay(configuration.v2.algorithm.buffer_size)
        rollout_rng = np.random.default_rng(
            np.random.SeedSequence([training_seed, 0x4D564552])
        )
        pid_gate = RealEpisodePIDGate()
        target_accumulator = TargetAccumulator()
        counters = {
            "real_transitions": 0,
            "synthetic_transitions": 0,
            "model_target_transitions": 0,
            "rl_gradient_updates": 0,
        }
        diagnostics: list[dict[str, Any]] = []
        anticipation_probes: list[Any] = []
        anticipation_track_signatures: set[tuple[bytes, bytes]] = set()
        previous_completed = 0
        development_completed: set[int] = set()
        checkpoints = set(configuration.raw["real_transition_checkpoints"])
        final_checkpoint: Path | None = None
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
                raise RuntimeError("Checkpoint belongs to another attempt")
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
                replay_buffer=payload["real_replay"],
            )
            recorder = restored_recorder
            dense = recorder.env
            track_stream = dense.env.stream
            track_stream.load_state_dict(payload["track_stream_state"])
            base = recorder.unwrapped
            sidecar = payload["physics_sidecar"]
            counters = dict(payload["counters"])
            diagnostics = payload["diagnostics"]
            target_accumulator = payload["target_accumulator"]
            anticipation_probes = payload["anticipation_probes"]
            anticipation_track_signatures = {
                (
                    probe.context.track.positions_m.tobytes(),
                    probe.context.track.limits_m_s.tobytes(),
                )
                for probe in anticipation_probes
            }
            previous_completed = int(payload["previous_completed_episodes"])
            pid_gate.real_episode_updates = int(payload["pid_real_episode_updates"])
            development_completed = set(
                payload["development_checkpoints_completed"]
            )
            restore_rng_states(payload["rng_states"], rollout_rng)
            last_checkpoint = str(checkpoint_path)

        run_directory = Path(output_root) / run_id
        atomic_json(
            run_directory / "heartbeat.json",
            heartbeat_payload(
                condition=condition,
                training_seed=training_seed,
                attempt_id=lease.attempt_id,
                counters=counters,
                recorder=recorder,
                status="RUNNING",
                last_checkpoint=last_checkpoint,
            ),
        )

        model_steps = (
            0
            if condition == "matched_real_only"
            else configuration.model_expansion_steps
        )
        while counters["real_transitions"] < 300_000:
            stats = collector.collect(n_step=1)
            records = recorder.drain()
            if len(records) != 1 or int(stats["n/st"]) != 1:
                raise RuntimeError("Native transition accounting changed")
            sidecar.append(records[0])
            if len(anticipation_probes) < 9:
                probe_metadata = downward_limit_probe_metadata(records[0])
                if probe_metadata is not None:
                    probe_signature = (
                        records[0].context.track.positions_m.tobytes(),
                        records[0].context.track.limits_m_s.tobytes(),
                    )
                    if probe_signature not in anticipation_track_signatures:
                        anticipation_probes.append(records[0])
                        anticipation_track_signatures.add(probe_signature)
            counters["real_transitions"] += 1
            real_count = counters["real_transitions"]

            if recorder.completed_episodes != previous_completed:
                completed = dense.last_completed_episode
                if completed is None:
                    raise RuntimeError("Completed physical episode was lost")
                pid_gate.update(
                    policy,
                    source="real_completed_episode",
                    stats={"cost": np.asarray(completed.costs)},
                )
                policy.post_update_fn(stats_train={"cost": completed.costs})
                previous_completed = recorder.completed_episodes

            target_updates = math.floor(
                real_count * configuration.v2.algorithm.update_per_step + 1e-12
            )
            while (
                counters["rl_gradient_updates"] < target_updates
                and len(collector.buffer) >= configuration.v2.algorithm.batch_size
            ):
                decomposition = real_replay_update(
                    policy,
                    collector.buffer,
                    sidecar,
                    model_steps=model_steps,
                    rollout_rng=rollout_rng,
                    vehicle=base.vehicle,
                    simulation_config=base.config,
                    batch_size=configuration.v2.algorithm.batch_size,
                    energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
                    deadline_s=configuration.v2.task.max_time_s,
                    max_episode_steps=configuration.v2.max_episode_steps,
                )
                if decomposition is not None:
                    target_accumulator.add(decomposition)
                    counters["model_target_transitions"] += int(
                        decomposition.model_steps.sum()
                    )
                counters["rl_gradient_updates"] += 1

            scientific = real_count in checkpoints
            recovery = (
                real_count % configuration.raw["recovery_checkpoint_interval"] == 0
            )
            if scientific:
                episodes, summary = evaluate_development(policy, configuration)
                directory = run_directory / f"step-{real_count:06d}"
                atomic_json(
                    directory / "development.json",
                    {
                        "episodes": [asdict(item) for item in episodes],
                        "summary": asdict(summary),
                        "actor_only": True,
                    },
                )
                diagnostics.append(target_accumulator.drain(real_transitions=real_count))
                probe_rows = []
                for probe_index, probe in enumerate(anticipation_probes):
                    metadata = downward_limit_probe_metadata(probe)
                    if metadata is None:
                        raise RuntimeError("A fixed downward-limit probe changed")
                    diagnostic_rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [training_seed, real_count, probe_index, 0x4D564544]
                        )
                    )
                    probe_rows.append(
                        {
                            "probe_index": probe_index,
                            **metadata,
                            "predicted_discounted_speed_cost": (
                                anticipatory_speed_diagnostic(
                                    policy=policy,
                                    state=probe.state,
                                    context=probe.context,
                                    horizon=configuration.model_expansion_steps,
                                    rollout_rng=diagnostic_rng,
                                    vehicle=base.vehicle,
                                    simulation_config=base.config,
                                    gamma=configuration.v2.algorithm.gamma,
                                )
                            ),
                        }
                    )
                diagnostics.append(
                    {
                        "record_type": "anticipatory_speed",
                        "real_transitions": real_count,
                        "controls_actions": False,
                        "fixed_probe_count": len(probe_rows),
                        "probes": probe_rows,
                    }
                )
                diagnostics.append(
                    {
                        "record_type": "learner_stability",
                        "real_transitions": real_count,
                        "lambda_speed": float(policy.lag_optims[0].get_lag()),
                        "lambda_deadline": float(policy.lag_optims[1].get_lag()),
                        "alpha": float(policy._alpha.detach().cpu()),  # noqa: SLF001
                        "learner_metrics": logger_snapshot(policy.logger),
                    }
                )
                development_completed.add(real_count)
            elif recovery:
                directory = run_directory / "recovery" / f"step-{real_count:06d}"

            if scientific or recovery:
                checkpoint = directory / "checkpoint.pt"
                payload = _checkpoint_payload(
                    condition=condition,
                    training_seed=training_seed,
                    attempt_id=lease.attempt_id,
                    policy=policy,
                    collector=collector,
                    recorder=recorder,
                    sidecar=sidecar,
                    track_stream=track_stream,
                    counters=counters,
                    diagnostics=diagnostics,
                    target_accumulator=target_accumulator,
                    anticipation_probes=anticipation_probes,
                    rollout_rng=rollout_rng,
                    pid_gate=pid_gate,
                    previous_completed_episodes=previous_completed,
                    development_checkpoints_completed=development_completed,
                )
                metadata = write_verified_checkpoint(checkpoint, payload)
                last_checkpoint = str(checkpoint)
                atomic_json(run_directory / "recovery" / "latest.json", metadata)
                lease.record_checkpoint(counters, metadata)
                if scientific:
                    final_checkpoint = checkpoint

            if real_count % HEARTBEAT_INTERVAL == 0 or scientific or recovery:
                atomic_json(
                    run_directory / "heartbeat.json",
                    heartbeat_payload(
                        condition=condition,
                        training_seed=training_seed,
                        attempt_id=lease.attempt_id,
                        counters=counters,
                        recorder=recorder,
                        status="RUNNING",
                        last_checkpoint=last_checkpoint,
                    ),
                )

        if final_checkpoint is None or development_completed != checkpoints:
            raise RuntimeError("Final scientific checkpoint schedule is incomplete")
        lease.complete(counters, final_checkpoint)
        atomic_json(
            run_directory / "heartbeat.json",
            heartbeat_payload(
                condition=condition,
                training_seed=training_seed,
                attempt_id=lease.attempt_id,
                counters=counters,
                recorder=recorder,
                status="COMPLETED",
                last_checkpoint=last_checkpoint,
            ),
        )
        return final_checkpoint


def prepare_launchd_job(
    *,
    repository: str | Path,
    output_root: str | Path,
    python: str,
) -> Path:
    """Prepare, but never bootstrap, the detached serial six-policy job."""

    root = Path(repository).resolve()
    runs = Path(output_root)
    if not runs.is_absolute():
        runs = root / runs
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / "launchd.plist"
    payload = {
        "Label": "org.longicontrol.multistep-physics-mbrl.v1",
        "ProgramArguments": [
            python,
            "-m",
            "benchmarks.multistep_physics_mbrl.runner",
            "matrix",
            "--output-root",
            str(runs),
        ],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": {
            "PYTHONPATH": str(root),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "StandardOutPath": str(runs / "launcher.stdout.log"),
        "StandardErrorPath": str(runs / "launcher.stderr.log"),
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Background",
        "AbandonProcessGroup": False,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(plistlib.dumps(payload, sort_keys=True))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    atomic_json(
        runs / "launcher.json",
        {
            "schema_version": 1,
            "state": "PREPARED_NOT_LAUNCHED",
            "created_at_utc": utc_now(),
            "plist_path": str(destination),
            "matrix": [
                [condition, seed] for condition in CONDITIONS for seed in TRAINING_SEEDS
            ],
            "main_training_authorized": load_configuration().raw["authorization"][
                "main_training_authorized"
            ],
        },
    )
    return destination


def benchmark_environment_steps(iterations: int = 2000) -> dict[str, float]:
    """Measure real actor/environment collection without consuming research tracks."""

    configuration = load_configuration()
    environment = _base_environment(configuration.v2)
    observation, _info = environment.reset(seed=0)
    started = time.perf_counter()
    completed = 0
    for _ in range(iterations):
        observation, _reward, terminated, truncated, _info = environment.step(
            np.zeros(1, dtype=np.float64)
        )
        completed += 1
        if terminated or truncated:
            observation, _info = environment.reset(seed=0)
    elapsed = time.perf_counter() - started
    environment.close()
    del observation
    return {
        "transitions": completed,
        "seconds": elapsed,
        "transitions_per_second": completed / elapsed,
    }
