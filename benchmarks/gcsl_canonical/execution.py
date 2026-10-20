"""Hardened control plane, checkpoint schedule, progress, and launch status."""

from __future__ import annotations

import json
import os
import platform
import plistlib
import socket
import subprocess
import sys
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpointing import file_sha256, runtime_versions
from .config import (
    CHECKPOINTS,
    TRAINING_SEEDS,
    load_configuration,
    require_main_training_authorized,
)
from .evaluation import evaluate_diagnostic
from .progress import status_progress
from .runner import CANONICAL_TASK, PolicyRunner

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("runs/gcsl-canonical-v1")
EXPECTED_BRANCH = "research/gcsl-canonical"
BASE_COMMIT = "78a50b7ad866beb72a93f06cb1c5d6bcaa25cc12"
HEARTBEAT_INTERVAL = 250
RECOVERY_INTERVAL = 10_000
SCIENTIFIC_FILES = (
    "canonical.json",
    "upstream.json",
    "historical_overlap_ledger.json",
    "config.py",
    "goal_adapter.py",
    "policy.py",
    "learner.py",
    "replay.py",
    "sampling.py",
    "split.py",
)
EXECUTION_FILES = (
    "checkpointing.py",
    "cli.py",
    "diagnostics.py",
    "evaluation.py",
    "execution.py",
    "progress.py",
    "reference.py",
    "resource_check.py",
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


def _git(*arguments: str, repository: str | Path = ".") -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def source_hashes() -> dict[str, str]:
    return {
        f"benchmarks/gcsl_canonical/{name}": file_sha256(HERE / name)
        for name in (*SCIENTIFIC_FILES, *EXECUTION_FILES)
    }


def verify_preflight(
    repository: str | Path,
    *,
    require_clean: bool = True,
    require_training_authorization: bool = True,
) -> dict[str, Any]:
    root = Path(repository).resolve()
    if _git("branch", "--show-current", repository=root) != EXPECTED_BRANCH:
        raise RuntimeError(f"runner must execute on {EXPECTED_BRANCH}")
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"),
        cwd=root,
        check=True,
    )
    if require_clean and _git("status", "--porcelain", repository=root):
        raise RuntimeError("working-tree changes block canonical GCSL execution")
    configuration = load_configuration()
    if require_training_authorization:
        require_main_training_authorized(configuration)
    status = json.loads((HERE / "preparation_status.json").read_text(encoding="utf-8"))
    if not status.get("all_scientific_readiness_flags_true"):
        raise RuntimeError("canonical GCSL readiness gates are incomplete")
    hashes = source_hashes()
    if status.get("scientific_source_hashes") != hashes:
        raise RuntimeError("frozen GCSL source hashes changed")
    return {
        "branch": EXPECTED_BRANCH,
        "base_commit": BASE_COMMIT,
        "execution_commit": _git("rev-parse", "HEAD", repository=root),
        "configuration_sha256": configuration.configuration_sha256,
        "source_hashes": hashes,
        "runtime_versions": runtime_versions(),
        "platform": platform.platform(),
    }


def initial_manifest(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "canonical-goal-conditioned-supervised-learning-v1",
        "state": "NOT_STARTED",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "provenance": provenance,
        "execution_order": list(TRAINING_SEEDS),
        "runs": {
            f"G{seed}": {
                "training_seed": seed,
                "state": "NOT_STARTED",
                "attempts": [],
                "real_transitions": 0,
                "supervised_updates": 0,
                "diagnostic_checkpoints_completed": [],
            }
            for seed in TRAINING_SEEDS
        },
        "validation": {"state": "FROZEN_UNOPENED", "episode_count": 0},
        "paper": {"state": "SEALED_UNINSPECTED", "episode_count": 0},
    }


class RunLease(AbstractContextManager):
    def __init__(self, output_root: Path):
        self.path = output_root / "ACTIVE.lock"
        self.acquired = False

    def __enter__(self):
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": utc_now(),
        }
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            raise RuntimeError("duplicate canonical GCSL process blocked") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.acquired and self.path.exists():
            self.path.unlink()
        return False


def load_or_create_manifest(
    output_root: Path, provenance: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["provenance"] != provenance:
            raise RuntimeError("existing study manifest provenance changed")
        return path, manifest
    manifest = initial_manifest(provenance)
    atomic_json(path, manifest)
    return path, manifest


def _heartbeat(
    runner: PolicyRunner, run_directory: Path, latest_rsr: int
) -> dict[str, Any]:
    support = runner.replay.support_summary(CANONICAL_TASK)
    summary = runner.diagnostics.summary()
    outcomes = runner.outcomes_summary()
    payload = {
        "updated_at": utc_now(),
        "training_seed": runner.training_seed,
        "attempt_id": runner.attempt_id,
        "real_transitions": runner.transition_count,
        "supervised_updates": runner.update_count,
        "completed_episodes": outcomes["completed_training_episodes"],
        "canonical_training_successes": outcomes["canonical_training_successes"],
        "latest_diagnostic_rsr": int(latest_rsr),
        "maximum_replay_progress": support["maximum_replay_progress"],
        "canonical_hindsight_tuples": summary["canonical_hindsight_tuples"],
        "walltime_s": runner.walltime_s(),
        "status": (
            "HEALTHY" if summary["nonfinite_value_count"] == 0 else "CHECK_REQUIRED"
        ),
    }
    atomic_json(run_directory / "heartbeat.json", payload)
    return payload


def _technical_checkpoint(runner: PolicyRunner, run_directory: Path) -> dict[str, Any]:
    return runner.save(run_directory / "recovery" / "checkpoint.pkl")


def _scientific_checkpoint(
    runner: PolicyRunner,
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    transition = runner.transition_count
    final = run_directory / f"step-{transition:06d}"
    if final.exists():
        raise FileExistsError(f"scientific checkpoint exists: {final}")
    pending = run_directory / f".step-{transition:06d}.pending-{uuid.uuid4().hex}"
    pending.mkdir(parents=True)
    result = evaluate_diagnostic(runner.learner)
    runner.diagnostic_checkpoints_completed.append(transition)
    try:
        metadata = runner.save(pending / "checkpoint.pkl")
        support = runner.replay.support_summary(CANONICAL_TASK)
        atomic_json(pending / "diagnostic.json", result)
        atomic_json(
            pending / "training-diagnostics.json",
            {
                "transition": transition,
                "support": support,
                "learning": runner.diagnostics.summary(),
                "latest_update": runner.diagnostics.rows[-1],
                "training_outcomes": runner.outcomes_summary(),
            },
        )
        pending.replace(final)
    except BaseException:
        runner.diagnostic_checkpoints_completed.remove(transition)
        raise
    metadata["path"] = str(final / "checkpoint.pkl")
    return metadata, result, support


def run_policy(
    *,
    seed: int,
    output_root: Path,
    provenance: dict[str, Any],
    manifest: dict[str, Any],
    resume_checkpoint: Path | None = None,
) -> Path:
    key = f"G{seed}"
    record = manifest["runs"][key]
    run_directory = output_root / key
    if resume_checkpoint is None:
        if record["state"] != "NOT_STARTED" or run_directory.exists():
            raise RuntimeError(f"duplicate or fresh restart blocked for {key}")
        attempt_id = str(uuid.uuid4())
        run_directory.mkdir(parents=True)
        runner = PolicyRunner(
            training_seed=seed,
            attempt_id=attempt_id,
            configuration_sha256=provenance["configuration_sha256"],
            source_hashes=provenance["source_hashes"],
        )
        attempt = {
            "attempt_id": attempt_id,
            "attempt_number": 1,
            "state": "RUNNING",
            "started_at": utc_now(),
            "resume_events": [],
        }
        record["attempts"].append(attempt)
        print(f"G{seed} | policy started | attempt {attempt_id}", flush=True)
    else:
        if record["state"] != "INTERRUPTED":
            raise RuntimeError("same-Attempt resume requires INTERRUPTED state")
        attempt = record["attempts"][-1]
        attempt_id = attempt["attempt_id"]
        latest = record["last_verified_checkpoint"]
        if file_sha256(resume_checkpoint) != latest["sha256"]:
            raise RuntimeError("resume checkpoint differs from recorded attempt")
        runner = PolicyRunner.restore(
            resume_checkpoint,
            expected_sha256=latest["sha256"],
            configuration_sha256=provenance["configuration_sha256"],
            source_hashes=provenance["source_hashes"],
            attempt_id=attempt_id,
        )
        attempt["state"] = "RUNNING"
        attempt["resume_events"].append(
            {"resumed_at": utc_now(), "checkpoint_sha256": latest["sha256"]}
        )
        print(f"G{seed} | policy resumed | attempt {attempt_id}", flush=True)
    record["state"] = "RUNNING"
    manifest["updated_at"] = utc_now()
    atomic_json(output_root / "manifest.json", manifest)
    latest_rsr = 0
    try:
        while runner.transition_count < 300_000:
            runner.advance()
            if runner.transition_count % HEARTBEAT_INTERVAL == 0:
                _heartbeat(runner, run_directory, latest_rsr)
            if runner.transition_count % RECOVERY_INTERVAL != 0:
                continue
            if runner.transition_count in CHECKPOINTS:
                metadata, diagnostic, support = _scientific_checkpoint(
                    runner, run_directory
                )
                latest_rsr = int(diagnostic["summary"]["diagnostic_rsr"])
                record["diagnostic_checkpoints_completed"].append(
                    runner.transition_count
                )
                record.setdefault("scientific_checkpoints", []).append(
                    {
                        **metadata,
                        "diagnostic_rsr": latest_rsr,
                        "support": support,
                    }
                )
                heartbeat = _heartbeat(runner, run_directory, latest_rsr)
                print(status_progress(heartbeat), flush=True)
            else:
                metadata = _technical_checkpoint(runner, run_directory)
            record["last_verified_checkpoint"] = metadata
            record["real_transitions"] = runner.transition_count
            record["supervised_updates"] = runner.update_count
            manifest["updated_at"] = utc_now()
            atomic_json(output_root / "manifest.json", manifest)
        if runner.update_count != 7_250:
            raise RuntimeError("policy did not complete exactly 7,250 updates")
        record.update(
            {
                "state": "COMPLETED",
                "completed_at": utc_now(),
                "real_transitions": runner.transition_count,
                "supervised_updates": runner.update_count,
                "completed_training_episodes": len(runner.training_outcomes),
                "canonical_training_successes": sum(
                    row["canonical_success"] for row in runner.training_outcomes
                ),
                "final_checkpoint": record["last_verified_checkpoint"],
            }
        )
        attempt.update({"state": "COMPLETED", "completed_at": utc_now()})
        atomic_json(run_directory / "training-outcomes.json", runner.outcomes_summary())
        atomic_json(
            run_directory / "gcsl-diagnostics.json",
            {"summary": runner.diagnostics.summary(), "rows": runner.diagnostics.rows},
        )
        print(f"G{seed} | policy finished | HEALTHY", flush=True)
    except BaseException as error:
        metadata = runner.save(run_directory / "interruption" / "checkpoint.pkl")
        record.update(
            {
                "state": "INTERRUPTED",
                "interrupted_at": utc_now(),
                "last_verified_checkpoint": metadata,
                "real_transitions": runner.transition_count,
                "supervised_updates": runner.update_count,
                "interruption": f"{type(error).__name__}: {error}",
            }
        )
        attempt.update({"state": "INTERRUPTED", "interrupted_at": utc_now()})
        raise
    finally:
        runner.close()
        manifest["updated_at"] = utc_now()
        atomic_json(output_root / "manifest.json", manifest)
    return run_directory


def run_matrix(repository: Path, output_root: Path) -> Path:
    provenance = verify_preflight(repository, require_training_authorization=True)
    root = output_root if output_root.is_absolute() else repository / output_root
    if root.exists():
        launch_only = {
            "launchd.plist",
            "launcher.json",
            "launcher.stdout.log",
            "launcher.stderr.log",
        }
        unexpected = {path.name for path in root.iterdir()} - launch_only
        if unexpected:
            raise FileExistsError("fixed output root exists; duplicate run blocked")
    else:
        root.mkdir(parents=True)
    _, manifest = load_or_create_manifest(root, provenance)
    with RunLease(root):
        manifest["state"] = "RUNNING"
        atomic_json(root / "manifest.json", manifest)
        try:
            for seed in TRAINING_SEEDS:
                run_policy(
                    seed=seed,
                    output_root=root,
                    provenance=provenance,
                    manifest=manifest,
                )
            manifest["state"] = "COMPLETED"
            manifest["completed_at"] = utc_now()
        except BaseException as error:
            manifest["state"] = "INTERRUPTED"
            manifest["interruption"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            manifest["updated_at"] = utc_now()
            atomic_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def resume_matrix(repository: Path, output_root: Path) -> Path:
    provenance = verify_preflight(repository, require_training_authorization=True)
    root = output_root if output_root.is_absolute() else repository / output_root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["provenance"] != provenance:
        raise RuntimeError("resume provenance changed")
    interrupted = [
        row for row in manifest["runs"].values() if row["state"] == "INTERRUPTED"
    ]
    if manifest["state"] != "INTERRUPTED" or len(interrupted) != 1:
        raise RuntimeError("resume requires exactly one interrupted policy")
    with RunLease(root):
        manifest["state"] = "RUNNING"
        target = interrupted[0]
        run_policy(
            seed=int(target["training_seed"]),
            output_root=root,
            provenance=provenance,
            manifest=manifest,
            resume_checkpoint=Path(target["last_verified_checkpoint"]["path"]),
        )
        for seed in TRAINING_SEEDS:
            if manifest["runs"][f"G{seed}"]["state"] == "NOT_STARTED":
                run_policy(
                    seed=seed,
                    output_root=root,
                    provenance=provenance,
                    manifest=manifest,
                )
        manifest["state"] = "COMPLETED"
        manifest["completed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def prepare_launchd_job(repository: Path, output_root: Path, python: str) -> Path:
    root = output_root if output_root.is_absolute() else repository / output_root
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "launchd.plist"
    payload = {
        "Label": "org.longicontrol.gcsl-canonical-v1",
        "ProgramArguments": [
            python,
            "-m",
            "benchmarks.gcsl_canonical.cli",
            "matrix",
            "--output-root",
            str(root),
        ],
        "WorkingDirectory": str(repository),
        "EnvironmentVariables": {
            "PYTHONPATH": str(repository),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "StandardOutPath": str(root / "launcher.stdout.log"),
        "StandardErrorPath": str(root / "launcher.stderr.log"),
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Background",
        "AbandonProcessGroup": False,
    }
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_bytes(plistlib.dumps(payload, sort_keys=True))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_json(
        root / "launcher.json",
        {
            "state": "PREPARED_NOT_LAUNCHED",
            "main_training_authorized": False,
            "fresh_restart_authorized": False,
            "same_attempt_exact_resume_authorized": True,
        },
    )
    return destination


def read_status(output_root: Path) -> str:
    heartbeats = sorted(output_root.glob("G*/heartbeat.json"))
    if heartbeats:
        newest = max(heartbeats, key=lambda path: path.stat().st_mtime_ns)
        return status_progress(json.loads(newest.read_text(encoding="utf-8")))
    manifest = output_root / "manifest.json"
    if manifest.exists():
        return json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2)
    return "canonical GCSL study has not started"


def environment_report() -> dict[str, Any]:
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "runtime": runtime_versions(),
        "generated_at": utc_now(),
    }
