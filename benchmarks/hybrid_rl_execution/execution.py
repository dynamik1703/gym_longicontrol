"""Hardened, authorization-gated Hybrid RL execution control plane."""

from __future__ import annotations

import hashlib
import json
import os
import plistlib
import socket
import subprocess
import sys
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpointing import file_sha256, runtime_versions
from .evaluation import evaluate_diagnostic
from .runner import GRADIENT_UPDATE_BUDGET, NATIVE_BUDGET, TRAINING_SEEDS, HybridRunner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "canonical.json"
STATUS = HERE / "preparation_status.json"
DEFAULT_OUTPUT_ROOT = Path("runs/hybrid-rl-execution-v1")
EXPECTED_BRANCH = "research/hybrid-rl-execution"
BASE_COMMIT = "337fe498083eb3d4c63263f696ee8b30f6800fec"
HEARTBEAT_INTERVAL = 250
RECOVERY_INTERVAL = 10_000
SCIENTIFIC_CHECKPOINTS = frozenset(
    (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
)
HASHED_SOURCE_FILES = (
    "canonical.json",
    "agent.py",
    "macro_transition.py",
    "replay.py",
    "runner.py",
    "evaluation.py",
    "execution.py",
    "checkpointing.py",
    "safety.py",
    "resource.py",
    "verification.py",
    "prepare.py",
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


def _git(*arguments: str, repository: str | Path = ROOT) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def configuration_sha256() -> str:
    return hashlib.sha256(CANONICAL.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        f"benchmarks/hybrid_rl_execution/{name}": file_sha256(HERE / name)
        for name in HASHED_SOURCE_FILES
    }


def _authorization() -> dict[str, Any]:
    return json.loads(CANONICAL.read_text(encoding="utf-8"))["authorization"]


def require_main_training_authorized() -> None:
    authorization = _authorization()
    required = (
        authorization["ready_for_main_training"],
        authorization["main_training_authorized"],
        authorization["main_training_enabled"],
    )
    if not all(required):
        raise PermissionError("main Hybrid RL training is not authorized or enabled")


def verify_preflight(
    repository: str | Path,
    *,
    require_clean: bool = True,
    require_authorization: bool = True,
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
        raise RuntimeError("working-tree changes block canonical Hybrid execution")
    if require_authorization:
        require_main_training_authorized()
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if not status.get("ready_for_main_training"):
        raise RuntimeError("Hybrid preparation gates are incomplete")
    hashes = source_hashes()
    if status.get("scientific_source_sha256") != hashes:
        raise RuntimeError("frozen Hybrid execution source hashes changed")
    if status.get("configuration_sha256") != configuration_sha256():
        raise RuntimeError("frozen Hybrid configuration hash changed")
    return {
        "branch": EXPECTED_BRANCH,
        "base_commit": BASE_COMMIT,
        "execution_commit": _git("rev-parse", "HEAD", repository=root),
        "configuration_sha256": configuration_sha256(),
        "source_hashes": hashes,
        "runtime_versions": runtime_versions(),
    }


def initial_manifest(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "HYBRID_RL_EXECUTION_V1",
        "state": "NOT_STARTED",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "provenance": provenance,
        "execution_order": list(TRAINING_SEEDS),
        "runs": {
            str(seed): {
                "training_seed": seed,
                "state": "NOT_STARTED",
                "attempts": [],
                "native_transitions": 0,
                "macro_transitions": 0,
                "gradient_updates": 0,
                "diagnostic_checkpoints_completed": [],
            }
            for seed in TRAINING_SEEDS
        },
        "validation": {"state": "FROZEN_UNOPENED", "episode_count": 0},
        "paper": {"state": "SEALED_UNINSPECTED", "episode_count": 0},
    }


class RunLease(AbstractContextManager):
    def __init__(self, output_root: Path) -> None:
        self.path = output_root / "ACTIVE.lock"
        self.acquired = False

    def __enter__(self) -> RunLease:
        payload = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": utc_now(),
        }
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            raise RuntimeError("duplicate Hybrid execution process blocked") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self.acquired and self.path.exists():
            self.path.unlink()
        return False


def _heartbeat(runner: HybridRunner, run_directory: Path) -> dict[str, Any]:
    payload = {
        "updated_at": utc_now(),
        "training_seed": runner.training_seed,
        "attempt_id": runner.attempt_id,
        "native_transitions": runner.native_transition_count,
        "macro_transitions": runner.macro_transition_count,
        "gradient_updates": runner.gradient_update_count,
        "episode_id": runner.episode_id,
        "remaining_native_steps_in_block": runner.remaining_native_steps_in_block,
        "proposal_count": runner.proposal_count,
        "override_fraction": (
            runner.override_count / runner.proposal_count
            if runner.proposal_count
            else 0.0
        ),
        "native_action_influence_fraction": (
            runner.influenced_native_count / runner.proposal_count
            if runner.proposal_count
            else 0.0
        ),
        "speed_multiplier": float(runner.agent.speed_multiplier.detach()),
        "terminal_multiplier": float(runner.agent.terminal_multiplier.detach()),
        "walltime_s": runner.walltime_s(),
        "state": "RUNNING",
    }
    atomic_json(run_directory / "heartbeat.json", payload)
    return payload


def _save_scientific_checkpoint(
    runner: HybridRunner, run_directory: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    transition = runner.native_transition_count
    final = run_directory / f"step-{transition:06d}"
    if final.exists():
        raise FileExistsError(f"scientific checkpoint exists: {final}")
    pending = run_directory / f".step-{transition:06d}.pending-{uuid.uuid4().hex}"
    pending.mkdir(parents=True)
    diagnostic = evaluate_diagnostic(
        runner.agent,
        training_context={
            "native_transition_count": runner.native_transition_count,
            "macro_transition_count": runner.macro_transition_count,
            "gradient_update_count": runner.gradient_update_count,
            "latest_update_diagnostics": runner.update_diagnostics[-10:],
            "intervention_reasons": runner.intervention_counts,
        },
    )
    try:
        metadata = runner.save(pending / "checkpoint.pkl")
        atomic_json(pending / "diagnostic.json", diagnostic)
        atomic_json(
            pending / "training-diagnostics.json",
            {
                "native_transitions": runner.native_transition_count,
                "macro_transitions": runner.macro_transition_count,
                "gradient_updates": runner.gradient_update_count,
                "latest_updates": runner.update_diagnostics[-10:],
                "intervention_reasons": runner.intervention_counts,
                "training_outcomes": runner.training_outcomes,
            },
        )
        pending.replace(final)
    except BaseException:
        raise
    metadata["path"] = str(final / "checkpoint.pkl")
    return metadata, diagnostic


def _save_recovery(runner: HybridRunner, run_directory: Path) -> dict[str, Any]:
    return runner.save(run_directory / "recovery" / "checkpoint.pkl")


def _load_or_create_manifest(
    output_root: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["provenance"] != provenance:
            raise RuntimeError("existing Hybrid manifest provenance changed")
        return manifest
    manifest = initial_manifest(provenance)
    atomic_json(path, manifest)
    return manifest


def run_policy(
    *,
    seed: int,
    output_root: Path,
    provenance: dict[str, Any],
    manifest: dict[str, Any],
    resume_checkpoint: Path | None = None,
) -> Path:
    record = manifest["runs"][str(seed)]
    run_directory = output_root / f"seed-{seed}"
    if resume_checkpoint is None:
        if record["state"] != "NOT_STARTED" or run_directory.exists():
            raise RuntimeError(
                f"duplicate or silent fresh restart blocked for seed {seed}"
            )
        attempt_id = str(uuid.uuid4())
        run_directory.mkdir(parents=True)
        runner = HybridRunner(
            training_seed=seed,
            attempt_id=attempt_id,
            configuration_sha256=provenance["configuration_sha256"],
            source_hashes=provenance["source_hashes"],
        )
        attempt = {
            "attempt_id": attempt_id,
            "state": "RUNNING",
            "started_at": utc_now(),
            "resume_events": [],
        }
        record["attempts"].append(attempt)
    else:
        if record["state"] != "INTERRUPTED":
            raise RuntimeError("same-Attempt resume requires INTERRUPTED state")
        attempt = record["attempts"][-1]
        attempt_id = attempt["attempt_id"]
        latest = record["last_verified_checkpoint"]
        if file_sha256(resume_checkpoint) != latest["sha256"]:
            raise RuntimeError("resume checkpoint differs from recorded Attempt")
        runner = HybridRunner.restore(
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
    record["state"] = "RUNNING"
    atomic_json(output_root / "manifest.json", manifest)
    try:
        while runner.native_transition_count < NATIVE_BUDGET:
            runner.advance_native()
            count = runner.native_transition_count
            if count % HEARTBEAT_INTERVAL == 0:
                _heartbeat(runner, run_directory)
            if count % RECOVERY_INTERVAL:
                continue
            if count in SCIENTIFIC_CHECKPOINTS:
                metadata, diagnostic = _save_scientific_checkpoint(
                    runner, run_directory
                )
                record["diagnostic_checkpoints_completed"].append(count)
                record.setdefault("scientific_checkpoints", []).append(
                    {**metadata, "diagnostic_rsr": diagnostic["rsr"]}
                )
            else:
                metadata = _save_recovery(runner, run_directory)
            record["last_verified_checkpoint"] = metadata
            record["native_transitions"] = count
            record["macro_transitions"] = runner.macro_transition_count
            record["gradient_updates"] = runner.gradient_update_count
            manifest["updated_at"] = utc_now()
            atomic_json(output_root / "manifest.json", manifest)
        if runner.gradient_update_count != GRADIENT_UPDATE_BUDGET:
            raise RuntimeError("policy did not consume exactly 30,000 updates")
        record.update(
            {
                "state": "COMPLETED",
                "completed_at": utc_now(),
                "native_transitions": runner.native_transition_count,
                "macro_transitions": runner.macro_transition_count,
                "gradient_updates": runner.gradient_update_count,
                "training_outcomes": runner.training_outcomes,
                "final_checkpoint": record["last_verified_checkpoint"],
            }
        )
        attempt["state"] = "COMPLETED"
        attempt["completed_at"] = utc_now()
    except BaseException as error:
        metadata = runner.save(run_directory / "interruption" / "checkpoint.pkl")
        record.update(
            {
                "state": "INTERRUPTED",
                "interrupted_at": utc_now(),
                "interruption": f"{type(error).__name__}: {error}",
                "last_verified_checkpoint": metadata,
                "native_transitions": runner.native_transition_count,
                "macro_transitions": runner.macro_transition_count,
                "gradient_updates": runner.gradient_update_count,
            }
        )
        attempt["state"] = "INTERRUPTED"
        attempt["interrupted_at"] = utc_now()
        raise
    finally:
        runner.close()
        manifest["updated_at"] = utc_now()
        atomic_json(output_root / "manifest.json", manifest)
    return run_directory


def run_matrix(repository: Path, output_root: Path) -> Path:
    provenance = verify_preflight(repository, require_authorization=True)
    root = output_root if output_root.is_absolute() else repository / output_root
    if root.exists():
        allowed = {
            "launchd.plist",
            "launcher.json",
            "launcher.stdout.log",
            "launcher.stderr.log",
        }
        unexpected = {path.name for path in root.iterdir()} - allowed
        if unexpected:
            raise FileExistsError("fixed output root exists; duplicate run blocked")
    else:
        root.mkdir(parents=True)
    manifest = _load_or_create_manifest(root, provenance)
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
    provenance = verify_preflight(repository, require_authorization=True)
    root = output_root if output_root.is_absolute() else repository / output_root
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["provenance"] != provenance:
        raise RuntimeError("resume manifest provenance changed")
    interrupted = [
        row for row in manifest["runs"].values() if row["state"] == "INTERRUPTED"
    ]
    if manifest["state"] != "INTERRUPTED" or len(interrupted) != 1:
        raise RuntimeError("resume requires exactly one interrupted Hybrid policy")
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
            if manifest["runs"][str(seed)]["state"] == "NOT_STARTED":
                run_policy(
                    seed=seed,
                    output_root=root,
                    provenance=provenance,
                    manifest=manifest,
                )
        manifest["state"] = "COMPLETED"
        manifest["completed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
    return manifest_path


def prepare_launchd_job(repository: Path, output_root: Path, python: str) -> Path:
    root = output_root if output_root.is_absolute() else repository / output_root
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "launchd.plist"
    payload = {
        "Label": "org.longicontrol.hybrid-rl-execution-v1",
        "ProgramArguments": [
            python,
            "-m",
            "benchmarks.hybrid_rl_execution.execution",
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
            "main_training_enabled": False,
            "fresh_restart_authorized": False,
            "same_attempt_exact_resume_authorized": True,
        },
    )
    return destination


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    launchd = subparsers.add_parser("prepare-launchd")
    launchd.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    launchd.add_argument("--python", default=sys.executable)
    arguments = parser.parse_args(argv)
    if arguments.command == "matrix":
        print(run_matrix(ROOT, arguments.output_root))
    elif arguments.command == "resume":
        print(resume_matrix(ROOT, arguments.output_root))
    else:
        print(prepare_launchd_job(ROOT, arguments.output_root, arguments.python))


if __name__ == "__main__":
    main()
