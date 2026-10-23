"""Exclusive six-run execution with atomic checkpoints and native evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import tempfile
import uuid
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpointing import file_sha256
from .integrity import ROOT, verify_execution_preconditions
from .training import (
    GRADIENT_UPDATE_BUDGET,
    METHODS,
    NATIVE_BUDGET,
    TRAINING_SEEDS,
    TrainingRunner,
)

DEFAULT_OUTPUT_ROOT = Path("runs/feedback-chunk-rl-v1")
HEARTBEAT_INTERVAL = 250
RECOVERY_INTERVAL = 10_000
SCIENTIFIC_CHECKPOINTS = frozenset(range(50_000, 300_001, 50_000))
DIAGNOSTIC_CHECKPOINTS = frozenset((100_000, 200_000, 300_000))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


class RunLease(AbstractContextManager):
    def __init__(self, output_root: Path) -> None:
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
            raise RuntimeError("duplicate Feedback Chunk execution blocked") from error
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


class NativeEvidenceBuffer:
    """Write integrity-checked gzip JSONL segments only at checkpoint boundaries."""

    def __init__(self, run_directory: Path, starting_count: int = 0) -> None:
        self.directory = run_directory / "native-evidence"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.buffer: list[dict[str, Any]] = []
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"schema_version": 1, "segments": [], "row_count": 0}
        if int(self.manifest["row_count"]) != int(starting_count):
            raise RuntimeError("native evidence and checkpoint counts differ")

    def append(self, row: dict[str, Any]) -> None:
        expected = int(self.manifest["row_count"]) + len(self.buffer) + 1
        if int(row["native_transition_count"]) != expected:
            raise RuntimeError("native evidence row is not contiguous")
        self.buffer.append(row)

    def flush(self) -> dict[str, Any] | None:
        if not self.buffer:
            return None
        start = int(self.buffer[0]["native_transition_count"])
        end = int(self.buffer[-1]["native_transition_count"])
        destination = self.directory / f"native-{start:06d}-{end:06d}.jsonl.gz"
        if destination.exists():
            raise FileExistsError("native evidence segment already exists")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=self.directory
        )
        temporary = Path(temporary_name)
        os.close(descriptor)
        try:
            with gzip.open(
                temporary, "wt", encoding="utf-8", compresslevel=6
            ) as stream:
                for row in self.buffer:
                    stream.write(
                        json.dumps(
                            row,
                            allow_nan=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    )
                    stream.write("\n")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        metadata = {
            "start_native_transition": start,
            "end_native_transition": end,
            "row_count": len(self.buffer),
            "path": str(destination),
            "sha256": file_sha256(destination),
            "size_bytes": destination.stat().st_size,
        }
        self.manifest["segments"].append(metadata)
        self.manifest["row_count"] = end
        atomic_json(self.manifest_path, self.manifest)
        self.buffer.clear()
        return metadata


def _run_key(method: str, seed: int) -> str:
    return f"{method}__seed-{seed}"


def initial_manifest(provenance: dict[str, Any]) -> dict[str, Any]:
    order = [_run_key(method, seed) for method in METHODS for seed in TRAINING_SEEDS]
    return {
        "schema_version": 1,
        "study": "FEEDBACK_CORRECTED_RESIDUAL_ACTION_CHUNKING_RL_V1",
        "state": "NOT_STARTED",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "provenance": provenance,
        "execution_order": order,
        "runs": {
            _run_key(method, seed): {
                "method": method,
                "training_seed": seed,
                "state": "NOT_STARTED",
                "attempts": [],
                "native_transitions": 0,
                "macro_transitions": 0,
                "gradient_updates": 0,
                "scientific_checkpoints_completed": [],
                "diagnostic_checkpoints_pending": sorted(DIAGNOSTIC_CHECKPOINTS),
            }
            for method in METHODS
            for seed in TRAINING_SEEDS
        },
        "diagnostic": {"state": "FROZEN_PENDING_SOURCE_FREEZE", "episodes": 0},
        "validation": {"state": "FROZEN_UNOPENED", "episodes": 0},
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "episodes": 0,
            "content_fingerprints_constructed": 0,
        },
    }


def _heartbeat(runner: TrainingRunner, run_directory: Path) -> None:
    active = runner.controller.active_chunk
    chunk_phase = None
    if active is not None:
        offset = runner.controller.native_step - active.start_native_step
        if 0 <= offset < active.duration_native_steps:
            chunk_phase = active.phase(runner.controller.native_step)
    atomic_json(
        run_directory / "heartbeat.json",
        {
            "updated_at": utc_now(),
            "method": runner.method,
            "training_seed": runner.training_seed,
            "attempt_id": runner.attempt_id,
            "native_transitions": runner.native_transition_count,
            "macro_transitions": runner.macro_transition_count,
            "gradient_updates": runner.gradient_update_count,
            "episode_id": runner.episode_id,
            "chunk_phase": chunk_phase,
            "proposal_count": runner.proposal_count,
            "residual_accept_fraction": (
                runner.residual_accept_count / runner.proposal_count
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
            "process_time_s": runner.process_time_s(),
            "state": "RUNNING",
        },
    )


def _save_scientific_checkpoint(
    runner: TrainingRunner, run_directory: Path
) -> dict[str, Any]:
    transition = runner.native_transition_count
    final = run_directory / f"step-{transition:06d}"
    if final.exists():
        raise FileExistsError(f"scientific checkpoint exists: {final}")
    pending = run_directory / f".step-{transition:06d}.pending-{uuid.uuid4().hex}"
    pending.mkdir(parents=True)
    try:
        metadata = runner.save(pending / "checkpoint.pkl")
        atomic_json(
            pending / "training-diagnostics.json",
            {
                "method": runner.method,
                "training_seed": runner.training_seed,
                "native_transitions": runner.native_transition_count,
                "macro_transitions": runner.macro_transition_count,
                "gradient_updates": runner.gradient_update_count,
                "latest_updates": runner.update_diagnostics[-10:],
                "cancellation_counts": runner.cancellation_counts,
                "training_outcomes": runner.training_outcomes,
                "chunk_outcomes": runner.chunk_outcomes,
                "diagnostic_evaluation_state": (
                    "PENDING_EXACTLY_ONCE_EVALUATION"
                    if transition in DIAGNOSTIC_CHECKPOINTS
                    else "NOT_SCHEDULED"
                ),
            },
        )
        pending.replace(final)
    except BaseException:
        raise
    metadata["path"] = str(final / "checkpoint.pkl")
    metadata["sha256"] = file_sha256(final / "checkpoint.pkl")
    return metadata


def _save_recovery(runner: TrainingRunner, run_directory: Path) -> dict[str, Any]:
    return runner.save(run_directory / "recovery" / "checkpoint.pkl")


def run_treatment(
    *,
    method: str,
    seed: int,
    output_root: Path,
    provenance: dict[str, Any],
    manifest: dict[str, Any],
    resume_checkpoint: Path | None = None,
) -> Path:
    key = _run_key(method, seed)
    record = manifest["runs"][key]
    run_directory = output_root / method.lower() / f"seed-{seed}"
    if resume_checkpoint is None:
        if record["state"] != "NOT_STARTED" or run_directory.exists():
            raise RuntimeError(f"duplicate or silent fresh restart blocked: {key}")
        attempt_id = str(uuid.uuid4())
        run_directory.mkdir(parents=True)
        runner = TrainingRunner(
            method=method,
            training_seed=seed,
            attempt_id=attempt_id,
            configuration_sha256=provenance["canonical_sha256"],
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
        runner = TrainingRunner.restore(
            resume_checkpoint,
            expected_sha256=latest["sha256"],
            configuration_sha256=provenance["canonical_sha256"],
            source_hashes=provenance["source_hashes"],
            attempt_id=attempt_id,
        )
        attempt["state"] = "RUNNING"
        attempt["resume_events"].append(
            {"resumed_at": utc_now(), "checkpoint_sha256": latest["sha256"]}
        )
    evidence = NativeEvidenceBuffer(
        run_directory, starting_count=runner.native_transition_count
    )
    record["state"] = "RUNNING"
    atomic_json(output_root / "manifest.json", manifest)
    try:
        while runner.native_transition_count < NATIVE_BUDGET:
            evidence.append(runner.advance_native())
            count = runner.native_transition_count
            if count % HEARTBEAT_INTERVAL == 0:
                _heartbeat(runner, run_directory)
            if count % RECOVERY_INTERVAL:
                continue
            segment = evidence.flush()
            if count in SCIENTIFIC_CHECKPOINTS:
                metadata = _save_scientific_checkpoint(runner, run_directory)
                record["scientific_checkpoints_completed"].append(count)
                record.setdefault("scientific_checkpoints", []).append(metadata)
            else:
                metadata = _save_recovery(runner, run_directory)
            record["last_verified_checkpoint"] = metadata
            record["last_evidence_segment"] = segment
            record["native_transitions"] = count
            record["macro_transitions"] = runner.macro_transition_count
            record["gradient_updates"] = runner.gradient_update_count
            record["walltime_s"] = runner.walltime_s()
            record["process_time_s"] = runner.process_time_s()
            manifest["updated_at"] = utc_now()
            atomic_json(output_root / "manifest.json", manifest)
        if runner.gradient_update_count != GRADIENT_UPDATE_BUDGET:
            raise RuntimeError("run did not consume exactly 30,000 updates")
        record.update(
            {
                "state": "COMPLETED",
                "completed_at": utc_now(),
                "native_transitions": runner.native_transition_count,
                "macro_transitions": runner.macro_transition_count,
                "gradient_updates": runner.gradient_update_count,
                "training_outcomes": runner.training_outcomes,
                "chunk_outcomes": runner.chunk_outcomes,
                "update_diagnostics": runner.update_diagnostics,
                "cancellation_counts": runner.cancellation_counts,
                "proposal_count": runner.proposal_count,
                "residual_attempt_count": runner.residual_attempt_count,
                "residual_accept_count": runner.residual_accept_count,
                "influenced_native_count": runner.influenced_native_count,
                "final_checkpoint": record["last_verified_checkpoint"],
                "walltime_s": runner.walltime_s(),
                "process_time_s": runner.process_time_s(),
            }
        )
        attempt["state"] = "COMPLETED"
        attempt["completed_at"] = utc_now()
    except BaseException as error:
        evidence.flush()
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


def _load_or_create_manifest(
    output_root: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["provenance"] != provenance:
            raise RuntimeError("existing execution manifest provenance changed")
        return manifest
    manifest = initial_manifest(provenance)
    atomic_json(path, manifest)
    return manifest


def run_matrix(repository: Path, output_root: Path) -> Path:
    if repository.resolve() != ROOT.resolve():
        raise RuntimeError("execution repository differs from study root")
    provenance = verify_execution_preconditions()
    root = output_root if output_root.is_absolute() else repository / output_root
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("fixed output root exists; duplicate run blocked")
    root.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(root, provenance)
    with RunLease(root):
        manifest["state"] = "RUNNING"
        atomic_json(root / "manifest.json", manifest)
        try:
            for method in METHODS:
                for seed in TRAINING_SEEDS:
                    run_treatment(
                        method=method,
                        seed=seed,
                        output_root=root,
                        provenance=provenance,
                        manifest=manifest,
                    )
            manifest["state"] = "TRAINING_COMPLETED_DIAGNOSTIC_PENDING"
            manifest["training_completed_at"] = utc_now()
        except BaseException as error:
            manifest["state"] = "INTERRUPTED"
            manifest["interruption"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            manifest["updated_at"] = utc_now()
            atomic_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def resume_matrix(repository: Path, output_root: Path) -> Path:
    provenance = verify_execution_preconditions()
    root = output_root if output_root.is_absolute() else repository / output_root
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["provenance"] != provenance:
        raise RuntimeError("resume manifest provenance changed")
    interrupted = [
        row for row in manifest["runs"].values() if row["state"] == "INTERRUPTED"
    ]
    if manifest["state"] != "INTERRUPTED" or len(interrupted) != 1:
        raise RuntimeError("resume requires exactly one interrupted run")
    with RunLease(root):
        manifest["state"] = "RUNNING"
        target = interrupted[0]
        run_treatment(
            method=target["method"],
            seed=int(target["training_seed"]),
            output_root=root,
            provenance=provenance,
            manifest=manifest,
            resume_checkpoint=Path(target["last_verified_checkpoint"]["path"]),
        )
        for key in manifest["execution_order"]:
            record = manifest["runs"][key]
            if record["state"] == "NOT_STARTED":
                run_treatment(
                    method=record["method"],
                    seed=int(record["training_seed"]),
                    output_root=root,
                    provenance=provenance,
                    manifest=manifest,
                )
        manifest["state"] = "TRAINING_COMPLETED_DIAGNOSTIC_PENDING"
        manifest["training_completed_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("matrix", "resume"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    if arguments.command == "matrix":
        print(run_matrix(ROOT, arguments.output_root))
    else:
        print(resume_matrix(ROOT, arguments.output_root))


if __name__ == "__main__":
    main()
