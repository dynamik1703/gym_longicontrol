"""Explicit, evidence-preserving recovery operations for failed MBRL workers."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

from .checkpointing import file_sha256
from .config import load_configuration
from .execution import (
    DEFAULT_OUTPUT_ROOT,
    atomic_json,
    load_or_create_manifest,
    utc_now,
)
from .recovery import verify_recovery_checkpoint

ATTEMPT_TWO_ID = "8d35a1ba-853b-444e-bf59-92aa2eb0c00a"
ATTEMPT_TWO_PID = 94949


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _other_runner_pids() -> list[int]:
    completed = subprocess.run(
        ["pgrep", "-f", "benchmarks.model_based_rl.runner"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError("Could not inspect MBRL worker processes")
    return [
        int(value)
        for value in completed.stdout.split()
        if int(value) != os.getpid()
    ]


def preserve_attempt_two_and_authorize_fresh_restart(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    root = Path(output_root)
    manifest_path, manifest = load_or_create_manifest(root)
    run = manifest["runs"]["learned-seed-11"]
    if not run["attempts"] or run["attempts"][-1]["attempt_id"] != ATTEMPT_TWO_ID:
        raise RuntimeError("Attempt 2 identity differs from preserved evidence")
    attempt = run["attempts"][-1]
    lock_path = root / "learned-seed-11" / ".run.lock"
    if not lock_path.is_file():
        raise RuntimeError("Attempt 2 stale lock is missing")
    lock = _lock_payload(lock_path)
    if int(lock["pid"]) != ATTEMPT_TWO_PID:
        raise RuntimeError("Attempt 2 lock PID differs from preserved evidence")
    if process_alive(ATTEMPT_TWO_PID):
        raise RuntimeError("Attempt 2 worker PID is still alive")
    other_pids = _other_runner_pids()
    if other_pids:
        raise RuntimeError(f"Alternate MBRL workers are active: {other_pids}")
    if tuple(root.rglob("checkpoint.pt")):
        raise RuntimeError("Attempt 2 unexpectedly has a checkpoint")
    snapshot = {
        "schema_version": 1,
        "captured_at_utc": utc_now(),
        "run_id": "learned-seed-11",
        "attempt_id": ATTEMPT_TWO_ID,
        "lock_path": str(lock_path),
        "lock_payload": lock,
        "lock_sha256": file_sha256(lock_path),
        "worker_pid_alive": False,
        "alternate_worker_pids": [],
        "host": socket.gethostname(),
    }
    snapshot_path = root / "forensics" / "attempt-2-stale-lock.json"
    atomic_json(snapshot_path, snapshot)
    attempt.update(
        {
            "state": "HARD_FAILURE_UNRECOVERABLE",
            "failure_detected_at": utc_now(),
            "classification": "HARD_FAILURE_UNRECOVERABLE",
            "recovery_class": (
                "CAUSE_REQUIRES_INFRASTRUCTURE_REPAIR_BEFORE_RESTART"
            ),
            "last_durable_scientific_checkpoint": None,
            "durable_scientific_work": {
                "real_transitions": 0,
                "rl_gradient_updates": 0,
                "model_gradient_updates": 0,
                "model_refreshes": 0,
                "synthetic_transitions": 0,
            },
            "actual_consumed_scientific_work": {
                "real_transitions": None,
                "rl_gradient_updates": None,
                "model_gradient_updates": None,
                "model_refreshes": None,
                "synthetic_transitions": None,
                "status": "UNKNOWN",
            },
            "known_resource_lower_bounds": {
                "wall_clock_hours": 12.5,
                "observed_cumulative_cpu_time": "237:11 min",
                "last_observed_rss_mib": 324,
            },
            "forensic_snapshot": str(snapshot_path),
        }
    )
    run["state"] = "INTERRUPTED"
    run["failure_classification"] = "HARD_FAILURE_UNRECOVERABLE"
    run["durable_transition_count"] = 0
    run["actual_consumed_transition_count"] = None
    authorization = {
        "authorization_id": "attempt-3-user-authorization-2026-10-10",
        "authorized_at": "2026-10-10",
        "authorized_by": "user",
        "scope": "fresh learned-seed-11 Attempt 3 from transition 0",
        "consumed_at": None,
    }
    run.setdefault("fresh_restart_authorizations", []).append(authorization)
    atomic_json(manifest_path, manifest)
    lock_path.unlink()
    snapshot["lock_removed_at_utc"] = utc_now()
    snapshot["manifest_sha256_after_classification"] = file_sha256(manifest_path)
    atomic_json(snapshot_path, snapshot)
    return snapshot_path


def classify_verified_resume(
    *, output_root: str | Path, run_id: str
) -> Path:
    root = Path(output_root)
    manifest_path, manifest = load_or_create_manifest(root)
    run = manifest["runs"][run_id]
    lock_path = root / run_id / ".run.lock"
    if not lock_path.is_file():
        raise RuntimeError("No stale worker lock exists")
    lock = _lock_payload(lock_path)
    if process_alive(int(lock["pid"])):
        raise RuntimeError("Worker is still alive")
    latest_path = root / run_id / "recovery" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = Path(latest["path"])
    configuration = load_configuration()
    payload, metadata = verify_recovery_checkpoint(
        checkpoint,
        configuration_sha256=configuration.configuration_sha256,
        expected_attempt_id=run["attempts"][-1]["attempt_id"],
    )
    attempt = run["attempts"][-1]
    attempt.update(
        {
            "state": "INTERRUPTED",
            "failure_detected_at": utc_now(),
            "recovery_class": "VERIFIED_CHECKPOINT_RESUMABLE",
            "resume_checkpoint": metadata,
        }
    )
    run["state"] = "INTERRUPTED"
    run.update(payload["counters"])
    atomic_json(manifest_path, manifest)
    lock_path.unlink()
    return checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("attempt-2", "resume-ready"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.command == "attempt-2":
        print(preserve_attempt_two_and_authorize_fresh_restart(args.output_root))
    else:
        if args.run_id is None:
            raise ValueError("resume-ready requires --run-id")
        print(
            classify_verified_resume(
                output_root=args.output_root, run_id=args.run_id
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
