"""Evidence-preserving exact-resume classification for Physics-only workers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from benchmarks.model_based_rl.execution import atomic_json, utc_now
from benchmarks.model_based_rl.recovery import verify_recovery_checkpoint

from .config import load_configuration
from .execution import DEFAULT_OUTPUT_ROOT, load_or_create_manifest


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _other_worker_pids() -> list[int]:
    completed = subprocess.run(
        ["pgrep", "-f", "benchmarks.physics_model_mbrl.runner"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError("Could not inspect Physics-only worker processes")
    return [
        int(value) for value in completed.stdout.split() if int(value) != os.getpid()
    ]


def classify_verified_resume(*, output_root: str | Path, run_id: str) -> Path:
    root = Path(output_root)
    manifest_path, manifest = load_or_create_manifest(root)
    if run_id not in manifest["runs"]:
        raise ValueError("Run is not part of the frozen Physics-only matrix")
    run = manifest["runs"][run_id]
    if run["state"] not in {"RUNNING", "INTERRUPTED"}:
        raise RuntimeError("Only a failed in-progress run can be classified")
    if not run["attempts"]:
        raise RuntimeError("Run has no attempt to resume")
    lock_path = root / run_id / ".run.lock"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if process_alive(int(lock["pid"])):
            raise RuntimeError("Worker is still alive")
    workers = _other_worker_pids()
    if workers:
        raise RuntimeError(f"Alternate Physics-only workers are active: {workers}")
    latest_path = root / run_id / "recovery" / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint = Path(latest["path"])
    parent = load_configuration().parent
    payload, metadata = verify_recovery_checkpoint(
        checkpoint,
        configuration_sha256=parent.configuration_sha256,
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
    if lock_path.exists():
        lock_path.unlink()
    return checkpoint


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("resume-ready",))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    print(
        classify_verified_resume(
            output_root=args.output_root,
            run_id=args.run_id,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
