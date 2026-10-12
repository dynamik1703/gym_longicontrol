"""Detached serial worker for the frozen three-policy Physics-only matrix."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.model_based_rl.execution import atomic_json, utc_now

from .execution import DEFAULT_OUTPUT_ROOT

FROZEN_PLAN = (("physics", 11), ("physics", 29), ("physics", 47))


def _write_status(output_root: Path, status: dict[str, Any]) -> None:
    status["updated_at_utc"] = utc_now()
    atomic_json(output_root / "launcher-status.json", status)


def run_matrix(*, repository: str | Path, output_root: str | Path, python: str) -> int:
    root = Path(repository).resolve()
    runs = Path(output_root)
    if not runs.is_absolute():
        runs = root / runs
    runs.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "schema_version": 1,
        "launcher": "launchd",
        "launcher_pid": os.getpid(),
        "launcher_parent_pid": os.getppid(),
        "process_group_id": os.getpgrp(),
        "session_id": os.getsid(0),
        "host": socket.gethostname(),
        "python": python,
        "started_at_utc": utc_now(),
        "state": "RUNNING",
        "plan": [list(item) for item in FROZEN_PLAN],
        "completed": [],
        "active": None,
    }
    _write_status(runs, status)
    active: subprocess.Popen | None = None

    def handle_signal(number, _frame):
        status["state"] = "TERMINATED_BY_SIGNAL"
        status["terminating_signal"] = int(number)
        if active is not None and active.poll() is None:
            active.terminate()
        _write_status(runs, status)
        raise SystemExit(128 + int(number))

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    for condition, seed in FROZEN_PLAN:
        run_id = f"{condition}-seed-{seed}"
        log_path = runs / run_id / "worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            python,
            "-m",
            "benchmarks.physics_model_mbrl.runner",
            "run",
            "--seed",
            str(seed),
            "--output-root",
            str(runs),
            "--device",
            "cpu",
            "--threads",
            "4",
        ]
        started_at = utc_now()
        with log_path.open("ab", buffering=0) as log:
            active = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
            status["active"] = {
                "condition": condition,
                "seed": seed,
                "run_id": run_id,
                "worker_pid": active.pid,
                "command": command,
                "log_path": str(log_path),
                "started_at_utc": started_at,
            }
            _write_status(runs, status)
            returncode = active.wait()
        result = {
            **status["active"],
            "ended_at_utc": utc_now(),
            "exit_code": returncode if returncode >= 0 else None,
            "terminating_signal": -returncode if returncode < 0 else None,
        }
        status["active"] = None
        if returncode != 0:
            status["state"] = "STOPPED_ON_WORKER_FAILURE"
            status["failure"] = result
            _write_status(runs, status)
            return returncode if returncode > 0 else 128 - returncode
        status["completed"].append(result)
        _write_status(runs, status)
    status["state"] = "COMPLETED"
    status["ended_at_utc"] = utc_now()
    _write_status(runs, status)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    return run_matrix(
        repository=args.repository, output_root=args.output_root, python=args.python
    )


if __name__ == "__main__":
    raise SystemExit(main())
