"""Create and control a one-shot macOS launchd job for the frozen MBRL matrix."""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from .execution import DEFAULT_OUTPUT_ROOT, atomic_json, utc_now

LABEL = "org.longicontrol.mbrl.frozen-v1"


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_launchd_job(
    *,
    repository: str | Path,
    output_root: str | Path,
    python: str,
    pythonpath: str,
) -> tuple[Path, dict]:
    root = Path(repository).resolve()
    runs = Path(output_root)
    if not runs.is_absolute():
        runs = root / runs
    runs.mkdir(parents=True, exist_ok=True)
    plist_path = runs / "launchd.plist"
    stdout_path = runs / "launcher.stdout.log"
    stderr_path = runs / "launcher.stderr.log"
    payload = {
        "Label": LABEL,
        "ProgramArguments": [
            python,
            "-m",
            "benchmarks.model_based_rl.matrix_worker",
            "--repository",
            str(root),
            "--output-root",
            str(runs),
            "--python",
            python,
        ],
        "WorkingDirectory": str(root),
        "EnvironmentVariables": {
            "PYTHONPATH": pythonpath,
            "PYTHONUNBUFFERED": "1",
        },
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Background",
        "AbandonProcessGroup": False,
    }
    _atomic_bytes(plist_path, plistlib.dumps(payload, sort_keys=True))
    metadata = {
        "schema_version": 1,
        "label": LABEL,
        "created_at_utc": utc_now(),
        "plist_path": str(plist_path),
        "repository": str(root),
        "output_root": str(runs),
        "python": python,
        "pythonpath": pythonpath,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "state": "PREPARED",
    }
    atomic_json(runs / "launcher.json", metadata)
    return plist_path, metadata


def bootstrap(plist_path: str | Path) -> None:
    target = f"gui/{os.getuid()}"
    subprocess.run(
        ["launchctl", "bootstrap", target, str(Path(plist_path).resolve())],
        check=True,
    )


def launchd_status() -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", "print", f"gui/{os.getuid()}/{LABEL}"],
        check=False,
        capture_output=True,
        text=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "launch", "status"))
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pythonpath", default=os.environ.get("PYTHONPATH", ""))
    args = parser.parse_args(argv)
    if args.command == "status":
        completed = launchd_status()
        print(completed.stdout or completed.stderr)
        return completed.returncode
    plist, metadata = create_launchd_job(
        repository=args.repository,
        output_root=args.output_root,
        python=args.python,
        pythonpath=args.pythonpath,
    )
    if args.command == "launch":
        bootstrap(plist)
        metadata["state"] = "BOOTSTRAPPED"
        metadata["bootstrapped_at_utc"] = utc_now()
        root = Path(metadata["output_root"])
        atomic_json(root / "launcher.json", metadata)
    print(plist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
