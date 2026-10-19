"""Command-line control plane for the frozen V3 factorial."""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import sys
from pathlib import Path

from .config import CONDITION_NAMES
from .execution import (
    DEFAULT_OUTPUT_ROOT,
    assert_source_ancestor,
    atomic_json,
    initial_manifest,
    preparation_status,
    run_matrix,
    run_policy,
)


def prepare_launchd_job(
    *, repository: Path, output_root: Path, python: str
) -> Path:
    root = repository.resolve()
    runs = output_root if output_root.is_absolute() else root / output_root
    runs.mkdir(parents=True, exist_ok=True)
    destination = runs / "launchd.plist"
    payload = {
        "Label": "org.longicontrol.constrained-v2-seed11-factorial-v3",
        "ProgramArguments": [
            python,
            "-m",
            "benchmarks.constrained_v2_seed11_factorial_v3.runner",
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
    try:
        with temporary.open("wb") as stream:
            stream.write(plistlib.dumps(payload, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_json(
        runs / "launcher.json",
        {
            "schema_version": 1,
            "state": "PREPARED_NOT_LAUNCHED",
            "plist_path": str(destination),
            "condition_order": list(CONDITION_NAMES),
            "fresh_restart_authorized": False,
            "same_attempt_exact_resume_authorized": True,
        },
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "status",
            "run",
            "resume",
            "matrix",
            "prepare-launchd",
        ),
    )
    parser.add_argument("--condition", choices=CONDITION_NAMES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    assert_source_ancestor()
    if args.command == "preflight":
        print(json.dumps(preparation_status(), indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        path = args.output_root / "manifest.json"
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else initial_manifest()
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-launchd":
        print(
            prepare_launchd_job(
                repository=Path.cwd(),
                output_root=args.output_root,
                python=args.python,
            )
        )
        return 0
    if args.command == "matrix":
        run_matrix(
            output_root=args.output_root, device=args.device, threads=args.threads
        )
        return 0
    if args.condition is None:
        raise ValueError("run/resume requires --condition")
    if args.command == "resume" and args.checkpoint is None:
        raise ValueError("resume requires --checkpoint")
    if args.command == "run" and args.checkpoint is not None:
        raise ValueError("--checkpoint is accepted only for exact resume")
    checkpoint = run_policy(
        condition_name=args.condition,
        output_root=args.output_root,
        device=args.device,
        threads=args.threads,
        resume=args.command == "resume",
        checkpoint_path=args.checkpoint,
    )
    print(checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
