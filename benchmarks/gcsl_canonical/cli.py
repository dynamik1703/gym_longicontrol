"""Command-line entry point; training and Validation fail closed by default."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .execution import (
    DEFAULT_OUTPUT_ROOT,
    prepare_launchd_job,
    read_status,
    resume_matrix,
    run_matrix,
    verify_preflight,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "status", "matrix", "resume", "prepare-launchd"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[2]
    output = (
        args.output_root
        if args.output_root.is_absolute()
        else repository / args.output_root
    )
    if args.command == "preflight":
        print(
            json.dumps(
                verify_preflight(
                    repository,
                    require_clean=False,
                    require_training_authorization=False,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "status":
        print(read_status(output))
    elif args.command == "prepare-launchd":
        print(prepare_launchd_job(repository, output, args.python))
    elif args.command == "matrix":
        print(run_matrix(repository, output))
    else:
        print(resume_matrix(repository, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
