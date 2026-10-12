"""Command-line control plane for Physics-only constrained MBRL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .execution import (
    DEFAULT_OUTPUT_ROOT,
    assert_paper_test_blocked,
    assert_source_ancestor,
    initial_manifest,
    preflight_payload,
)
from .freeze import verify_study_freeze
from .training import run_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "status", "run", "resume", "validate", "paper")
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    assert_source_ancestor()
    verify_study_freeze()
    if args.command == "preflight":
        print(json.dumps(preflight_payload(), indent=2, sort_keys=True))
        return 0
    if args.command == "status":
        manifest_path = args.output_root / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else initial_manifest()
        )
        print(manifest_path)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.command == "paper":
        assert_paper_test_blocked()
    if args.command == "validate":
        from .final_evaluation import run_validation

        run_validation(args.output_root, device=args.device, threads=args.threads)
        return 0
    if args.seed is None:
        raise ValueError("run/resume requires --seed")
    if args.command == "resume" and args.checkpoint is None:
        raise ValueError("resume requires --checkpoint")
    if args.command != "resume" and args.checkpoint is not None:
        raise ValueError("--checkpoint is accepted only for exact resume")
    print(
        run_policy(
            training_seed=args.seed,
            output_root=args.output_root,
            device=args.device,
            threads=args.threads,
            resume=args.command == "resume",
            checkpoint_path=args.checkpoint,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
