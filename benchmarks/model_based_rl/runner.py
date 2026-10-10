"""Command-line control plane for the frozen MBRL study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_configuration
from .execution import (
    DEFAULT_OUTPUT_ROOT,
    assert_freeze_ancestor,
    assert_paper_test_blocked,
    initial_manifest,
    load_or_create_manifest,
    scientific_hashes,
    validation_ready,
)
from .training import run_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "status",
            "model-disabled-parity",
            "run",
            "restart",
            "resume",
            "validate",
            "paper",
        ),
    )
    parser.add_argument("--condition", choices=("learned", "physics"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    assert_freeze_ancestor()
    configuration = load_configuration()
    if args.command == "preflight":
        print(
            json.dumps(
                {
                    "configuration_sha256": configuration.configuration_sha256,
                    "matrix": [
                        [condition, seed]
                        for condition in configuration.raw["conditions"]
                        for seed in configuration.raw["training_seeds"]
                    ],
                    "authorization": configuration.raw["authorization"],
                    "scientific_source_sha256": scientific_hashes(),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "model-disabled-parity":
        from .model_disabled_parity import run_check

        print(json.dumps(run_check(), indent=2, sort_keys=True))
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
    manifest_path, manifest = load_or_create_manifest(args.output_root)
    if args.command == "validate":
        if not validation_ready(manifest):
            raise RuntimeError("Validation remains sealed until all six runs complete")
        raise PermissionError("Use the separately authorized one-shot evaluator")
    if args.condition is None or args.seed is None:
        raise ValueError("run/resume requires --condition and --seed")
    if args.command == "resume" and args.checkpoint is None:
        raise ValueError("resume requires --checkpoint")
    if args.command != "resume" and args.checkpoint is not None:
        raise ValueError("--checkpoint is accepted only for exact resume")
    print(
        run_policy(
            condition=args.condition,
            training_seed=args.seed,
            output_root=args.output_root,
            device=args.device,
            threads=args.threads,
            resume=args.command == "resume",
            restart=args.command == "restart",
            checkpoint_path=args.checkpoint,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
