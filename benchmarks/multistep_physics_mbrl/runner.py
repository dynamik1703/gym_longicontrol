"""Control plane for the frozen multi-step physics-MVE study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import CONDITIONS, TRAINING_SEEDS, load_configuration
from .evaluation import evaluate_paper_test
from .execution import (
    DEFAULT_OUTPUT_ROOT,
    assert_source_ancestor,
    initial_manifest,
    prepare_launchd_job,
    run_policy,
    runtime_versions,
    source_hashes,
)


def preflight_payload(repository: str | Path = ".") -> dict:
    configuration = load_configuration()
    status_path = (
        Path(repository)
        / "benchmarks"
        / "multistep_physics_mbrl"
        / "preparation_status.json"
    )
    preparation = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.exists()
        else {"ready_for_main_training": False, "reason": "preparation incomplete"}
    )
    return {
        "study": configuration.raw["name"],
        "configuration_sha256": configuration.configuration_sha256,
        "source_commit": configuration.raw["source_commit"],
        "matrix": [
            [condition, seed] for condition in CONDITIONS for seed in TRAINING_SEEDS
        ],
        "real_transitions": configuration.raw["total_real_transitions"],
        "rl_updates_per_policy": configuration.raw["rl_updates_per_policy"],
        "target": configuration.raw["target"],
        "authorization": configuration.raw["authorization"],
        "preparation": preparation,
        "runtime": runtime_versions(),
        "source_sha256": source_hashes(repository),
    }


def run_matrix(*, output_root: Path, device: str, threads: int) -> int:
    configuration = load_configuration()
    authorization = configuration.raw["authorization"]
    if not (
        authorization["main_training_authorized"]
        and authorization["main_training_enabled"]
    ):
        raise PermissionError("The six-policy matrix is prepared but not authorized")
    for condition in CONDITIONS:
        for seed in TRAINING_SEEDS:
            run_policy(
                condition=condition,
                training_seed=seed,
                output_root=output_root,
                device=device,
                threads=threads,
            )
    return 0


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
            "validate",
            "paper",
        ),
    )
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS)
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
        print(json.dumps(preflight_payload(), indent=2, sort_keys=True))
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
    if args.command == "paper":
        evaluate_paper_test()
    if args.command == "validate":
        raise PermissionError("Validation remains sealed during preparation")
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
        return run_matrix(
            output_root=args.output_root, device=args.device, threads=args.threads
        )
    if args.condition is None or args.seed is None:
        raise ValueError("run/resume requires --condition and --seed")
    if args.command == "resume" and args.checkpoint is None:
        raise ValueError("resume requires --checkpoint")
    if args.command == "run" and args.checkpoint is not None:
        raise ValueError("--checkpoint is accepted only for exact resume")
    print(
        run_policy(
            condition=args.condition,
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
