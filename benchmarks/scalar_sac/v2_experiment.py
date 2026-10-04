"""Run the pre-registered Scalar Reward V2 budget study."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from training.checkpoint import save_checkpoint
from training.cli import _build_agent
from training.sac import InitPolicy

from .evaluation import evaluate_policy
from .experiment import _agent_arguments, _base_environment, _policy_adapter
from .v2_config import (
    DEFAULT_V2_CONFIG_PATH,
    ScalarSACV2Configuration,
    V2RewardParameters,
    load_v2_configuration,
    v2_configuration_sha256,
)
from .v2_evaluation import V2BenchmarkRunResult, save_v2_result
from .v2_reward import ScalarBenchmarkRewardV2


def _training_environment(
    configuration: ScalarSACV2Configuration,
    parameters: V2RewardParameters,
):
    return ScalarBenchmarkRewardV2(
        _base_environment(configuration),
        parameters=parameters,
        max_time_s=configuration.task.max_time_s,
        max_speed_violation_m_s=configuration.task.max_speed_violation_m_s,
        energy_normalization_kwh=configuration.energy_normalization_kwh,
        speed_violation_normalization_m=(
            configuration.speed_violation_normalization_m
        ),
    )


def _evaluate_and_save(
    agent,
    environment,
    configuration,
    parameters,
    training_seed,
    training_steps,
    split_id,
    evaluation_seeds,
    output_path,
):
    episodes, summary = evaluate_policy(
        _policy_adapter(agent),
        environment,
        task=configuration.task,
        evaluation_seeds=evaluation_seeds,
    )
    result = V2BenchmarkRunResult(
        benchmark_name=configuration.name,
        configuration_sha256=v2_configuration_sha256(configuration),
        environment_id=configuration.environment_id,
        evaluation_split_id=split_id,
        training_seed=training_seed,
        training_steps=training_steps,
        task=configuration.task,
        reward_parameters=parameters,
        energy_normalization_kwh=configuration.energy_normalization_kwh,
        speed_violation_normalization_m=(
            configuration.speed_violation_normalization_m
        ),
        episodes=episodes,
        summary=summary,
    )
    return save_v2_result(output_path, result)


def run_v2_configuration(
    configuration: ScalarSACV2Configuration,
    parameters: V2RewardParameters,
    *,
    training_seed: int,
    output_directory: str | Path,
    device: str = "auto",
    overwrite: bool = False,
) -> Path:
    run_directory = (
        Path(output_directory)
        / parameters.configuration_id
        / f"training-seed-{training_seed}"
    )
    completion_marker = (
        run_directory
        / f"step-{configuration.training.total_training_steps:06d}"
        / "exploratory-result.json"
    )
    if completion_marker.exists() and not overwrite:
        raise FileExistsError(f"V2 run already exists: {completion_marker}")
    run_directory.mkdir(parents=True, exist_ok=True)

    training_environment = _training_environment(configuration, parameters)
    validation_environment = _base_environment(configuration)
    exploratory_environment = _base_environment(configuration)
    try:
        arguments = _agent_arguments(
            configuration, training_seed=training_seed, device=device
        )
        agent = _build_agent(
            arguments, training_environment, validation_environment
        )
        initial_policy = InitPolicy.from_action_space(
            training_environment.action_space, seed=training_seed
        )
        agent.init_replay_buffer(
            initial_policy,
            configuration.training.initial_random_steps,
            seed=training_seed + 1,
        )
        (run_directory / "run-config.json").write_text(
            json.dumps(
                {
                    "benchmark": asdict(configuration),
                    "configuration_sha256": v2_configuration_sha256(
                        configuration
                    ),
                    "reward_parameters": asdict(parameters),
                    "training_seed": training_seed,
                    "random_warmup_steps": (
                        configuration.training.initial_random_steps
                    ),
                    "milestone_training_is_continuous": True,
                    "device": str(agent.device),
                    "runtime": {
                        "torch_num_threads": __import__("torch").get_num_threads(),
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        previous_step = 0
        for milestone_index, training_steps in enumerate(
            configuration.learning_curve_steps
        ):
            agent.do_training(
                training_steps - previous_step,
                seed=training_seed + 2 + milestone_index,
            )
            milestone_directory = run_directory / f"step-{training_steps:06d}"
            _evaluate_and_save(
                agent,
                validation_environment,
                configuration,
                parameters,
                training_seed,
                training_steps,
                "validation-3000-3008-v1",
                configuration.track_splits.validation,
                milestone_directory / "validation-result.json",
            )
            if training_steps in configuration.comparison_steps:
                _evaluate_and_save(
                    agent,
                    exploratory_environment,
                    configuration,
                    parameters,
                    training_seed,
                    training_steps,
                    "v1-exploratory-1000-1008-v1",
                    configuration.track_splits.v1_exploratory_evaluation,
                    milestone_directory / "exploratory-result.json",
                )
                save_checkpoint(
                    milestone_directory / "checkpoint.tar",
                    agent,
                    epoch=training_steps,
                    history={"training_steps": [training_steps]},
                    metadata={
                        "benchmark_name": configuration.name,
                        "configuration_sha256": v2_configuration_sha256(
                            configuration
                        ),
                        "reward_parameters": asdict(parameters),
                        "training_seed": training_seed,
                        "training_steps": training_steps,
                    },
                )
            previous_step = training_steps
    finally:
        training_environment.close()
        validation_environment.close()
        exploratory_environment.close()
    return completion_marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_V2_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/scalar-sac-v2"))
    parser.add_argument("--reward-id", action="append")
    parser.add_argument("--training-seed", action="append", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one 64-step policy on two validation and exploratory tracks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    configuration = load_v2_configuration(args.config)
    if args.smoke:
        configuration = replace(
            configuration,
            track_splits=replace(
                configuration.track_splits,
                validation=configuration.track_splits.validation[:2],
                v1_exploratory_evaluation=(
                    configuration.track_splits.v1_exploratory_evaluation[:2]
                ),
            ),
            learning_curve_steps=(64,),
            comparison_steps=(64,),
            training=replace(
                configuration.training,
                total_training_steps=64,
                initial_random_steps=256,
                replay_buffer_capacity=512,
            ),
        )
    candidates = configuration.reward_candidates
    if args.reward_id:
        selected = set(args.reward_id)
        candidates = tuple(
            item for item in candidates if item.configuration_id in selected
        )
        missing = selected - {item.configuration_id for item in candidates}
        if missing:
            raise ValueError(f"Unknown V2 reward configuration(s): {sorted(missing)}")
    training_seeds = tuple(args.training_seed or configuration.training_seeds)
    if args.smoke:
        candidates = candidates[:1]
        training_seeds = training_seeds[:1]
    jobs = [
        (parameters, training_seed)
        for parameters in candidates
        for training_seed in training_seeds
    ]
    keyword_arguments = {
        "output_directory": args.output_dir,
        "device": args.device,
        "overwrite": args.overwrite,
    }
    if args.workers == 1:
        for parameters, training_seed in jobs:
            print(
                run_v2_configuration(
                    configuration,
                    parameters,
                    training_seed=training_seed,
                    **keyword_arguments,
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_v2_configuration,
                    configuration,
                    parameters,
                    training_seed=training_seed,
                    **keyword_arguments,
                ): (parameters.configuration_id, training_seed)
                for parameters, training_seed in jobs
            }
            for future in as_completed(futures):
                identifier, training_seed = futures[future]
                print(
                    f"completed {identifier}/seed-{training_seed}: "
                    f"{future.result()}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
