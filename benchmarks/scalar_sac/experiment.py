"""Train the existing SAC implementation over the scalar reward grid."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from training.checkpoint import save_checkpoint
from training.cli import _build_agent, get_args
from training.sac import InitPolicy

from .config import (
    DEFAULT_CONFIG_PATH,
    RewardParameters,
    ScalarSACBenchmarkConfiguration,
    configuration_sha256,
    load_configuration,
)
from .evaluation import BenchmarkRunResult, evaluate_policy, save_run_result
from .reward import ScalarBenchmarkReward


def _base_environment(configuration: ScalarSACBenchmarkConfiguration):
    import gymnasium as gym

    import gym_longicontrol  # noqa: F401

    return gym.make(
        configuration.environment_id,
        max_episode_steps=configuration.max_episode_steps,
    )


def _training_environment(
    configuration: ScalarSACBenchmarkConfiguration,
    parameters: RewardParameters,
):
    return ScalarBenchmarkReward(
        _base_environment(configuration),
        parameters=parameters,
        task=configuration.task,
        energy_normalization_kwh=configuration.energy_normalization_kwh,
        speed_violation_normalization_m=(
            configuration.speed_violation_normalization_m
        ),
    )


def _agent_arguments(
    configuration: ScalarSACBenchmarkConfiguration,
    *,
    training_seed: int,
    device: str,
) -> argparse.Namespace:
    training = configuration.training
    return get_args(
        [
            "--env_id",
            configuration.environment_id,
            "--seed",
            str(training_seed),
            "--device",
            device,
            "--replay_buffer_capacity",
            str(training.replay_buffer_capacity),
            "--optimization_batch",
            str(training.batch_size),
            "--hidden_layer_sizes",
            *(str(width) for width in training.hidden_layer_sizes),
            "--adam_lr",
            str(training.learning_rate),
            "--discount_factor_gamma",
            str(training.discount_factor),
            "--soft_update_factor_tau",
            str(training.soft_update_factor),
        ]
    )


def _policy_adapter(agent: Any):
    def policy(observation: np.ndarray) -> np.ndarray:
        return agent._policy_action(  # noqa: SLF001 - local trainer adapter
            np.asarray(observation, dtype=np.float32), deterministic=True
        )

    return policy


def run_configuration(
    configuration: ScalarSACBenchmarkConfiguration,
    parameters: RewardParameters,
    *,
    training_seed: int,
    output_directory: str | Path,
    device: str = "auto",
    training_steps: int | None = None,
    initial_random_steps: int | None = None,
    evaluation_seeds: Sequence[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Train and evaluate one grid point without using reward as an outcome."""

    steps = (
        configuration.training.total_training_steps
        if training_steps is None
        else int(training_steps)
    )
    initial_steps = (
        configuration.training.initial_random_steps
        if initial_random_steps is None
        else int(initial_random_steps)
    )
    seeds = tuple(evaluation_seeds or configuration.evaluation_seeds)
    evaluation_set_id = configuration.evaluation_set_id
    if seeds != configuration.evaluation_seeds:
        seed_list = ",".join(str(seed) for seed in seeds)
        evaluation_set_id = f"{evaluation_set_id}:subset[{seed_list}]"
    if steps <= 0 or initial_steps <= 0:
        raise ValueError("Training and initial-random step counts must be positive")
    if initial_steps > configuration.training.replay_buffer_capacity:
        raise ValueError("Initial-random steps exceed replay capacity")

    run_directory = (
        Path(output_directory)
        / parameters.configuration_id
        / f"training-seed-{training_seed}"
    )
    result_path = run_directory / "result.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"Result already exists: {result_path}")
    run_directory.mkdir(parents=True, exist_ok=True)

    training_env = _training_environment(configuration, parameters)
    evaluation_env = _base_environment(configuration)
    try:
        args = _agent_arguments(
            configuration, training_seed=training_seed, device=device
        )
        agent = _build_agent(args, training_env, evaluation_env)
        initial_policy = InitPolicy.from_action_space(
            training_env.action_space, seed=training_seed
        )
        agent.init_replay_buffer(
            initial_policy, initial_steps, seed=training_seed + 1
        )
        agent.do_training(steps, seed=training_seed + 2)
        episodes, summary = evaluate_policy(
            _policy_adapter(agent),
            evaluation_env,
            task=configuration.task,
            evaluation_seeds=seeds,
        )
        result = BenchmarkRunResult(
            benchmark_name=configuration.name,
            configuration_sha256=configuration_sha256(configuration),
            environment_id=configuration.environment_id,
            evaluation_set_id=evaluation_set_id,
            training_seed=training_seed,
            training_steps=steps,
            task=configuration.task,
            reward_parameters=parameters,
            energy_normalization_kwh=configuration.energy_normalization_kwh,
            speed_violation_normalization_m=(
                configuration.speed_violation_normalization_m
            ),
            episodes=episodes,
            summary=summary,
        )
        with (run_directory / "run-config.json").open("w", encoding="utf-8") as stream:
            json.dump(
                {
                    "benchmark": asdict(configuration),
                    "configuration_sha256": configuration_sha256(configuration),
                    "reward_parameters": asdict(parameters),
                    "training_seed": training_seed,
                    "training_steps": steps,
                    "initial_random_steps": initial_steps,
                    "evaluation_seeds": list(seeds),
                    "device": str(agent.device),
                    "runtime": {
                        "torch_num_threads": __import__("torch").get_num_threads(),
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                    },
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        save_checkpoint(
            run_directory / "checkpoint.tar",
            agent,
            epoch=1,
            history={"training_steps": [steps]},
            metadata={
                "benchmark_name": configuration.name,
                "reward_parameters": asdict(parameters),
                "training_seed": training_seed,
                "training_steps": steps,
            },
        )
        # The result is the completion marker and is therefore written last.
        save_run_result(result_path, result)
    finally:
        training_env.close()
        evaluation_env.close()
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/scalar-sac"))
    parser.add_argument("--reward-id", action="append")
    parser.add_argument("--training-seed", action="append", type=int)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--initial-random-steps", type=int)
    parser.add_argument("--evaluation-seeds", nargs="+", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent policies to train concurrently.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one configuration with 64 updates and two evaluation tracks.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configuration = load_configuration(args.config)
    reward_grid = configuration.reward_grid
    training_seeds = configuration.training_seeds
    evaluation_seeds = args.evaluation_seeds
    training_steps = args.training_steps
    initial_random_steps = args.initial_random_steps

    if args.reward_id:
        selected = set(args.reward_id)
        reward_grid = tuple(
            item for item in reward_grid if item.configuration_id in selected
        )
        missing = selected - {item.configuration_id for item in reward_grid}
        if missing:
            raise ValueError(f"Unknown reward configuration(s): {sorted(missing)}")
    if args.training_seed:
        training_seeds = tuple(args.training_seed)
    if args.smoke:
        reward_grid = reward_grid[:1]
        training_seeds = training_seeds[:1]
        evaluation_seeds = configuration.evaluation_seeds[:2]
        training_steps = 64
        initial_random_steps = max(configuration.training.batch_size, 256)
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    jobs = [
        (parameters, training_seed)
        for parameters in reward_grid
        for training_seed in training_seeds
    ]
    keyword_arguments = {
        "output_directory": args.output_dir,
        "device": args.device,
        "training_steps": training_steps,
        "initial_random_steps": initial_random_steps,
        "evaluation_seeds": evaluation_seeds,
        "overwrite": args.overwrite,
    }
    if args.workers == 1:
        for parameters, training_seed in jobs:
            path = run_configuration(
                configuration,
                parameters,
                training_seed=training_seed,
                **keyword_arguments,
            )
            print(path, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_configuration,
                    configuration,
                    parameters,
                    training_seed=training_seed,
                    **keyword_arguments,
                ): (parameters.configuration_id, training_seed)
                for parameters, training_seed in jobs
            }
            for future in as_completed(futures):
                identifier, training_seed = futures[future]
                path = future.result()
                print(
                    f"completed {identifier}/seed-{training_seed}: {path}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
