"""Run the preregistered dense-deadline FSRL SAC-Lagrangian study."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.scalar_sac.evaluation import evaluate_policy
from benchmarks.scalar_sac.experiment import _base_environment

from .adapter import (
    alpha_value,
    build_agent,
    collect_training_episode,
    deterministic_policy_adapter,
    logger_snapshot,
    multiplier_values,
    save_checkpoint,
)
from .config import (
    DEFAULT_CONFIG_PATH,
    ConstrainedRLV2Configuration,
    configuration_sha256,
    load_configuration,
)
from .costs import DenseDeadlineTaskWrapper
from .results import ConstrainedV2RunResult, save_result


def _training_environment(
    configuration: ConstrainedRLV2Configuration, training_seed: int
) -> DenseDeadlineTaskWrapper:
    return DenseDeadlineTaskWrapper(
        _base_environment(configuration),
        task=configuration.task,
        energy_scale_kwh=configuration.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.deadline_cost.normalization_s,
        initial_seed=training_seed,
        max_simulator_steps=configuration.simulator_step_budget,
    )


def _evaluate(policy: Any, configuration: ConstrainedRLV2Configuration, seeds):
    environment = _base_environment(configuration)
    policy.eval()
    try:
        return evaluate_policy(
            deterministic_policy_adapter(policy),
            environment,
            task=configuration.task,
            evaluation_seeds=seeds,
        )
    finally:
        environment.close()
        policy.train()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _training_row(
    *,
    simulator_steps: int,
    gradient_updates: int,
    completed,
    stats,
    multipliers,
    alpha: float,
    losses,
) -> dict[str, Any]:
    objective_return = float(np.asarray(stats["rew"]).item())
    if not np.isclose(objective_return, completed.objective_return, atol=1e-10):
        raise RuntimeError("FSRL and wrapper objective returns differ")
    if not np.isclose(
        completed.costs[0], completed.metrics.integrated_speed_violation_m, atol=1e-10
    ):
        raise RuntimeError("Speed cost diverged from EpisodeMetrics semantics")
    return {
        "simulator_steps": simulator_steps,
        "gradient_updates": gradient_updates,
        "episode_steps": completed.simulator_steps,
        "objective_return": completed.objective_return,
        "energy_kwh": completed.metrics.energy_kwh,
        "speed_integral_m": completed.costs[0],
        "deadline_deficit_integral_s": completed.costs[1],
        "final_deadline_deficit_s": completed.final_deadline_deficit_s,
        "maximum_deadline_deficit_s": completed.maximum_deadline_deficit_s,
        "final_position_m": completed.final_position_m,
        "completed": completed.metrics.completed,
        "travel_time_s": completed.metrics.travel_time_s,
        "budget_truncated": completed.budget_truncated,
        "lagrange_speed": multipliers[0],
        "lagrange_deadline": multipliers[1],
        "alpha": alpha,
        **losses,
    }


def run_policy(
    configuration: ConstrainedRLV2Configuration,
    *,
    training_seed: int,
    output_directory: str | Path,
    device: str = "cpu",
    threads: int = 4,
    overwrite: bool = False,
) -> Path:
    """Train and externally evaluate one independent V2 policy."""

    import fsrl
    import tianshou
    from fsrl.data import FastCollector
    from tianshou.data import ReplayBuffer

    if tianshou.__version__ != configuration.algorithm.tianshou_version:
        raise RuntimeError(
            f"Expected Tianshou {configuration.algorithm.tianshou_version}, "
            f"found {tianshou.__version__}"
        )
    run_directory = Path(output_directory) / f"training-seed-{training_seed}"
    marker = (
        run_directory
        / f"target-{configuration.simulator_step_budget:06d}"
        / "validation-3000-3008-result.json"
    )
    existing_results = tuple(run_directory.glob("target-*/*-result.json"))
    if (marker.exists() or existing_results) and not overwrite:
        raise FileExistsError(f"Constrained V2 run already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    environment = _training_environment(configuration, training_seed)
    try:
        agent, logger = build_agent(
            configuration,
            environment,
            training_seed=training_seed,
            device=device,
            threads=threads,
        )
        policy = agent.policy
        policy.train()
        collector = FastCollector(
            policy,
            environment,
            ReplayBuffer(configuration.algorithm.buffer_size),
            exploration_noise=True,
        )
        config_hash = configuration_sha256(configuration)
        _write_json(
            run_directory / "run-config.json",
            {
                "benchmark": asdict(configuration),
                "configuration_sha256": config_hash,
                "algorithm": "FSRL-SACLag",
                "training_seed": training_seed,
                "fsrl_version": getattr(fsrl, "__version__", "unknown"),
                "fsrl_git_commit": configuration.algorithm.fsrl_git_commit,
                "tianshou_version": tianshou.__version__,
                "task_cost_intervention": (
                    "V1 terminal task_failure replaced by normalized optimistic "
                    "deadline-deficit integral; no terminal failure cost"
                ),
                "vector_cost_adapter": (
                    "split info.cost columns for critics and restore exact episodic "
                    "cost vector after FastCollector aggregation"
                ),
                "observation_normalization": None,
                "reward_normalization": None,
                "device": device,
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch_num_threads": __import__("torch").get_num_threads(),
                    "requested_threads": threads,
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                },
            },
        )
        pending = list(configuration.simulator_step_checkpoints)
        diagnostics: list[dict[str, Any]] = []
        gradient_updates = 0
        evaluation_time_s = 0.0
        started_at = time.perf_counter()
        while pending:
            stats, completed = collect_training_episode(collector, environment)
            logger.reset_data()
            policy.pre_update_fn(
                stats_train=stats,
                batch_size=configuration.algorithm.batch_size,
                buffer=collector.buffer,
                update_per_step=configuration.algorithm.update_per_step,
            )
            update_count = round(
                configuration.algorithm.update_per_step * int(stats["n/st"])
            )
            for _ in range(update_count):
                policy.update(configuration.algorithm.batch_size, collector.buffer)
                gradient_updates += 1
            policy.post_update_fn(stats_train=stats)
            losses = logger_snapshot(logger)
            row = _training_row(
                simulator_steps=environment.simulator_steps,
                gradient_updates=gradient_updates,
                completed=completed,
                stats=stats,
                multipliers=multiplier_values(policy),
                alpha=alpha_value(policy),
                losses=losses,
            )
            diagnostics.append(row)
            _write_json(run_directory / "training-diagnostics.json", diagnostics)
            while pending and environment.simulator_steps >= pending[0]:
                target = pending.pop(0)
                actual = environment.simulator_steps
                checkpoint_directory = run_directory / f"target-{target:06d}"
                checkpoint_directory.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    checkpoint_directory / "policy.pt",
                    policy,
                    metadata={
                        "configuration_sha256": config_hash,
                        "training_seed": training_seed,
                        "simulator_step_target": target,
                        "simulator_steps": actual,
                        "gradient_updates": gradient_updates,
                    },
                )
                evaluation_started = time.perf_counter()
                evaluated = []
                for split_id, seeds in (
                    (
                        "development-2000-2008-v2",
                        configuration.track_splits.development_calibration,
                    ),
                    (
                        "validation-3000-3008-v2",
                        configuration.track_splits.validation,
                    ),
                ):
                    episodes, summary = _evaluate(policy, configuration, seeds)
                    evaluated.append((split_id, episodes, summary))
                evaluation_time_s += time.perf_counter() - evaluation_started
                training_wall_time_s = (
                    time.perf_counter() - started_at - evaluation_time_s
                )
                checkpoint_diagnostics = {
                    "latest_training_episode": row,
                    "checkpoint_overshoot_steps": actual - target,
                    "training_episode_count": len(diagnostics),
                }
                for split_id, episodes, summary in evaluated:
                    result = ConstrainedV2RunResult(
                        benchmark_name=configuration.name,
                        configuration_sha256=config_hash,
                        environment_id=configuration.environment_id,
                        evaluation_split_id=split_id,
                        algorithm="FSRL-SACLag",
                        library_version=getattr(fsrl, "__version__", "unknown"),
                        library_revision=configuration.algorithm.fsrl_git_commit,
                        training_seed=training_seed,
                        simulator_step_target=target,
                        simulator_steps=actual,
                        gradient_updates=gradient_updates,
                        training_wall_time_s=training_wall_time_s,
                        task=configuration.task,
                        objective_name=configuration.objective.name,
                        cost_names=configuration.constraints.names,
                        cost_limits=configuration.constraints.cost_limits,
                        episodes=episodes,
                        summary=summary,
                        diagnostics=checkpoint_diagnostics,
                    )
                    filename = split_id.split("-v2")[0] + "-result.json"
                    save_result(checkpoint_directory / filename, result)
                print(
                    f"seed={training_seed} target={target} actual={actual} "
                    f"updates={gradient_updates}",
                    flush=True,
                )
        if environment.simulator_steps != configuration.simulator_step_budget:
            raise RuntimeError("Final physical simulator budget is not exact")
        if not marker.exists():
            raise RuntimeError(f"Training stopped without final result: {marker}")
    finally:
        environment.close()
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/constrained-rl-v2")
    )
    parser.add_argument("--training-seed", action="append", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the registered one-seed 2,048-transition implementation smoke.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0 or args.threads <= 0:
        raise ValueError("workers and threads must be positive")
    configuration = load_configuration(args.config)
    training_seeds = tuple(args.training_seed or configuration.training_seeds)
    if args.smoke:
        training_seeds = training_seeds[:1]
        configuration = replace(
            configuration,
            name=f"{configuration.name}-smoke",
            track_splits=replace(
                configuration.track_splits,
                development_calibration=(2000,),
                validation=(3000,),
            ),
            simulator_step_checkpoints=(2_048,),
        )
    kwargs = {
        "output_directory": args.output_dir,
        "device": args.device,
        "threads": args.threads,
        "overwrite": args.overwrite,
    }
    if args.workers == 1:
        for seed in training_seeds:
            print(run_policy(configuration, training_seed=seed, **kwargs), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_policy, configuration, training_seed=seed, **kwargs
                ): seed
                for seed in training_seeds
            }
            for future in as_completed(futures):
                print(
                    f"completed seed-{futures[future]}: {future.result()}", flush=True
                )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
