"""Train frozen SACLag with an explicit episode deadline observation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.scalar_sac.evaluation import EpisodeRecorder
from benchmarks.scalar_sac.experiment import _base_environment
from gym_longicontrol.domain.metrics import EpisodeMetrics

from .adapter import (
    alpha_value,
    build_agent,
    collect_training_episode,
    deterministic_policy_adapter,
    logger_snapshot,
    multiplier_values,
    save_checkpoint,
)
from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .requirements import BalancedMarginSampler, RequirementConditionedTaskWrapper
from .results import RequirementEpisodeEvaluation, make_result, save_result


def _environment(configuration, *, sampler_seed=0, initial_seed=None, budget=None):
    return RequirementConditionedTaskWrapper(
        _base_environment(configuration),
        margins_s=configuration.requirements.training_margins_s,
        sampler_seed=sampler_seed,
        time_scale_s=configuration.requirements.time_scale_s,
        maximum_t_min_start_s=configuration.requirements.maximum_t_min_start_s,
        energy_scale_kwh=configuration.objective.energy_scale_kwh,
        max_speed_violation_m_s=configuration.max_speed_violation_m_s,
        initial_seed=initial_seed,
        max_simulator_steps=budget,
    )


def _requirement_kind(configuration, margin):
    if margin in configuration.requirements.training_margins_s:
        return "seen"
    if margin in configuration.requirements.interpolation_margins_s:
        return "interpolation"
    return "canonical-140"


def _action_profile(positions, actions, track_length_m=1000.0):
    grid = np.linspace(0.0, track_length_m, 101)
    return tuple(
        float(value)
        for value in np.interp(
            grid,
            np.asarray(positions, dtype=np.float64),
            np.asarray(actions, dtype=np.float64),
        )
    )


def evaluate(policy, configuration, seeds, margins):
    environment = _environment(configuration, sampler_seed=0)
    act = deterministic_policy_adapter(policy)
    episodes = []
    policy.eval()
    try:
        for track_seed in seeds:
            for requested in margins:
                if requested == "canonical-140":
                    observation, info = environment.reset(seed=track_seed)
                    margin = 140.0 - float(info["requirement_t_min_start_s"])
                    observation, info = environment.reset(
                        seed=track_seed, options={"requirement_margin_s": margin}
                    )
                else:
                    margin = float(requested)
                    observation, info = environment.reset(
                        seed=track_seed, options={"requirement_margin_s": margin}
                    )
                recorder = EpisodeRecorder()
                positions = [0.0]
                actions = [0.0]
                deadline_integral = 0.0
                maximum_deficit = 0.0
                while True:
                    action = act(observation)
                    observation, _reward, terminated, truncated, info = (
                        environment.step(action)
                    )
                    recorder.observe(action, info)
                    positions.append(float(info["position_m"]))
                    actions.append(float(action[0]))
                    deadline_integral += float(info["deadline_deficit_cost_s"])
                    maximum_deficit = max(
                        maximum_deficit, float(info["deadline_deficit_s"])
                    )
                    if terminated or truncated:
                        metrics = EpisodeMetrics(**info["episode_metrics"])
                        task = environment.current_task
                        base = recorder.finalize(track_seed, metrics, task)
                        episodes.append(
                            RequirementEpisodeEvaluation(
                                track_seed=track_seed,
                                requirement_kind=_requirement_kind(
                                    configuration, margin
                                ),
                                requirement_margin_s=margin,
                                t_min_start_s=float(info["requirement_t_min_start_s"]),
                                max_time_s=float(info["requirement_max_time_s"]),
                                completed=base.completed,
                                deadline_met=base.travel_time_s <= task.max_time_s,
                                speed_compliant=(
                                    base.max_speed_violation_m_s
                                    <= task.max_speed_violation_m_s
                                ),
                                feasible=base.feasible,
                                travel_time_s=base.travel_time_s,
                                energy_kwh=base.energy_kwh,
                                speed_violation_count=base.speed_violation_count,
                                max_speed_violation_m_s=base.max_speed_violation_m_s,
                                integrated_speed_violation_m=(
                                    base.integrated_speed_violation_m
                                ),
                                deadline_deficit_integral_s=deadline_integral,
                                maximum_deadline_deficit_s=maximum_deficit,
                                step_count=base.step_count,
                                final_position_m=base.final_position_m,
                                traction_energy_kwh=base.traction_energy_kwh,
                                regenerative_energy_kwh=base.regenerative_energy_kwh,
                                mean_abs_jerk_m_s3=base.mean_abs_jerk_m_s3,
                                max_abs_jerk_m_s3=base.max_abs_jerk_m_s3,
                                mean_abs_acceleration_m_s2=(
                                    base.mean_abs_acceleration_m_s2
                                ),
                                mean_abs_action=base.mean_abs_action,
                                action_total_variation=base.action_total_variation,
                                mean_abs_action_change=base.mean_abs_action_change,
                                acceleration_sign_change_count=(
                                    base.acceleration_sign_change_count
                                ),
                                action_profile=_action_profile(positions, actions),
                            )
                        )
                        break
    finally:
        environment.close()
        policy.train()
    return tuple(episodes)


def _write_json(path: Path, payload: Any):
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _training_row(environment, completed, stats, updates, policy, logger):
    objective_return = float(np.asarray(stats["rew"]).item())
    if not np.isclose(objective_return, completed.objective_return, atol=1e-10):
        raise RuntimeError("FSRL and physical objective returns differ")
    return {
        "simulator_steps": environment.simulator_steps,
        "gradient_updates": updates,
        "episode_steps": completed.simulator_steps,
        "requirement_margin_s": completed.requirement_margin_s,
        "t_min_start_s": completed.t_min_start_s,
        "max_time_s": completed.max_time_s,
        "objective_return": completed.objective_return,
        "energy_kwh": completed.metrics.energy_kwh,
        "speed_integral_m": completed.costs[0],
        "deadline_deficit_integral_s": completed.costs[1],
        "maximum_deadline_deficit_s": completed.maximum_deadline_deficit_s,
        "completed": completed.metrics.completed,
        "travel_time_s": completed.metrics.travel_time_s,
        "budget_truncated": completed.budget_truncated,
        "lagrange_speed": multiplier_values(policy)[0],
        "lagrange_deadline": multiplier_values(policy)[1],
        "alpha": alpha_value(policy),
        **logger_snapshot(logger),
    }


def run_policy(
    configuration, *, training_seed, output_directory, device="cpu", threads=4
):
    import fsrl
    import tianshou
    from fsrl.data import FastCollector
    from tianshou.data import ReplayBuffer

    if tianshou.__version__ != configuration.algorithm.tianshou_version:
        raise RuntimeError("Pinned Tianshou version differs")
    run_directory = Path(output_directory) / f"training-seed-{training_seed}"
    marker = run_directory / "target-300000" / "validation-result.json"
    if marker.exists():
        raise FileExistsError(f"Run already exists: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    environment = _environment(
        configuration,
        sampler_seed=90_000 + training_seed,
        initial_seed=training_seed,
        budget=configuration.simulator_step_budget,
    )
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
                "observation_features": ["T_max / 180", "elapsed_time / 180"],
                "runtime": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "torch_num_threads": __import__("torch").get_num_threads(),
                    "requested_threads": threads,
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                },
            },
        )
        pending = list(configuration.simulator_step_checkpoints)
        training_margin_sampler = BalancedMarginSampler(
            configuration.requirements.training_margins_s,
            seed=90_000 + training_seed,
        )
        diagnostics = []
        updates = 0
        evaluation_time = 0.0
        started = time.perf_counter()
        while pending:
            stats, completed = collect_training_episode(
                collector,
                environment,
                requirement_margin_s=training_margin_sampler.sample(),
            )
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
                updates += 1
            policy.post_update_fn(stats_train=stats)
            row = _training_row(environment, completed, stats, updates, policy, logger)
            diagnostics.append(row)
            _write_json(run_directory / "training-diagnostics.json", diagnostics)
            while pending and environment.simulator_steps >= pending[0]:
                target = pending.pop(0)
                actual = environment.simulator_steps
                checkpoint = run_directory / f"target-{target:06d}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    checkpoint / "policy.pt",
                    policy,
                    metadata={
                        "configuration_sha256": config_hash,
                        "training_seed": training_seed,
                        "simulator_step_target": target,
                        "simulator_steps": actual,
                        "gradient_updates": updates,
                    },
                )
                evaluation_started = time.perf_counter()
                margins = configuration.requirements.evaluation_margins_s
                evaluations = (
                    (
                        "development",
                        configuration.track_splits.development_calibration,
                        margins,
                    ),
                    ("validation", configuration.track_splits.validation, margins),
                )
                for split_id, seeds, values in evaluations:
                    episodes = evaluate(policy, configuration, seeds, values)
                    evaluation_time += time.perf_counter() - evaluation_started
                    wall_time = time.perf_counter() - started - evaluation_time
                    result = make_result(
                        benchmark_name=configuration.name,
                        configuration_sha256=config_hash,
                        evaluation_split_id=f"{split_id}-requirements-v1",
                        training_seed=training_seed,
                        simulator_step_target=target,
                        simulator_steps=actual,
                        gradient_updates=updates,
                        training_wall_time_s=wall_time,
                        episodes=episodes,
                        checkpoint_multipliers=multiplier_values(policy),
                    )
                    save_result(checkpoint / f"{split_id}-result.json", result)
                    evaluation_started = time.perf_counter()
                if target == configuration.simulator_step_budget:
                    episodes = evaluate(
                        policy,
                        configuration,
                        configuration.track_splits.validation,
                        ("canonical-140",),
                    )
                    evaluation_time += time.perf_counter() - evaluation_started
                    wall_time = time.perf_counter() - started - evaluation_time
                    save_result(
                        checkpoint / "canonical-140-result.json",
                        make_result(
                            benchmark_name=configuration.name,
                            configuration_sha256=config_hash,
                            evaluation_split_id="validation-canonical-140-v1",
                            training_seed=training_seed,
                            simulator_step_target=target,
                            simulator_steps=actual,
                            gradient_updates=updates,
                            training_wall_time_s=wall_time,
                            episodes=episodes,
                            checkpoint_multipliers=multiplier_values(policy),
                        ),
                    )
                print(
                    f"seed={training_seed} target={target} actual={actual} "
                    f"updates={updates}",
                    flush=True,
                )
        if environment.simulator_steps != configuration.simulator_step_budget:
            raise RuntimeError("Final simulator budget is not exact")
    finally:
        environment.close()
    return marker


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/requirement-conditioned")
    )
    parser.add_argument("--training-seed", action="append", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    configuration = load_configuration(args.config)
    seeds = tuple(args.training_seed or configuration.training_seeds)
    kwargs = {
        "output_directory": args.output_dir,
        "device": args.device,
        "threads": args.threads,
    }
    if args.workers == 1:
        for seed in seeds:
            print(run_policy(configuration, training_seed=seed, **kwargs), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_policy, configuration, training_seed=seed, **kwargs
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                print(
                    f"completed seed-{futures[future]}: {future.result()}", flush=True
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
