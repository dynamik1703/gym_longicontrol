"""Run the preregistered SB3 SAC credit-assignment experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.logger import KVWriter

from benchmarks.scalar_sac.evaluation import evaluate_policy
from benchmarks.scalar_sac.experiment import _base_environment
from benchmarks.scalar_sac.v2_reward import ScalarBenchmarkRewardV2

from .action_repeat import ActionRepeat
from .config import (
    DEFAULT_CONFIG_PATH,
    CreditAssignmentConfiguration,
    CreditCondition,
    configuration_sha256,
    load_configuration,
)
from .results import CreditAssignmentRunResult, load_result, save_result

DIAGNOSTIC_KEYS = {
    "rollout/ep_rew_mean",
    "train/actor_loss",
    "train/critic_loss",
    "train/ent_coef",
    "train/ent_coef_loss",
    "train/n_updates",
}


def _training_environment(
    configuration: CreditAssignmentConfiguration,
    condition: CreditCondition,
) -> ActionRepeat:
    reward_environment = ScalarBenchmarkRewardV2(
        _base_environment(configuration),
        parameters=configuration.reward,
        max_time_s=configuration.task.max_time_s,
        max_speed_violation_m_s=configuration.task.max_speed_violation_m_s,
        energy_normalization_kwh=configuration.energy_normalization_kwh,
        speed_violation_normalization_m=(
            configuration.speed_violation_normalization_m
        ),
    )
    return ActionRepeat(
        reward_environment,
        condition.action_repeat,
        max_simulator_steps=configuration.simulator_step_budget,
    )


def _evaluation_environment(
    configuration: CreditAssignmentConfiguration,
    condition: CreditCondition,
) -> ActionRepeat:
    return ActionRepeat(
        _base_environment(configuration),
        condition.action_repeat,
    )


def _policy_adapter(model: Any):
    def policy(observation: np.ndarray) -> np.ndarray:
        action, _state = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64).reshape((1,))

    return policy


def _numeric(value: Any) -> float | None:
    try:
        result = float(np.asarray(value).item())
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


class DiagnosticWriter(KVWriter):
    """Capture standard SB3 SAC diagnostics, indexed by agent decisions."""

    def __init__(self):
        self.latest: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []

    def write(
        self,
        key_values: Mapping[str, Any],
        key_excluded: Mapping[str, Any],
        step: int = 0,
    ) -> None:
        del key_excluded
        row = {"agent_decisions": int(step)}
        for name, value in key_values.items():
            numeric = _numeric(value)
            if numeric is not None:
                row[name] = numeric
                self.latest[name] = numeric
        self.history.append(row)

    def close(self) -> None:
        return None


def _model(
    configuration: CreditAssignmentConfiguration,
    condition: CreditCondition,
    environment,
    training_seed: int,
    device: str,
):
    from stable_baselines3 import SAC

    values = configuration.sac
    return SAC(
        policy="MlpPolicy",
        env=environment,
        seed=training_seed,
        device=device,
        verbose=0,
        learning_rate=values.learning_rate,
        buffer_size=values.buffer_size,
        learning_starts=values.learning_starts,
        batch_size=values.batch_size,
        tau=values.tau,
        gamma=condition.gamma,
        train_freq=values.train_freq,
        gradient_steps=values.gradient_steps,
        ent_coef=values.ent_coef,
        policy_kwargs={"net_arch": list(values.policy_network)},
    )


def _callback(
    *,
    configuration,
    condition,
    training_seed,
    model,
    environment,
    run_directory,
    diagnostics_writer,
    started_at,
):
    from stable_baselines3 import __version__ as sb3_version
    from stable_baselines3.common.callbacks import BaseCallback

    class SimulatorMilestoneCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._remaining = list(configuration.simulator_step_checkpoints)
            self._evaluation_time_s = 0.0

        def _on_step(self) -> bool:
            simulator_steps = environment.simulator_steps
            if not self._remaining or simulator_steps < self._remaining[0]:
                return True
            target = self._remaining.pop(0)
            checkpoint_directory = (
                run_directory / f"simulator-step-{simulator_steps:06d}"
            )
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint_directory / "model")
            evaluation_started = time.perf_counter()
            pending = []
            for split_id, seeds in (
                (
                    "development-2000-2008-v1",
                    configuration.track_splits.development_calibration,
                ),
                ("validation-3000-3008-v1", configuration.track_splits.validation),
            ):
                evaluation_environment = _evaluation_environment(
                    configuration, condition
                )
                try:
                    episodes, summary = evaluate_policy(
                        _policy_adapter(model),
                        evaluation_environment,
                        task=configuration.task,
                        evaluation_seeds=seeds,
                    )
                finally:
                    evaluation_environment.close()
                pending.append((split_id, episodes, summary))
            self._evaluation_time_s += time.perf_counter() - evaluation_started
            training_wall_time = (
                time.perf_counter() - started_at - self._evaluation_time_s
            )
            latest = {
                name: value
                for name, value in diagnostics_writer.latest.items()
                if name in DIAGNOSTIC_KEYS
            }
            updates = int(getattr(model, "_n_updates", 0))
            for split_id, episodes, summary in pending:
                result = CreditAssignmentRunResult(
                    benchmark_name=configuration.name,
                    configuration_sha256=configuration_sha256(configuration),
                    environment_id=configuration.environment_id,
                    evaluation_split_id=split_id,
                    condition_id=condition.condition_id,
                    condition_label=condition.label,
                    gamma=condition.gamma,
                    action_repeat=condition.action_repeat,
                    library_version=sb3_version,
                    training_seed=training_seed,
                    simulator_step_target=target,
                    simulator_steps=simulator_steps,
                    agent_decisions=self.num_timesteps,
                    gradient_updates=updates,
                    training_wall_time_s=training_wall_time,
                    task=configuration.task,
                    reward_parameters=configuration.reward,
                    episodes=episodes,
                    summary=summary,
                    diagnostics=latest,
                )
                filename = split_id.split("-v1")[0] + "-result.json"
                save_result(checkpoint_directory / filename, result)
            diagnostics_path = run_directory / "training-diagnostics.json"
            diagnostics_path.write_text(
                json.dumps(diagnostics_writer.history, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            return bool(self._remaining)

    return SimulatorMilestoneCallback()


def run_policy(
    configuration: CreditAssignmentConfiguration,
    condition: CreditCondition,
    *,
    training_seed: int,
    output_directory: str | Path,
    device: str = "auto",
    overwrite: bool = False,
) -> Path:
    from stable_baselines3 import __version__ as sb3_version
    from stable_baselines3.common.logger import Logger

    run_directory = (
        Path(output_directory)
        / condition.condition_id
        / f"training-seed-{training_seed}"
    )
    existing_results = tuple(run_directory.glob("simulator-step-*/*-result.json"))
    if existing_results and not overwrite:
        raise FileExistsError(
            f"Credit-assignment run already exists or is partial: {run_directory}"
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    environment = _training_environment(configuration, condition)
    try:
        model = _model(configuration, condition, environment, training_seed, device)
        diagnostics_writer = DiagnosticWriter()
        model.set_logger(Logger(folder=None, output_formats=[diagnostics_writer]))
        (run_directory / "run-config.json").write_text(
            json.dumps(
                {
                    "benchmark": asdict(configuration),
                    "configuration_sha256": configuration_sha256(configuration),
                    "condition": asdict(condition),
                    "algorithm": "stable-baselines3-sac",
                    "training_seed": training_seed,
                    "stable_baselines3_version": sb3_version,
                    "interaction_budget": {
                        "unit": "underlying simulator transitions",
                        "simulator_steps": configuration.simulator_step_budget,
                    },
                    "observation_normalization": None,
                    "reward_normalization": None,
                    "deterministic_evaluation": True,
                    "device": str(model.device),
                    "runtime": {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
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
        started_at = time.perf_counter()
        callback = _callback(
            configuration=configuration,
            condition=condition,
            training_seed=training_seed,
            model=model,
            environment=environment,
            run_directory=run_directory,
            diagnostics_writer=diagnostics_writer,
            started_at=started_at,
        )
        model.learn(
            total_timesteps=configuration.simulator_step_budget,
            callback=callback,
            log_interval=1,
            progress_bar=False,
        )
        final_results = tuple(run_directory.glob("simulator-step-*/*-result.json"))
        final_validation = [
            load_result(path)
            for path in final_results
            if path.name == "validation-3000-3008-result.json"
        ]
        if not any(
            item.simulator_step_target == configuration.simulator_step_budget
            and item.simulator_steps == configuration.simulator_step_budget
            for item in final_validation
        ):
            raise RuntimeError(
                "Training stopped without exact final simulator budget: "
                f"{run_directory}"
            )
    finally:
        environment.close()
    return next(
        path
        for path in final_results
        if path.name == "validation-3000-3008-result.json"
        and load_result(path).simulator_step_target
        == configuration.simulator_step_budget
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/credit-assignment")
    )
    parser.add_argument("--condition", action="append")
    parser.add_argument("--training-seed", action="append", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one condition and seed for 2,048 simulator transitions.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    configuration = load_configuration(args.config)
    requested = set(args.condition or ())
    known = {item.condition_id for item in configuration.conditions}
    if requested - known:
        raise ValueError(f"Unknown conditions: {sorted(requested - known)}")
    conditions = tuple(
        item
        for item in configuration.conditions
        if not requested or item.condition_id in requested
    )
    training_seeds = tuple(args.training_seed or configuration.training_seeds)
    if not set(training_seeds) <= set(configuration.training_seeds):
        raise ValueError("Training seed is outside the preregistered set")
    if args.smoke:
        conditions = conditions[:1]
        training_seeds = training_seeds[:1]
        configuration = replace(
            configuration,
            track_splits=replace(
                configuration.track_splits,
                development_calibration=(2000,),
                validation=(3000,),
            ),
            simulator_step_checkpoints=(2_048,),
        )
    jobs = [
        (condition, seed)
        for condition in conditions
        for seed in training_seeds
    ]
    arguments = {
        "output_directory": args.output_dir,
        "device": args.device,
        "overwrite": args.overwrite,
    }
    if args.workers == 1:
        for condition, training_seed in jobs:
            print(
                run_policy(
                    configuration,
                    condition,
                    training_seed=training_seed,
                    **arguments,
                ),
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_policy,
                    configuration,
                    condition,
                    training_seed=training_seed,
                    **arguments,
                ): (condition.condition_id, training_seed)
                for condition, training_seed in jobs
            }
            for future in as_completed(futures):
                condition_id, seed = futures[future]
                print(
                    f"completed {condition_id}/seed-{seed}: {future.result()}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
