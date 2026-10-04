"""Train SB3 SAC and PPO under the frozen scalar benchmark protocol."""

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

from .config import (
    DEFAULT_CONFIG_PATH,
    ScalarSB3Configuration,
    configuration_sha256,
    load_configuration,
)
from .results import SB3BenchmarkRunResult, save_result

DIAGNOSTIC_KEYS = {
    "rollout/ep_rew_mean",
    "train/actor_loss",
    "train/critic_loss",
    "train/ent_coef",
    "train/ent_coef_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/n_updates",
}


def _training_environment(configuration: ScalarSB3Configuration):
    return ScalarBenchmarkRewardV2(
        _base_environment(configuration),
        parameters=configuration.reward,
        max_time_s=configuration.task.max_time_s,
        max_speed_violation_m_s=configuration.task.max_speed_violation_m_s,
        energy_normalization_kwh=configuration.energy_normalization_kwh,
        speed_violation_normalization_m=(
            configuration.speed_violation_normalization_m
        ),
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
    """Keep SB3's public logger output without relying on TensorBoard files."""

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
        row = {"training_steps": int(step)}
        for name, value in key_values.items():
            numeric = _numeric(value)
            if numeric is not None:
                row[name] = numeric
                self.latest[name] = numeric
        self.history.append(row)

    def close(self) -> None:
        return None


def _model(configuration, algorithm, environment, training_seed, device):
    from stable_baselines3 import PPO, SAC

    common = {
        "policy": "MlpPolicy",
        "env": environment,
        "seed": training_seed,
        "device": device,
        "verbose": 0,
    }
    if algorithm == "sac":
        values = configuration.sac
        return SAC(
            **common,
            learning_rate=values.learning_rate,
            buffer_size=values.buffer_size,
            learning_starts=values.learning_starts,
            batch_size=values.batch_size,
            tau=values.tau,
            gamma=values.gamma,
            train_freq=values.train_freq,
            gradient_steps=values.gradient_steps,
            ent_coef=values.ent_coef,
            policy_kwargs={"net_arch": list(values.policy_network)},
        )
    if algorithm == "ppo":
        values = configuration.ppo
        return PPO(
            **common,
            learning_rate=values.learning_rate,
            n_steps=values.n_steps,
            batch_size=values.batch_size,
            n_epochs=values.n_epochs,
            gamma=values.gamma,
            gae_lambda=values.gae_lambda,
            clip_range=values.clip_range,
            ent_coef=values.ent_coef,
            vf_coef=values.vf_coef,
            policy_kwargs={"net_arch": list(values.policy_network)},
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _callback(
    *,
    configuration,
    algorithm,
    training_seed,
    model,
    run_directory,
    diagnostics_writer,
    started_at,
):
    from stable_baselines3 import __version__ as sb3_version
    from stable_baselines3.common.callbacks import BaseCallback

    class MilestoneCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._remaining = list(configuration.learning_curve_steps)
            self._evaluation_time_s = 0.0

        def _on_step(self) -> bool:
            if not self._remaining or self.num_timesteps < self._remaining[0]:
                return True
            milestone = self._remaining.pop(0)
            if self.num_timesteps != milestone:
                raise RuntimeError(
                    f"Expected exact milestone {milestone}, got {self.num_timesteps}"
                )
            checkpoint_directory = run_directory / f"step-{milestone:06d}"
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint_directory / "model.zip")
            evaluation_started = time.perf_counter()
            pending = []
            for split_id, seeds in (
                (
                    "development-2000-2008-v1",
                    configuration.track_splits.development_calibration,
                ),
                ("validation-3000-3008-v1", configuration.track_splits.validation),
            ):
                environment = _base_environment(configuration)
                try:
                    episodes, summary = evaluate_policy(
                        _policy_adapter(model),
                        environment,
                        task=configuration.task,
                        evaluation_seeds=seeds,
                    )
                finally:
                    environment.close()
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
                result = SB3BenchmarkRunResult(
                    benchmark_name=configuration.name,
                    configuration_sha256=configuration_sha256(configuration),
                    environment_id=configuration.environment_id,
                    evaluation_split_id=split_id,
                    algorithm=algorithm,
                    library_version=sb3_version,
                    training_seed=training_seed,
                    training_steps=milestone,
                    training_wall_time_s=training_wall_time,
                    policy_updates=updates,
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

    return MilestoneCallback()


def run_policy(
    configuration: ScalarSB3Configuration,
    algorithm: str,
    *,
    training_seed: int,
    output_directory: str | Path,
    device: str = "auto",
    overwrite: bool = False,
) -> Path:
    from stable_baselines3 import __version__ as sb3_version
    from stable_baselines3.common.logger import Logger

    run_directory = (
        Path(output_directory) / algorithm / f"training-seed-{training_seed}"
    )
    marker = (
        run_directory
        / f"step-{configuration.total_training_steps:06d}"
        / "validation-3000-3008-result.json"
    )
    existing_results = tuple(run_directory.glob("step-*/*-result.json"))
    if (marker.exists() or existing_results) and not overwrite:
        raise FileExistsError(f"SB3 run already exists or is partial: {run_directory}")
    run_directory.mkdir(parents=True, exist_ok=True)
    environment = _training_environment(configuration)
    try:
        model = _model(
            configuration, algorithm, environment, training_seed, device
        )
        diagnostics_writer = DiagnosticWriter()
        model.set_logger(Logger(folder=None, output_formats=[diagnostics_writer]))
        resolved_device = str(model.device)
        (run_directory / "run-config.json").write_text(
            json.dumps(
                {
                    "benchmark": asdict(configuration),
                    "configuration_sha256": configuration_sha256(configuration),
                    "algorithm": algorithm,
                    "training_seed": training_seed,
                    "stable_baselines3_version": sb3_version,
                    "observation_normalization": None,
                    "reward_normalization": None,
                    "deterministic_evaluation": True,
                    "device": resolved_device,
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
            algorithm=algorithm,
            training_seed=training_seed,
            model=model,
            run_directory=run_directory,
            diagnostics_writer=diagnostics_writer,
            started_at=started_at,
        )
        model.learn(
            total_timesteps=configuration.total_training_steps,
            callback=callback,
            log_interval=1,
            progress_bar=False,
        )
        if not marker.exists():
            raise RuntimeError(f"Training stopped without final result: {marker}")
    finally:
        environment.close()
    return marker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/scalar-sb3"))
    parser.add_argument("--algorithm", action="append", choices=("sac", "ppo"))
    parser.add_argument("--training-seed", action="append", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one algorithm and seed for 2,048 steps on one track per split.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    configuration = load_configuration(args.config)
    algorithms = tuple(args.algorithm or ("sac", "ppo"))
    training_seeds = tuple(args.training_seed or configuration.training_seeds)
    if args.smoke:
        algorithms = algorithms[:1]
        training_seeds = training_seeds[:1]
        configuration = replace(
            configuration,
            track_splits=replace(
                configuration.track_splits,
                development_calibration=(2000,),
                validation=(3000,),
            ),
            learning_curve_steps=(2_048,),
        )
    jobs = [(algorithm, seed) for algorithm in algorithms for seed in training_seeds]
    arguments = {
        "output_directory": args.output_dir,
        "device": args.device,
        "overwrite": args.overwrite,
    }
    if args.workers == 1:
        for algorithm, training_seed in jobs:
            print(
                run_policy(
                    configuration,
                    algorithm,
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
                    algorithm,
                    training_seed=training_seed,
                    **arguments,
                ): (algorithm, training_seed)
                for algorithm, training_seed in jobs
            }
            for future in as_completed(futures):
                algorithm, seed = futures[future]
                print(
                    f"completed {algorithm}/seed-{seed}: {future.result()}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
