"""Train frozen SB3 SAC with a terminal-only binary success reward."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.logger import KVWriter

from benchmarks.scalar_sac.evaluation import evaluate_policy
from benchmarks.scalar_sac.experiment import _base_environment
from benchmarks.scalar_sb3.experiment import _model, _policy_adapter

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import BinaryRunResult, save_result
from .reward import BinarySuccessReward

DIAGNOSTIC_KEYS = {
    "rollout/ep_rew_mean",
    "train/actor_loss",
    "train/critic_loss",
    "train/ent_coef",
    "train/ent_coef_loss",
    "train/n_updates",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _training_environment(configuration):
    return BinarySuccessReward(
        _base_environment(configuration), task=configuration.task
    )


def _numeric(value: Any) -> float | None:
    try:
        result = float(np.asarray(value).item())
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


class DiagnosticWriter(KVWriter):
    """Capture SB3 optimizer diagnostics without TensorBoard."""

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


def _failure_mode(metrics, configuration) -> str:
    failures = []
    if not metrics["completed"]:
        failures.append("incomplete")
    if metrics["travel_time_s"] > configuration.task.max_time_s:
        failures.append("time")
    if metrics["max_speed_violation_m_s"] > configuration.task.max_speed_violation_m_s:
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def _training_outcome_payload(configuration, training_seed, outcomes):
    successes = [row for row in outcomes if row["success"]]
    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": configuration_sha256(configuration),
        "training_seed": training_seed,
        "completed_training_episode_count": len(outcomes),
        "successful_training_episode_count": len(successes),
        "first_success_training_step": (
            successes[0]["training_step"] if successes else None
        ),
        "last_success_training_step": (
            successes[-1]["training_step"] if successes else None
        ),
        "episodes": outcomes,
    }


def _callback(
    *,
    configuration,
    training_seed,
    model,
    run_directory,
    diagnostics_writer,
    started_at,
):
    from stable_baselines3 import __version__ as sb3_version
    from stable_baselines3.common.callbacks import BaseCallback

    class BinaryMilestoneCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._remaining = list(configuration.learning_curve_steps)
            self._evaluation_time_s = 0.0
            self._outcomes = []

        def _record_finished_episode(self) -> None:
            dones = np.asarray(self.locals["dones"]).reshape(-1)
            if not bool(dones[0]):
                return
            info = self.locals["infos"][0]
            metrics = info["episode_metrics"]
            success = bool(info["binary_reward_success"])
            reward = float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
            if not info["binary_reward_outcome_known"] or reward != float(success):
                raise RuntimeError("Binary terminal reward and outcome disagree")
            self._outcomes.append(
                {
                    "episode_index": len(self._outcomes) + 1,
                    "training_step": int(self.num_timesteps),
                    "reward": reward,
                    "success": success,
                    "failure_mode": _failure_mode(metrics, configuration),
                    "completed": bool(metrics["completed"]),
                    "travel_time_s": float(metrics["travel_time_s"]),
                    "energy_kwh": float(metrics["energy_kwh"]),
                    "speed_violation_count": int(metrics["speed_violation_count"]),
                    "max_speed_violation_m_s": float(
                        metrics["max_speed_violation_m_s"]
                    ),
                    "integrated_speed_violation_m": float(
                        metrics["integrated_speed_violation_m"]
                    ),
                    "final_position_m": float(info["position_m"]),
                }
            )

        def _evaluate_checkpoint(self, milestone: int) -> None:
            checkpoint = run_directory / f"step-{milestone:06d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.save(checkpoint / "model.zip")
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
            wall_time = time.perf_counter() - started_at - self._evaluation_time_s
            latest = {
                name: value
                for name, value in diagnostics_writer.latest.items()
                if name in DIAGNOSTIC_KEYS
            }
            updates = int(getattr(model, "_n_updates", 0))
            for split_id, episodes, summary in pending:
                filename = (
                    "development-result.json"
                    if split_id.startswith("development")
                    else "validation-result.json"
                )
                save_result(
                    checkpoint / filename,
                    BinaryRunResult(
                        benchmark_name=configuration.name,
                        configuration_sha256=configuration_sha256(configuration),
                        environment_id=configuration.environment_id,
                        evaluation_split_id=split_id,
                        library_version=sb3_version,
                        training_seed=training_seed,
                        training_steps=milestone,
                        training_wall_time_s=wall_time,
                        policy_updates=updates,
                        task=configuration.task,
                        reward=configuration.reward,
                        episodes=episodes,
                        summary=summary,
                        diagnostics=latest,
                    ),
                )
            _write_json(
                run_directory / "training-diagnostics.json",
                diagnostics_writer.history,
            )
            _write_json(
                run_directory / "training-outcomes.json",
                _training_outcome_payload(configuration, training_seed, self._outcomes),
            )

        def _on_step(self) -> bool:
            self._record_finished_episode()
            if not self._remaining or self.num_timesteps < self._remaining[0]:
                return True
            milestone = self._remaining.pop(0)
            if self.num_timesteps != milestone:
                raise RuntimeError(
                    f"Expected exact milestone {milestone}, got {self.num_timesteps}"
                )
            self._evaluate_checkpoint(milestone)
            print(
                f"seed={training_seed} step={milestone} "
                f"training_successes={sum(row['success'] for row in self._outcomes)}",
                flush=True,
            )
            return bool(self._remaining)

    return BinaryMilestoneCallback()


def run_policy(
    configuration,
    *,
    training_seed: int,
    output_directory: str | Path,
    device: str = "auto",
) -> Path:
    from stable_baselines3 import __version__ as sb3_version
    from stable_baselines3.common.logger import Logger

    run_directory = Path(output_directory) / f"training-seed-{training_seed}"
    marker = (
        run_directory
        / f"step-{configuration.total_training_steps:06d}"
        / "validation-result.json"
    )
    if marker.exists() or tuple(run_directory.glob("step-*/*-result.json")):
        raise FileExistsError(
            f"Binary run already exists or is partial: {run_directory}"
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    environment = _training_environment(configuration)
    try:
        model = _model(configuration, "sac", environment, training_seed, device)
        diagnostics_writer = DiagnosticWriter()
        model.set_logger(Logger(folder=None, output_formats=[diagnostics_writer]))
        _write_json(
            run_directory / "run-config.json",
            {
                "benchmark": asdict(configuration),
                "configuration_sha256": configuration_sha256(configuration),
                "algorithm": "Stable-Baselines3 SAC",
                "stable_baselines3_version": sb3_version,
                "training_seed": training_seed,
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
        )
        started_at = time.perf_counter()
        callback = _callback(
            configuration=configuration,
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


def smoke(configuration, output_directory: str | Path, *, device="cpu") -> Path:
    """Exercise real SB3 collection/update and the binary wrapper before training."""

    from stable_baselines3.common.callbacks import BaseCallback

    environment = _training_environment(configuration)
    outcomes = []

    class SmokeCallback(BaseCallback):
        def _on_step(self):
            if bool(np.asarray(self.locals["dones"]).reshape(-1)[0]):
                info = self.locals["infos"][0]
                outcomes.append(bool(info["binary_reward_success"]))
            return True

    try:
        model = _model(configuration, "sac", environment, 11, device)
        model.learn(total_timesteps=2_048, callback=SmokeCallback(), progress_bar=False)
        evaluation = _base_environment(configuration)
        try:
            episodes, summary = evaluate_policy(
                _policy_adapter(model),
                evaluation,
                task=configuration.task,
                evaluation_seeds=(3000,),
            )
        finally:
            evaluation.close()
    finally:
        environment.close()
    destination = Path(output_directory) / "smoke-result.json"
    _write_json(
        destination,
        {
            "training_steps": 2_048,
            "completed_training_episodes": len(outcomes),
            "training_successes": sum(outcomes),
            "evaluation_summary": asdict(summary),
            "evaluation_episode": asdict(episodes[0]),
        },
    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/binary-reward"))
    parser.add_argument("--training-seed", action="append", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    configuration = load_configuration(args.config)
    if args.smoke:
        print(smoke(configuration, args.output_dir, device=args.device))
        return 0
    seeds = tuple(args.training_seed or configuration.training_seeds)
    kwargs = {"output_directory": args.output_dir, "device": args.device}
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
                seed = futures[future]
                print(f"completed seed-{seed}: {future.result()}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
