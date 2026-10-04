"""Reproducible 50k SB3-SAC screening on Development tracks only."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import numpy as np

from benchmarks.scalar_sac.evaluation import evaluate_policy
from benchmarks.scalar_sac.experiment import _base_environment
from benchmarks.scalar_sb3.experiment import _model, _policy_adapter
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import is_feasible

from .candidate_validation import load_reward_function
from .history import (
    DEFAULT_HISTORY_PATH,
    candidate_by_id,
    load_history,
    save_history,
)
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_sha256
from .reward_api import CandidateRewardWrapper


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


@dataclass
class RunningStatistics:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def observe(self, value: float) -> None:
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("Screening diagnostic must be finite")
        self.count += 1
        self.total += numeric
        self.total_square += numeric * numeric
        self.minimum = min(self.minimum, numeric)
        self.maximum = max(self.maximum, numeric)

    def summary(self) -> dict[str, float | int]:
        if not self.count:
            raise ValueError("Cannot summarize an empty statistic")
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": variance**0.5,
            "min": self.minimum,
            "max": self.maximum,
        }


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": fmean(numeric),
        "std": pstdev(numeric),
        "min": min(numeric),
        "max": max(numeric),
    }


def _failure_mode(metrics: Mapping[str, Any], task) -> str:
    failures = []
    if not metrics["completed"]:
        failures.append("incomplete")
    if metrics["travel_time_s"] > task.max_time_s:
        failures.append("time")
    if metrics["max_speed_violation_m_s"] > task.max_speed_violation_m_s:
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def _training_environment(protocol, candidate):
    function, report = load_reward_function(candidate["source_path"])
    if report.source_sha256 != candidate["source_sha256"]:
        raise ValueError("Candidate source changed after ingestion")
    return CandidateRewardWrapper(
        _base_environment(protocol),
        compute_reward=function,
        task=protocol.task,
        candidate_id=candidate["candidate_id"],
        source_sha256=candidate["source_sha256"],
    )


def _screening_callback(protocol, model, wrapped_environment, candidate, started_at):
    from stable_baselines3.common.callbacks import BaseCallback

    class ScreeningCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.remaining = list(protocol.screening.evaluation_checkpoints)
            self.checkpoints: list[dict[str, Any]] = []
            self.training_outcomes: list[dict[str, Any]] = []
            self.component_stats: dict[str, RunningStatistics] = {}
            self.episode_rewards: list[float] = []
            self.episode_lengths: list[int] = []
            self._episode_reward = 0.0
            self._episode_length = 0
            self.evaluation_time_s = 0.0

        def _record_step(self) -> None:
            info = self.locals["infos"][0]
            reward = float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
            self._episode_reward += reward
            self._episode_length += 1
            for name, value in info["llm_reward_components"].items():
                self.component_stats.setdefault(name, RunningStatistics()).observe(
                    value
                )
            if not bool(np.asarray(self.locals["dones"]).reshape(-1)[0]):
                return
            metrics = info["episode_metrics"]
            success = is_feasible(EpisodeMetrics(**metrics), protocol.task)
            self.training_outcomes.append(
                {
                    "training_step": int(self.num_timesteps),
                    "success": success,
                    "failure_mode": _failure_mode(metrics, protocol.task),
                    "episode_reward": self._episode_reward,
                    "episode_length": self._episode_length,
                    "completed": bool(metrics["completed"]),
                    "travel_time_s": float(metrics["travel_time_s"]),
                    "energy_kwh": float(metrics["energy_kwh"]),
                    "max_speed_violation_m_s": float(
                        metrics["max_speed_violation_m_s"]
                    ),
                    "integrated_speed_violation_m": float(
                        metrics["integrated_speed_violation_m"]
                    ),
                    "final_position_m": float(info["position_m"]),
                }
            )
            self.episode_rewards.append(self._episode_reward)
            self.episode_lengths.append(self._episode_length)
            self._episode_reward = 0.0
            self._episode_length = 0

        def _evaluate(self, checkpoint: int) -> None:
            evaluation_started = time.perf_counter()
            environment = _base_environment(protocol)
            try:
                episodes, summary = evaluate_policy(
                    _policy_adapter(model),
                    environment,
                    task=protocol.task,
                    evaluation_seeds=protocol.screening.development_tracks,
                )
            finally:
                environment.close()
            self.evaluation_time_s += time.perf_counter() - evaluation_started
            self.checkpoints.append(
                {
                    "simulator_transitions": checkpoint,
                    "evaluation_split_id": protocol.screening.development_split_id,
                    "episodes": [asdict(item) for item in episodes],
                    "summary": asdict(summary),
                    "cumulative_training_successes": sum(
                        item["success"] for item in self.training_outcomes
                    ),
                }
            )

        def _on_step(self) -> bool:
            self._record_step()
            if not self.remaining or self.num_timesteps < self.remaining[0]:
                return True
            checkpoint = self.remaining.pop(0)
            if self.num_timesteps != checkpoint:
                raise RuntimeError("Screening missed an exact checkpoint")
            self._evaluate(checkpoint)
            return bool(self.remaining)

        def result(self) -> dict[str, Any]:
            import stable_baselines3

            import gym_longicontrol

            if self.remaining:
                raise RuntimeError("Screening ended before every checkpoint")
            return {
                "schema_version": 1,
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol_sha256(protocol),
                "candidate_id": candidate["candidate_id"],
                "candidate_source_sha256": candidate["source_sha256"],
                "generation": candidate["generation"],
                "environment_id": protocol.environment_id,
                "max_episode_steps": protocol.max_episode_steps,
                "task": asdict(protocol.task),
                "sac_configuration": asdict(protocol.sac),
                "training_seed": protocol.screening.training_seed,
                "simulator_transition_budget": (
                    protocol.screening.simulator_transitions_per_candidate
                ),
                "simulator_transitions_completed": int(self.num_timesteps),
                "gradient_updates": int(getattr(model, "_n_updates", 0)),
                "training_wall_time_s": (
                    time.perf_counter() - started_at - self.evaluation_time_s
                ),
                "runtime": {
                    "python": platform.python_version(),
                    "device": str(model.device),
                    "gym_longicontrol_version": gym_longicontrol.__version__,
                    "stable_baselines3_version": stable_baselines3.__version__,
                },
                "evaluation_scope": "development-only",
                "route_length_m": float(
                    wrapped_environment.unwrapped.config.track_length_m
                ),
                "checkpoints": self.checkpoints,
                "training_outcomes": self.training_outcomes,
                "episode_reward_statistics": _distribution(self.episode_rewards),
                "episode_length_statistics": _distribution(self.episode_lengths),
                "component_statistics": {
                    name: statistic.summary()
                    for name, statistic in sorted(self.component_stats.items())
                },
            }

    return ScreeningCallback()


def screen_candidate(
    candidate_id: str,
    *,
    protocol,
    history_path: str | Path,
    output_root: str | Path,
    device: str = "auto",
) -> Path:
    from stable_baselines3.common.logger import Logger

    history = load_history(history_path, protocol)
    if history["status"] != "OPEN":
        raise RuntimeError("Screening requires an OPEN search")
    candidate = candidate_by_id(history, candidate_id)
    if candidate["status"] != "validated" or candidate["screening"] is not None:
        raise RuntimeError("Candidate must be validated and unscreened")
    output = Path(output_root) / candidate_id
    if output.exists():
        raise FileExistsError(f"Screening output already exists: {output}")
    output.mkdir(parents=True)
    environment = _training_environment(protocol, candidate)
    started_at = time.perf_counter()
    try:
        model = _model(
            protocol,
            "sac",
            environment,
            protocol.screening.training_seed,
            device,
        )
        model.set_logger(Logger(folder=None, output_formats=[]))
        callback = _screening_callback(
            protocol, model, environment, candidate, started_at
        )
        model.learn(
            total_timesteps=protocol.screening.simulator_transitions_per_candidate,
            callback=callback,
            progress_bar=False,
        )
        result = callback.result()
        model.save(output / "screening-model.zip")
    finally:
        environment.close()
    result_path = _write_json(output / "screening-result.json", result)
    updated = deepcopy(history)
    record = candidate_by_id(updated, candidate_id)
    record["screening"] = {
        "result_path": str(result_path),
        "simulator_transitions": result["simulator_transitions_completed"],
        "gradient_updates": result["gradient_updates"],
    }
    record["status"] = "screened"
    updated["engineering_effort"]["screening_rl_transitions"] += result[
        "simulator_transitions_completed"
    ]
    save_history(history_path, updated, protocol)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_id")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args(argv)
    print(
        screen_candidate(
            args.candidate_id,
            protocol=load_protocol(args.protocol),
            history_path=args.history,
            output_root=args.output_root,
            device=args.device,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
