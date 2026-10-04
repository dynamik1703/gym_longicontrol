"""Algorithm-independent episode evaluation and result persistence."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .config import RewardParameters

Policy = Callable[[np.ndarray], np.ndarray]
RESULT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class EpisodeEvaluation:
    evaluation_seed: int
    completed: bool
    feasible: bool
    travel_time_s: float
    energy_kwh: float
    speed_violation_count: int
    max_speed_violation_m_s: float
    integrated_speed_violation_m: float
    step_count: int = 0
    final_position_m: float = 0.0
    traction_energy_kwh: float = 0.0
    regenerative_energy_kwh: float = 0.0
    mean_abs_jerk_m_s3: float = 0.0
    max_abs_jerk_m_s3: float = 0.0
    mean_abs_acceleration_m_s2: float = 0.0
    mean_abs_action: float = 0.0
    action_total_variation: float = 0.0
    mean_abs_action_change: float = 0.0
    acceleration_sign_change_count: int = 0

    @classmethod
    def from_metrics(
        cls, evaluation_seed: int, metrics: EpisodeMetrics, task: TaskSpecification
    ) -> EpisodeEvaluation:
        return cls(
            evaluation_seed=int(evaluation_seed),
            feasible=is_feasible(metrics, task),
            **asdict(metrics),
        )


@dataclass
class EpisodeRecorder:
    """Accumulate benchmark-only diagnostics without changing feasibility."""

    step_count: int = 0
    final_position_m: float = 0.0
    traction_energy_kwh: float = 0.0
    regenerative_energy_kwh: float = 0.0
    absolute_jerk_sum: float = 0.0
    max_abs_jerk_m_s3: float = 0.0
    absolute_acceleration_sum: float = 0.0
    absolute_action_sum: float = 0.0
    action_total_variation: float = 0.0
    acceleration_sign_change_count: int = 0
    _previous_action: float | None = None
    _previous_acceleration_sign: int = 0

    def observe(self, action: Any, info: Mapping[str, Any]) -> None:
        action_values = np.asarray(action, dtype=np.float64)
        if action_values.shape != (1,) or not np.isfinite(action_values).all():
            raise ValueError("Recorded action must be a finite one-element array")
        action_value = float(action_values[0])
        repeated = info.get("action_repeat_diagnostics")
        simulator_steps = int(info.get("simulator_steps_this_decision", 1))
        if simulator_steps <= 0:
            raise ValueError("Recorded simulator-step count must be positive")
        energy = float(info["step_energy_kwh"])
        jerk = abs(float(info["jerk_m_s3"]))
        acceleration = float(info["acceleration_m_s2"])
        acceleration_sign = (
            1 if acceleration > 1e-9 else -1 if acceleration < -1e-9 else 0
        )

        self.step_count += simulator_steps
        self.final_position_m = float(info["position_m"])
        if isinstance(repeated, Mapping):
            self.traction_energy_kwh += float(repeated["traction_energy_kwh"])
            self.regenerative_energy_kwh += float(
                repeated["regenerative_energy_kwh"]
            )
            self.absolute_jerk_sum += float(repeated["absolute_jerk_sum"])
            self.max_abs_jerk_m_s3 = max(
                self.max_abs_jerk_m_s3,
                float(repeated["max_abs_jerk_m_s3"]),
            )
            self.absolute_acceleration_sum += float(
                repeated["absolute_acceleration_sum"]
            )
            first_sign = int(repeated["first_nonzero_acceleration_sign"])
            last_sign = int(repeated["last_nonzero_acceleration_sign"])
            if (
                first_sign
                and self._previous_acceleration_sign
                and first_sign != self._previous_acceleration_sign
            ):
                self.acceleration_sign_change_count += 1
            self.acceleration_sign_change_count += int(
                repeated["acceleration_sign_change_count"]
            )
            if last_sign:
                self._previous_acceleration_sign = last_sign
        else:
            self.traction_energy_kwh += max(energy, 0.0)
            self.regenerative_energy_kwh += max(-energy, 0.0)
            self.absolute_jerk_sum += jerk
            self.max_abs_jerk_m_s3 = max(self.max_abs_jerk_m_s3, jerk)
            self.absolute_acceleration_sum += abs(acceleration)
            if (
                acceleration_sign
                and self._previous_acceleration_sign
                and acceleration_sign != self._previous_acceleration_sign
            ):
                self.acceleration_sign_change_count += 1
            if acceleration_sign:
                self._previous_acceleration_sign = acceleration_sign
        self.absolute_action_sum += abs(action_value) * simulator_steps
        if self._previous_action is not None:
            self.action_total_variation += abs(action_value - self._previous_action)
        self._previous_action = action_value

    def finalize(
        self,
        evaluation_seed: int,
        metrics: EpisodeMetrics,
        task: TaskSpecification,
    ) -> EpisodeEvaluation:
        transitions = max(self.step_count - 1, 1)
        return EpisodeEvaluation(
            evaluation_seed=int(evaluation_seed),
            feasible=is_feasible(metrics, task),
            **asdict(metrics),
            step_count=self.step_count,
            final_position_m=self.final_position_m,
            traction_energy_kwh=self.traction_energy_kwh,
            regenerative_energy_kwh=self.regenerative_energy_kwh,
            mean_abs_jerk_m_s3=(
                self.absolute_jerk_sum / self.step_count if self.step_count else 0.0
            ),
            max_abs_jerk_m_s3=self.max_abs_jerk_m_s3,
            mean_abs_acceleration_m_s2=(
                self.absolute_acceleration_sum / self.step_count
                if self.step_count
                else 0.0
            ),
            mean_abs_action=(
                self.absolute_action_sum / self.step_count if self.step_count else 0.0
            ),
            action_total_variation=self.action_total_variation,
            mean_abs_action_change=(
                self.action_total_variation / transitions
                if self.step_count > 1
                else 0.0
            ),
            acceleration_sign_change_count=self.acceleration_sign_change_count,
        )


@dataclass(frozen=True)
class EvaluationSummary:
    episode_count: int
    completion_rate: float
    requirement_satisfaction_rate: float
    mean_feasible_energy_kwh: float | None
    median_feasible_energy_kwh: float | None
    mean_travel_time_s: float
    median_travel_time_s: float
    incomplete_rate: float
    time_violation_rate: float
    speed_violation_rate: float
    speed_compliance_rate: float
    mean_speed_violation_count: float
    mean_max_speed_violation_m_s: float
    mean_integrated_speed_violation_m: float
    mean_traction_energy_kwh: float
    mean_regenerative_energy_kwh: float
    mean_abs_jerk_m_s3: float
    mean_max_abs_jerk_m_s3: float
    mean_action_total_variation: float

    @classmethod
    def from_episodes(
        cls,
        episodes: Iterable[EpisodeEvaluation],
        task: TaskSpecification,
    ) -> EvaluationSummary:
        values = tuple(episodes)
        if not values:
            raise ValueError("At least one episode is required for aggregation")
        count = len(values)
        feasible_energy = [item.energy_kwh for item in values if item.feasible]
        return cls(
            episode_count=count,
            completion_rate=sum(item.completed for item in values) / count,
            requirement_satisfaction_rate=sum(item.feasible for item in values)
            / count,
            mean_feasible_energy_kwh=(
                fmean(feasible_energy) if feasible_energy else None
            ),
            median_feasible_energy_kwh=(
                float(median(feasible_energy)) if feasible_energy else None
            ),
            mean_travel_time_s=fmean(item.travel_time_s for item in values),
            median_travel_time_s=float(
                median(item.travel_time_s for item in values)
            ),
            incomplete_rate=sum(not item.completed for item in values) / count,
            time_violation_rate=sum(
                item.travel_time_s > task.max_time_s for item in values
            )
            / count,
            speed_violation_rate=sum(
                item.max_speed_violation_m_s > task.max_speed_violation_m_s
                for item in values
            )
            / count,
            speed_compliance_rate=sum(
                item.max_speed_violation_m_s <= task.max_speed_violation_m_s
                for item in values
            )
            / count,
            mean_speed_violation_count=fmean(
                item.speed_violation_count for item in values
            ),
            mean_max_speed_violation_m_s=fmean(
                item.max_speed_violation_m_s for item in values
            ),
            mean_integrated_speed_violation_m=fmean(
                item.integrated_speed_violation_m for item in values
            ),
            mean_traction_energy_kwh=fmean(
                item.traction_energy_kwh for item in values
            ),
            mean_regenerative_energy_kwh=fmean(
                item.regenerative_energy_kwh for item in values
            ),
            mean_abs_jerk_m_s3=fmean(item.mean_abs_jerk_m_s3 for item in values),
            mean_max_abs_jerk_m_s3=fmean(
                item.max_abs_jerk_m_s3 for item in values
            ),
            mean_action_total_variation=fmean(
                item.action_total_variation for item in values
            ),
        )


@dataclass(frozen=True)
class BenchmarkRunResult:
    benchmark_name: str
    configuration_sha256: str
    environment_id: str
    evaluation_set_id: str
    training_seed: int
    training_steps: int
    task: TaskSpecification
    reward_parameters: RewardParameters
    energy_normalization_kwh: float
    speed_violation_normalization_m: float
    episodes: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": RESULT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BenchmarkRunResult:
        if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("Unsupported benchmark result schema version")
        episodes = tuple(EpisodeEvaluation(**item) for item in raw["episodes"])
        task = TaskSpecification(**raw["task"])
        summary = EvaluationSummary(**raw["summary"])
        if summary != EvaluationSummary.from_episodes(episodes, task):
            raise ValueError("Stored summary does not match raw episode results")
        return cls(
            benchmark_name=raw["benchmark_name"],
            configuration_sha256=raw["configuration_sha256"],
            environment_id=raw["environment_id"],
            evaluation_set_id=raw["evaluation_set_id"],
            training_seed=raw["training_seed"],
            training_steps=raw["training_steps"],
            task=task,
            reward_parameters=RewardParameters(**raw["reward_parameters"]),
            energy_normalization_kwh=raw["energy_normalization_kwh"],
            speed_violation_normalization_m=raw[
                "speed_violation_normalization_m"
            ],
            episodes=episodes,
            summary=summary,
        )


def evaluate_policy(
    policy: Policy,
    environment: Any,
    *,
    task: TaskSpecification,
    evaluation_seeds: Iterable[int],
) -> tuple[tuple[EpisodeEvaluation, ...], EvaluationSummary]:
    """Evaluate a policy solely from final physical episode metrics.

    The environment reward is deliberately ignored. Each seed resets both the
    stochastic track and the episode metrics before the deterministic rollout.
    """

    seeds = tuple(int(seed) for seed in evaluation_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation_seeds must be non-empty and unique")
    results = []
    for seed in seeds:
        observation, _ = environment.reset(seed=seed)
        recorder = EpisodeRecorder()
        while True:
            action = policy(np.asarray(observation, dtype=np.float32))
            observation, _reward, terminated, truncated, info = environment.step(
                action
            )
            recorder.observe(action, info)
            if terminated or truncated:
                metrics = EpisodeMetrics(**info["episode_metrics"])
                results.append(recorder.finalize(seed, metrics, task))
                break
    episodes = tuple(results)
    return episodes, EvaluationSummary.from_episodes(episodes, task)


def save_run_result(path: str | Path, result: BenchmarkRunResult) -> Path:
    """Write one self-contained result atomically as deterministic JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(result.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_run_result(path: str | Path) -> BenchmarkRunResult:
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("Benchmark result root must be an object")
    return BenchmarkRunResult.from_dict(raw)
