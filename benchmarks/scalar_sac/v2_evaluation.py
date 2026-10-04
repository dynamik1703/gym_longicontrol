"""Persistence for Scalar Reward V2 evaluations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gym_longicontrol.domain.task import TaskSpecification

from .evaluation import EpisodeEvaluation, EvaluationSummary
from .v2_config import V2RewardParameters

V2_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class V2BenchmarkRunResult:
    benchmark_name: str
    configuration_sha256: str
    environment_id: str
    evaluation_split_id: str
    training_seed: int
    training_steps: int
    task: TaskSpecification
    reward_parameters: V2RewardParameters
    energy_normalization_kwh: float
    speed_violation_normalization_m: float
    episodes: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": V2_RESULT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> V2BenchmarkRunResult:
        if raw.get("schema_version") != V2_RESULT_SCHEMA_VERSION:
            raise ValueError("Unsupported V2 result schema version")
        task = TaskSpecification(**raw["task"])
        episodes = tuple(EpisodeEvaluation(**item) for item in raw["episodes"])
        summary = EvaluationSummary(**raw["summary"])
        if summary != EvaluationSummary.from_episodes(episodes, task):
            raise ValueError("Stored V2 summary does not match episode results")
        return cls(
            benchmark_name=raw["benchmark_name"],
            configuration_sha256=raw["configuration_sha256"],
            environment_id=raw["environment_id"],
            evaluation_split_id=raw["evaluation_split_id"],
            training_seed=raw["training_seed"],
            training_steps=raw["training_steps"],
            task=task,
            reward_parameters=V2RewardParameters(**raw["reward_parameters"]),
            energy_normalization_kwh=raw["energy_normalization_kwh"],
            speed_violation_normalization_m=raw[
                "speed_violation_normalization_m"
            ],
            episodes=episodes,
            summary=summary,
        )


def save_v2_result(path: str | Path, result: V2BenchmarkRunResult) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_v2_result(path: str | Path) -> V2BenchmarkRunResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("V2 result root must be an object")
    return V2BenchmarkRunResult.from_dict(raw)
