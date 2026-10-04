"""Validated episode-level persistence for constrained V2."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.scalar_sac.evaluation import EpisodeEvaluation, EvaluationSummary
from gym_longicontrol.domain.task import TaskSpecification

RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConstrainedV2RunResult:
    benchmark_name: str
    configuration_sha256: str
    environment_id: str
    evaluation_split_id: str
    algorithm: str
    library_version: str
    library_revision: str
    training_seed: int
    simulator_step_target: int
    simulator_steps: int
    gradient_updates: int
    training_wall_time_s: float
    task: TaskSpecification
    objective_name: str
    cost_names: tuple[str, ...]
    cost_limits: tuple[float, ...]
    episodes: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": RESULT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ConstrainedV2RunResult:
        if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("Unsupported constrained V2 result schema")
        task = TaskSpecification(**raw["task"])
        episodes = tuple(EpisodeEvaluation(**item) for item in raw["episodes"])
        summary = EvaluationSummary(**raw["summary"])
        if summary != EvaluationSummary.from_episodes(episodes, task):
            raise ValueError("Stored V2 summary does not match episode metrics")
        cost_names = tuple(raw["cost_names"])
        expected = ("speed_integral_m", "deadline_deficit_integral_s")
        if cost_names != expected:
            raise ValueError("Stored V2 constraint order is not canonical")
        cost_limits = tuple(raw["cost_limits"])
        if cost_limits != (0.0, 0.0):
            raise ValueError("Stored V2 cost limits changed")
        return cls(
            benchmark_name=raw["benchmark_name"],
            configuration_sha256=raw["configuration_sha256"],
            environment_id=raw["environment_id"],
            evaluation_split_id=raw["evaluation_split_id"],
            algorithm=raw["algorithm"],
            library_version=raw["library_version"],
            library_revision=raw["library_revision"],
            training_seed=raw["training_seed"],
            simulator_step_target=raw["simulator_step_target"],
            simulator_steps=raw["simulator_steps"],
            gradient_updates=raw["gradient_updates"],
            training_wall_time_s=raw["training_wall_time_s"],
            task=task,
            objective_name=raw["objective_name"],
            cost_names=cost_names,
            cost_limits=cost_limits,
            episodes=episodes,
            summary=summary,
            diagnostics=raw["diagnostics"],
        )


def save_result(path: str | Path, result: ConstrainedV2RunResult) -> Path:
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


def load_result(path: str | Path) -> ConstrainedV2RunResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Constrained V2 result root must be an object")
    return ConstrainedV2RunResult.from_dict(raw)

