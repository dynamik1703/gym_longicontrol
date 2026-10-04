"""Validated episode-level result persistence for constrained SAC-Lagrangian."""

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
class ConstrainedRunResult:
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
    def from_dict(cls, raw: Mapping[str, Any]) -> ConstrainedRunResult:
        if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("Unsupported constrained result schema version")
        task = TaskSpecification(**raw["task"])
        episodes = tuple(EpisodeEvaluation(**item) for item in raw["episodes"])
        summary = EvaluationSummary(**raw["summary"])
        if summary != EvaluationSummary.from_episodes(episodes, task):
            raise ValueError("Stored summary does not match episode-level metrics")
        cost_names = tuple(raw["cost_names"])
        cost_limits = tuple(raw["cost_limits"])
        if cost_names != ("speed_integral_m", "task_failure"):
            raise ValueError("Stored constraint order is not canonical")
        if len(cost_names) != len(cost_limits):
            raise ValueError("Stored costs and limits do not align")
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


def save_result(path: str | Path, result: ConstrainedRunResult) -> Path:
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


def load_result(path: str | Path) -> ConstrainedRunResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Constrained result root must be an object")
    return ConstrainedRunResult.from_dict(raw)
