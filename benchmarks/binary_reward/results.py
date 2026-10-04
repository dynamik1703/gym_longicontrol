"""Episode-level result persistence for Binary Success Reward V1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.scalar_sac.evaluation import EpisodeEvaluation, EvaluationSummary
from gym_longicontrol.domain.task import TaskSpecification

from .config import BinaryRewardDefinition


@dataclass(frozen=True)
class BinaryRunResult:
    benchmark_name: str
    configuration_sha256: str
    environment_id: str
    evaluation_split_id: str
    library_version: str
    training_seed: int
    training_steps: int
    training_wall_time_s: float
    policy_updates: int
    task: TaskSpecification
    reward: BinaryRewardDefinition
    episodes: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary
    diagnostics: dict[str, float]

    def __post_init__(self):
        if self.training_steps <= 0 or self.policy_updates < 0:
            raise ValueError("Training steps must be positive and updates nonnegative")
        if self.training_wall_time_s < 0:
            raise ValueError("Training wall time must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}


def save_result(path: str | Path, result: BinaryRunResult) -> Path:
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


def load_result(path: str | Path) -> BinaryRunResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.pop("schema_version", None) != 1:
        raise ValueError("Unsupported binary result schema")
    raw["task"] = TaskSpecification(**raw["task"])
    raw["reward"] = BinaryRewardDefinition(**raw["reward"])
    raw["episodes"] = tuple(EpisodeEvaluation(**item) for item in raw["episodes"])
    raw["summary"] = EvaluationSummary(**raw["summary"])
    result = BinaryRunResult(**raw)
    if result.summary != EvaluationSummary.from_episodes(result.episodes, result.task):
        raise ValueError("Stored binary summary does not match episodes")
    return result
