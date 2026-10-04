"""Self-contained episode-level result persistence for the SB3 study."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.scalar_sac.evaluation import EpisodeEvaluation, EvaluationSummary
from benchmarks.scalar_sac.v2_config import V2RewardParameters
from gym_longicontrol.domain.task import TaskSpecification

RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SB3BenchmarkRunResult:
    benchmark_name: str
    configuration_sha256: str
    environment_id: str
    evaluation_split_id: str
    algorithm: str
    library_version: str
    training_seed: int
    training_steps: int
    training_wall_time_s: float
    policy_updates: int
    task: TaskSpecification
    reward_parameters: V2RewardParameters
    episodes: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary
    diagnostics: dict[str, float]

    def __post_init__(self):
        if self.algorithm not in {"sac", "ppo"}:
            raise ValueError("algorithm must be 'sac' or 'ppo'")
        if self.training_steps <= 0 or self.policy_updates < 0:
            raise ValueError("training_steps must be positive and updates nonnegative")
        if self.training_wall_time_s < 0:
            raise ValueError("training_wall_time_s must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": RESULT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SB3BenchmarkRunResult:
        if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("Unsupported SB3 result schema version")
        task = TaskSpecification(**raw["task"])
        episodes = tuple(EpisodeEvaluation(**item) for item in raw["episodes"])
        summary = EvaluationSummary(**raw["summary"])
        if summary != EvaluationSummary.from_episodes(episodes, task):
            raise ValueError("Stored SB3 summary does not match episode results")
        return cls(
            benchmark_name=raw["benchmark_name"],
            configuration_sha256=raw["configuration_sha256"],
            environment_id=raw["environment_id"],
            evaluation_split_id=raw["evaluation_split_id"],
            algorithm=raw["algorithm"],
            library_version=raw["library_version"],
            training_seed=raw["training_seed"],
            training_steps=raw["training_steps"],
            training_wall_time_s=raw["training_wall_time_s"],
            policy_updates=raw["policy_updates"],
            task=task,
            reward_parameters=V2RewardParameters(**raw["reward_parameters"]),
            episodes=episodes,
            summary=summary,
            diagnostics={
                str(name): float(value)
                for name, value in raw.get("diagnostics", {}).items()
            },
        )


def save_result(path: str | Path, result: SB3BenchmarkRunResult) -> Path:
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


def load_result(path: str | Path) -> SB3BenchmarkRunResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("SB3 result root must be an object")
    return SB3BenchmarkRunResult.from_dict(raw)
