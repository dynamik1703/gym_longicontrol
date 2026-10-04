"""Episode-level persistence for the credit-assignment study."""

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
class CreditAssignmentRunResult:
    benchmark_name: str
    configuration_sha256: str
    environment_id: str
    evaluation_split_id: str
    condition_id: str
    condition_label: str
    gamma: float
    action_repeat: int
    library_version: str
    training_seed: int
    simulator_step_target: int
    simulator_steps: int
    agent_decisions: int
    gradient_updates: int
    training_wall_time_s: float
    task: TaskSpecification
    reward_parameters: V2RewardParameters
    episodes: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary
    diagnostics: dict[str, float]

    def __post_init__(self):
        if not self.condition_id or not self.condition_label:
            raise ValueError("Condition identifiers must not be empty")
        if not 0 < self.gamma < 1 or self.action_repeat <= 0:
            raise ValueError("Invalid discount or action repeat")
        if self.simulator_step_target <= 0:
            raise ValueError("simulator_step_target must be positive")
        if self.simulator_steps < self.simulator_step_target:
            raise ValueError("simulator_steps must reach the requested checkpoint")
        if self.agent_decisions <= 0 or self.gradient_updates < 0:
            raise ValueError("Invalid decision or update count")
        if self.training_wall_time_s < 0:
            raise ValueError("training_wall_time_s must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": RESULT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CreditAssignmentRunResult:
        if raw.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("Unsupported credit-assignment result schema version")
        task = TaskSpecification(**raw["task"])
        episodes = tuple(EpisodeEvaluation(**item) for item in raw["episodes"])
        summary = EvaluationSummary(**raw["summary"])
        if summary != EvaluationSummary.from_episodes(episodes, task):
            raise ValueError("Stored summary does not match episode results")
        return cls(
            benchmark_name=raw["benchmark_name"],
            configuration_sha256=raw["configuration_sha256"],
            environment_id=raw["environment_id"],
            evaluation_split_id=raw["evaluation_split_id"],
            condition_id=raw["condition_id"],
            condition_label=raw["condition_label"],
            gamma=raw["gamma"],
            action_repeat=raw["action_repeat"],
            library_version=raw["library_version"],
            training_seed=raw["training_seed"],
            simulator_step_target=raw["simulator_step_target"],
            simulator_steps=raw["simulator_steps"],
            agent_decisions=raw["agent_decisions"],
            gradient_updates=raw["gradient_updates"],
            training_wall_time_s=raw["training_wall_time_s"],
            task=task,
            reward_parameters=V2RewardParameters(**raw["reward_parameters"]),
            episodes=episodes,
            summary=summary,
            diagnostics={
                str(name): float(value)
                for name, value in raw.get("diagnostics", {}).items()
            },
        )


def save_result(
    path: str | Path, result: CreditAssignmentRunResult
) -> Path:
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


def load_result(path: str | Path) -> CreditAssignmentRunResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Credit-assignment result root must be an object")
    return CreditAssignmentRunResult.from_dict(raw)
