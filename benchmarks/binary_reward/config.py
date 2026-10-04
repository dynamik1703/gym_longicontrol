"""Validated preregistration for Binary Success Reward V1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from benchmarks.scalar_sac.v2_config import TrackSplits
from benchmarks.scalar_sb3.config import SACConfiguration
from gym_longicontrol.domain.task import TaskSpecification

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")


@dataclass(frozen=True)
class BinaryRewardDefinition:
    success_reward: float
    otherwise_reward: float
    delivery: str
    success_semantics: str

    def __post_init__(self):
        if self.success_reward != 1.0 or self.otherwise_reward != 0.0:
            raise ValueError("Binary reward must be exactly one for success, else zero")
        if self.delivery != "terminal-outcome-only":
            raise ValueError("Binary reward must be delivered only at episode outcome")
        if self.success_semantics != "gym_longicontrol.domain.task.is_feasible":
            raise ValueError("Binary success must delegate to the external evaluator")


@dataclass(frozen=True)
class AcceptanceCriteria:
    minimum_pooled_validation_rsr: float
    minimum_rsr_per_training_seed: float
    minimum_seed_count_at_half_rsr: int
    minimum_training_successes_per_seed: int
    maximum_peak_to_final_rsr_drop: float

    def __post_init__(self):
        probabilities = (
            self.minimum_pooled_validation_rsr,
            self.minimum_rsr_per_training_seed,
            self.maximum_peak_to_final_rsr_drop,
        )
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("Binary acceptance rates must lie in [0, 1]")
        if self.minimum_seed_count_at_half_rsr != 2:
            raise ValueError("Exactly two seeds must reach at least half RSR")
        if self.minimum_training_successes_per_seed != 1:
            raise ValueError("Every seed must observe at least one training success")


@dataclass(frozen=True)
class BinaryRewardConfiguration:
    schema_version: int
    name: str
    environment_id: str
    max_episode_steps: int
    task: TaskSpecification
    track_splits: TrackSplits
    training_seeds: tuple[int, ...]
    learning_curve_steps: tuple[int, ...]
    reward: BinaryRewardDefinition
    sac: SACConfiguration
    acceptance: AcceptanceCriteria

    def __post_init__(self):
        if self.schema_version != 1 or not self.name or not self.environment_id:
            raise ValueError("Invalid binary benchmark identity")
        if self.environment_id != "StochasticTrack-v1":
            raise ValueError("Binary V1 uses the public stochastic v1 environment")
        if self.max_episode_steps != 1800:
            raise ValueError("Binary V1 retains the 180-second public horizon")
        if self.task != TaskSpecification(140.0, 0.0):
            raise ValueError("Binary V1 uses the fixed strict 140-second task")
        if self.training_seeds != (11, 29, 47):
            raise ValueError("Binary V1 training seeds changed")
        if self.learning_curve_steps != tuple(range(50_000, 300_001, 50_000)):
            raise ValueError("Binary V1 checkpoint schedule changed")
        if self.track_splits.development_calibration != tuple(range(2000, 2009)):
            raise ValueError("Development split changed")
        if self.track_splits.validation != tuple(range(3000, 3009)):
            raise ValueError("Validation split changed")
        if self.track_splits.v1_exploratory_evaluation != tuple(range(1000, 1009)):
            raise ValueError("Historical tracks changed")
        if self.track_splits.paper_final_test_reserved != tuple(range(4000, 4018)):
            raise ValueError("Reserved paper-test split changed")

    @property
    def total_training_steps(self) -> int:
        return self.learning_curve_steps[-1]


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = TrackSplits(
        **{name: tuple(values) for name, values in raw["track_splits"].items()}
    )
    sac = dict(raw["sac"])
    sac["policy_network"] = tuple(sac["policy_network"])
    return BinaryRewardConfiguration(
        schema_version=raw["schema_version"],
        name=raw["name"],
        environment_id=raw["environment_id"],
        max_episode_steps=raw["max_episode_steps"],
        task=TaskSpecification(**raw["task"]),
        track_splits=splits,
        training_seeds=tuple(raw["training_seeds"]),
        learning_curve_steps=tuple(raw["learning_curve_steps"]),
        reward=BinaryRewardDefinition(**raw["reward"]),
        sac=SACConfiguration(**sac),
        acceptance=AcceptanceCriteria(**raw["acceptance"]),
    )


def configuration_sha256(configuration: BinaryRewardConfiguration) -> str:
    payload = json.dumps(asdict(configuration), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()
