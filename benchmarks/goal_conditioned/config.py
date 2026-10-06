"""Validated preregistration for the SAC versus SAC+HER comparison."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from benchmarks.scalar_sac.v2_config import TrackSplits
from benchmarks.scalar_sb3.config import SACConfiguration
from gym_longicontrol.domain.task import TaskSpecification

from .goal import GoalScales

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")


@dataclass(frozen=True)
class SparseRewardConfiguration:
    success_reward: float
    otherwise_reward: float
    delivery: str
    energy_in_reward: bool

    def __post_init__(self):
        if self.success_reward != 1.0 or self.otherwise_reward != 0.0:
            raise ValueError("Sparse goal reward must be exactly 1/0")
        if self.delivery != "first-arrival-transition-only":
            raise ValueError("Goal reward must be delivered once on first arrival")
        if self.energy_in_reward:
            raise ValueError("Energy is evaluation-only in this study")


@dataclass(frozen=True)
class DependencyVersions:
    gym_longicontrol: str
    stable_baselines3: str
    gymnasium: str
    torch: str
    numpy: str

    def __post_init__(self):
        expected = ("1.0.0", "2.9.0", "1.3.0", "2.14.0", "2.4.6")
        if tuple(asdict(self).values()) != expected:
            raise ValueError("Recorded benchmark dependency versions changed")


@dataclass(frozen=True)
class ComparisonCondition:
    condition_id: str
    replay_buffer: str
    hindsight_relabeling: bool


@dataclass(frozen=True)
class HERConfiguration:
    n_sampled_goal: int
    goal_selection_strategy: str
    relabelled_fields: tuple[str, ...]
    fixed_fields: tuple[str, ...]
    copy_info_dict: bool
    recompute_terminal_mask: bool

    def __post_init__(self):
        if self.n_sampled_goal != 4:
            raise ValueError("HER ratio is frozen to four virtual goals")
        if self.goal_selection_strategy != "future":
            raise ValueError("HER strategy is frozen to future")
        if self.relabelled_fields != ("target_position",):
            raise ValueError("Only target position may be relabeled")
        if self.fixed_fields != ("deadline", "speed_tolerance"):
            raise ValueError("Deadline and speed tolerance must remain fixed")
        if self.copy_info_dict or not self.recompute_terminal_mask:
            raise ValueError("HER must use stored goals and recompute terminal masks")


@dataclass(frozen=True)
class GoalSACConfiguration:
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq: int
    gradient_steps: int
    ent_coef: str
    policy_network: tuple[int, ...]
    policy: str
    handle_timeout_termination: bool

    def as_standard_sac(self) -> SACConfiguration:
        return SACConfiguration(
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            ent_coef=self.ent_coef,
            policy_network=self.policy_network,
        )

    def __post_init__(self):
        self.as_standard_sac()
        if self.learning_starts != 1800:
            raise ValueError("Both arms must wait for one maximum-length episode")
        if self.policy != "MultiInputPolicy":
            raise ValueError("Dict observations require MultiInputPolicy")
        if self.handle_timeout_termination:
            raise ValueError("The finite 180-second horizon must remain terminal")


@dataclass(frozen=True)
class GoalConditionedConfiguration:
    schema_version: int
    name: str
    dependencies: DependencyVersions
    environment_id: str
    max_episode_steps: int
    task: TaskSpecification
    goal_scales: GoalScales
    track_splits: TrackSplits
    training_seeds: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    development_evaluation_steps: tuple[int, ...]
    validation_evaluation_steps: tuple[int, ...]
    reward: SparseRewardConfiguration
    conditions: tuple[ComparisonCondition, ...]
    her: HERConfiguration
    sac: GoalSACConfiguration

    def __post_init__(self):
        if self.schema_version != 1 or not self.name:
            raise ValueError("Invalid goal-conditioned benchmark identity")
        if self.environment_id != "StochasticTrack-v1":
            raise ValueError("The physical environment must remain StochasticTrack-v1")
        if self.max_episode_steps != 1800:
            raise ValueError("The public 180-second horizon changed")
        if self.task != TaskSpecification(140.0, 0.0):
            raise ValueError("The canonical strict 140-second task changed")
        if self.goal_scales != GoalScales():
            raise ValueError("Goal normalization scales changed")
        if self.training_seeds != (11, 29, 47):
            raise ValueError("Training seeds changed")
        expected_steps = tuple(range(50_000, 300_001, 50_000))
        if self.checkpoint_steps != expected_steps:
            raise ValueError("Checkpoint schedule changed")
        if self.development_evaluation_steps != expected_steps:
            raise ValueError("Development evaluation schedule changed")
        if self.validation_evaluation_steps != (300_000,):
            raise ValueError("Validation may be opened only for final policies")
        expected_conditions = (
            ComparisonCondition(
                "sac-no-her",
                "benchmarks.goal_conditioned.replay_buffer.GoalReplayBuffer",
                False,
            ),
            ComparisonCondition(
                "sac-her",
                "benchmarks.goal_conditioned.replay_buffer."
                "GoalReplayBuffer",
                True,
            ),
        )
        if self.conditions != expected_conditions:
            raise ValueError("Exactly the frozen no-HER and HER arms are required")
        if self.track_splits.development_calibration != tuple(range(2000, 2009)):
            raise ValueError("Development split changed")
        if self.track_splits.validation != tuple(range(3000, 3009)):
            raise ValueError("Validation split changed")
        if self.track_splits.v1_exploratory_evaluation != tuple(range(1000, 1009)):
            raise ValueError("Historical split changed")
        if self.track_splits.paper_final_test_reserved != tuple(range(4000, 4018)):
            raise ValueError("Reserved paper-test split changed")

    @property
    def total_training_steps(self) -> int:
        return self.checkpoint_steps[-1]

    def condition(self, condition_id: str) -> ComparisonCondition:
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                return condition
        raise KeyError(f"Unknown condition: {condition_id}")


def load_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> GoalConditionedConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = TrackSplits(
        **{name: tuple(values) for name, values in raw["track_splits"].items()}
    )
    sac = dict(raw["sac"])
    sac["policy_network"] = tuple(sac["policy_network"])
    her = dict(raw["her"])
    her["relabelled_fields"] = tuple(her["relabelled_fields"])
    her["fixed_fields"] = tuple(her["fixed_fields"])
    return GoalConditionedConfiguration(
        schema_version=raw["schema_version"],
        name=raw["name"],
        dependencies=DependencyVersions(**raw["dependencies"]),
        environment_id=raw["environment_id"],
        max_episode_steps=raw["max_episode_steps"],
        task=TaskSpecification(**raw["task"]),
        goal_scales=GoalScales(**raw["goal_scales"]),
        track_splits=splits,
        training_seeds=tuple(raw["training_seeds"]),
        checkpoint_steps=tuple(raw["checkpoint_steps"]),
        development_evaluation_steps=tuple(raw["development_evaluation_steps"]),
        validation_evaluation_steps=tuple(raw["validation_evaluation_steps"]),
        reward=SparseRewardConfiguration(**raw["reward"]),
        conditions=tuple(
            ComparisonCondition(**condition) for condition in raw["conditions"]
        ),
        her=HERConfiguration(**her),
        sac=GoalSACConfiguration(**sac),
    )


def configuration_sha256(configuration: GoalConditionedConfiguration) -> str:
    payload = json.dumps(asdict(configuration), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()
