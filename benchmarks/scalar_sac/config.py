"""Explicit configuration for the first scalar SAC benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from gym_longicontrol.domain.task import TaskSpecification

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")


def _finite_nonnegative(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed_tuple(name: str, values: Any) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON list")
    seeds = tuple(values)
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
        raise ValueError(f"{name} must contain integers")
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{name} must not contain duplicates")
    return seeds


@dataclass(frozen=True)
class RewardParameters:
    """Dimensionless weights for the documented scalar reward family."""

    configuration_id: str
    progress_weight: float
    energy_weight: float
    time_weight: float
    speed_violation_weight: float

    def __post_init__(self):
        if not self.configuration_id or not self.configuration_id.replace(
            "-", ""
        ).isalnum():
            raise ValueError("configuration_id must be a non-empty slug")
        for name in (
            "progress_weight",
            "energy_weight",
            "time_weight",
            "speed_violation_weight",
        ):
            _finite_nonnegative(name, getattr(self, name))


@dataclass(frozen=True)
class SACTrainingConfiguration:
    """Fixed SAC settings; reward weights are the experimental factors."""

    total_training_steps: int
    initial_random_steps: int
    replay_buffer_capacity: int
    batch_size: int
    hidden_layer_sizes: tuple[int, ...]
    learning_rate: float
    discount_factor: float
    soft_update_factor: float

    def __post_init__(self):
        for name in (
            "total_training_steps",
            "initial_random_steps",
            "replay_buffer_capacity",
            "batch_size",
        ):
            _positive_integer(name, getattr(self, name))
        if self.initial_random_steps > self.replay_buffer_capacity:
            raise ValueError("initial_random_steps exceed replay_buffer_capacity")
        if self.batch_size > self.replay_buffer_capacity:
            raise ValueError("batch_size exceeds replay_buffer_capacity")
        if not self.hidden_layer_sizes or any(
            isinstance(width, bool) or not isinstance(width, Integral) or width <= 0
            for width in self.hidden_layer_sizes
        ):
            raise ValueError("hidden_layer_sizes must contain positive integers")
        _finite_nonnegative("learning_rate", self.learning_rate, positive=True)
        if not 0 <= self.discount_factor <= 1:
            raise ValueError("discount_factor must be in [0, 1]")
        if not 0 <= self.soft_update_factor <= 1:
            raise ValueError("soft_update_factor must be in [0, 1]")


@dataclass(frozen=True)
class ScalarSACBenchmarkConfiguration:
    """The deliberately specific configuration of this benchmark."""

    schema_version: int
    name: str
    environment_id: str
    evaluation_set_id: str
    max_episode_steps: int
    task: TaskSpecification
    time_budget_sweep: tuple[float, ...]
    training_seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    energy_normalization_kwh: float
    speed_violation_normalization_m: float
    reward_grid: tuple[RewardParameters, ...]
    training: SACTrainingConfiguration

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError(
                "Only benchmark configuration schema_version 1 is supported"
            )
        if not self.name or not self.environment_id or not self.evaluation_set_id:
            raise ValueError("Benchmark names and identifiers must be non-empty")
        _positive_integer("max_episode_steps", self.max_episode_steps)
        if not self.time_budget_sweep:
            raise ValueError("time_budget_sweep must not be empty")
        for value in self.time_budget_sweep:
            _finite_nonnegative("time_budget_sweep value", value, positive=True)
        if self.task.max_time_s not in self.time_budget_sweep:
            raise ValueError("Canonical max_time_s must occur in time_budget_sweep")
        seed_sets = (
            set(self.training_seeds),
            set(self.calibration_seeds),
            set(self.evaluation_seeds),
        )
        pairs = ((0, 1), (0, 2), (1, 2))
        if any(seed_sets[left] & seed_sets[right] for left, right in pairs):
            raise ValueError(
                "Training, calibration and evaluation seeds must be disjoint"
            )
        _finite_nonnegative(
            "energy_normalization_kwh", self.energy_normalization_kwh, positive=True
        )
        _finite_nonnegative(
            "speed_violation_normalization_m",
            self.speed_violation_normalization_m,
            positive=True,
        )
        if not self.reward_grid:
            raise ValueError("reward_grid must not be empty")
        identifiers = [parameters.configuration_id for parameters in self.reward_grid]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("reward configuration identifiers must be unique")


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH):
    """Load and validate the canonical, human-readable JSON configuration."""

    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Benchmark configuration root must be an object")
    task = TaskSpecification(**raw["task"])
    seeds = raw["seeds"]
    normalizations = raw["reward_normalization"]
    training_raw = raw["training"]
    training = SACTrainingConfiguration(
        total_training_steps=training_raw["total_training_steps"],
        initial_random_steps=training_raw["initial_random_steps"],
        replay_buffer_capacity=training_raw["replay_buffer_capacity"],
        batch_size=training_raw["batch_size"],
        hidden_layer_sizes=tuple(training_raw["hidden_layer_sizes"]),
        learning_rate=training_raw["learning_rate"],
        discount_factor=training_raw["discount_factor"],
        soft_update_factor=training_raw["soft_update_factor"],
    )
    return ScalarSACBenchmarkConfiguration(
        schema_version=raw["schema_version"],
        name=raw["name"],
        environment_id=raw["environment_id"],
        evaluation_set_id=raw["evaluation_set_id"],
        max_episode_steps=raw["max_episode_steps"],
        task=task,
        time_budget_sweep=tuple(raw["time_budget_sweep"]),
        training_seeds=_seed_tuple("training seeds", seeds["training"]),
        calibration_seeds=_seed_tuple("calibration seeds", seeds["calibration"]),
        evaluation_seeds=_seed_tuple("evaluation seeds", seeds["evaluation"]),
        energy_normalization_kwh=normalizations["energy_kwh"],
        speed_violation_normalization_m=normalizations[
            "integrated_speed_violation_m"
        ],
        reward_grid=tuple(RewardParameters(**item) for item in raw["reward_grid"]),
        training=training,
    )


def configuration_sha256(configuration: ScalarSACBenchmarkConfiguration) -> str:
    """Hash the validated semantic configuration, independent of JSON spacing."""

    payload = json.dumps(
        asdict(configuration), sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(payload).hexdigest()
