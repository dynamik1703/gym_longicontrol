"""Validated configuration for Scalar Reward Study V2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from gym_longicontrol.domain.task import TaskSpecification

from .config import SACTrainingConfiguration

DEFAULT_V2_CONFIG_PATH = Path(__file__).with_name("canonical_v2.json")


def _finite_nonnegative(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _seeds(name: str, raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
        raise ValueError(f"{name} must contain integers")
    values = tuple(int(value) for value in raw)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


@dataclass(frozen=True)
class V2RewardParameters:
    configuration_id: str
    progress_weight: float
    energy_weight: float
    time_weight: float
    on_time_completion_bonus: float
    speed_integral_weight: float
    speed_violation_event_penalty: float
    design_intent: str

    def __post_init__(self):
        if not self.configuration_id or not self.configuration_id.replace(
            "-", ""
        ).isalnum():
            raise ValueError("configuration_id must be a non-empty slug")
        if not self.design_intent:
            raise ValueError("design_intent must not be empty")
        for name in (
            "progress_weight",
            "energy_weight",
            "time_weight",
            "on_time_completion_bonus",
            "speed_integral_weight",
            "speed_violation_event_penalty",
        ):
            _finite_nonnegative(name, getattr(self, name))


@dataclass(frozen=True)
class TrackSplits:
    development_calibration: tuple[int, ...]
    validation: tuple[int, ...]
    v1_exploratory_evaluation: tuple[int, ...]
    paper_final_test_reserved: tuple[int, ...]

    def __post_init__(self):
        groups = tuple(asdict(self).items())
        for name, values in groups:
            if not values:
                raise ValueError(f"Track split {name} must not be empty")
        for left_index, (left_name, left) in enumerate(groups):
            for right_name, right in groups[left_index + 1 :]:
                if set(left) & set(right):
                    raise ValueError(
                        f"Track splits {left_name} and {right_name} overlap"
                    )


@dataclass(frozen=True)
class BaselineAcceptanceCriteria:
    minimum_rsr_per_training_seed: float
    minimum_completion_rate_per_training_seed: float
    minimum_speed_compliance_rate_per_training_seed: float
    maximum_rsr_range_across_training_seeds: float
    maximum_mean_energy_ratio_to_fast_reference: float

    def __post_init__(self):
        for name in (
            "minimum_rsr_per_training_seed",
            "minimum_completion_rate_per_training_seed",
            "minimum_speed_compliance_rate_per_training_seed",
            "maximum_rsr_range_across_training_seeds",
        ):
            value = _finite_nonnegative(name, getattr(self, name))
            if value > 1:
                raise ValueError(f"{name} must not exceed one")
        _finite_nonnegative(
            "maximum_mean_energy_ratio_to_fast_reference",
            self.maximum_mean_energy_ratio_to_fast_reference,
            positive=True,
        )


@dataclass(frozen=True)
class ScalarSACV2Configuration:
    schema_version: int
    name: str
    environment_id: str
    max_episode_steps: int
    task: TaskSpecification
    track_splits: TrackSplits
    training_seeds: tuple[int, ...]
    energy_normalization_kwh: float
    speed_violation_normalization_m: float
    reward_candidates: tuple[V2RewardParameters, ...]
    learning_curve_steps: tuple[int, ...]
    comparison_steps: tuple[int, ...]
    acceptance: BaselineAcceptanceCriteria
    training: SACTrainingConfiguration

    def __post_init__(self):
        if self.schema_version != 2:
            raise ValueError("Only V2 configuration schema_version 2 is supported")
        if not self.name or not self.environment_id:
            raise ValueError("Benchmark identifiers must not be empty")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if not self.training_seeds:
            raise ValueError("training_seeds must not be empty")
        if len(self.training_seeds) != len(set(self.training_seeds)):
            raise ValueError("training_seeds must not contain duplicates")
        _finite_nonnegative(
            "energy_normalization_kwh", self.energy_normalization_kwh, positive=True
        )
        _finite_nonnegative(
            "speed_violation_normalization_m",
            self.speed_violation_normalization_m,
            positive=True,
        )
        if not 3 <= len(self.reward_candidates) <= 6:
            raise ValueError("V2 requires three to six reward candidates")
        identifiers = [item.configuration_id for item in self.reward_candidates]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Reward candidate identifiers must be unique")
        if (
            not self.learning_curve_steps
            or tuple(sorted(set(self.learning_curve_steps)))
            != self.learning_curve_steps
            or self.learning_curve_steps[-1] != self.training.total_training_steps
        ):
            raise ValueError("learning_curve_steps must be sorted and end at total")
        if not set(self.comparison_steps) <= set(self.learning_curve_steps):
            raise ValueError("comparison_steps must occur in learning_curve_steps")


def load_v2_configuration(
    path: str | Path = DEFAULT_V2_CONFIG_PATH,
) -> ScalarSACV2Configuration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    split_raw = raw["track_splits"]
    splits = TrackSplits(
        **{name: _seeds(name, values) for name, values in split_raw.items()}
    )
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
    normalization = raw["reward_normalization"]
    return ScalarSACV2Configuration(
        schema_version=raw["schema_version"],
        name=raw["name"],
        environment_id=raw["environment_id"],
        max_episode_steps=raw["max_episode_steps"],
        task=TaskSpecification(**raw["task"]),
        track_splits=splits,
        training_seeds=_seeds("training_seeds", raw["training_seeds"]),
        energy_normalization_kwh=normalization["energy_kwh"],
        speed_violation_normalization_m=normalization[
            "integrated_speed_violation_m"
        ],
        reward_candidates=tuple(
            V2RewardParameters(**item) for item in raw["reward_candidates"]
        ),
        learning_curve_steps=tuple(raw["learning_curve_steps"]),
        comparison_steps=tuple(raw["comparison_steps"]),
        acceptance=BaselineAcceptanceCriteria(**raw["acceptance"]),
        training=training,
    )


def v2_configuration_sha256(configuration: ScalarSACV2Configuration) -> str:
    payload = json.dumps(
        asdict(configuration), sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(payload).hexdigest()
