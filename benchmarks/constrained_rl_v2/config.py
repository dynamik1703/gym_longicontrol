"""Validated preregistration for the dense-deadline constrained study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from benchmarks.constrained_rl.config import (
    ObjectiveConfiguration,
    SACLagrangianConfiguration,
)
from benchmarks.scalar_sac.v2_config import TrackSplits
from gym_longicontrol.domain.task import TaskSpecification

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if result < 0 or (positive and result == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _probability(name: str, value: Any) -> float:
    result = _finite(name, value)
    if result > 1:
        raise ValueError(f"{name} must not exceed one")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


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
class ConstraintConfiguration:
    names: tuple[str, ...]
    cost_limits: tuple[float, ...]

    def __post_init__(self):
        expected = ("speed_integral_m", "deadline_deficit_integral_s")
        if self.names != expected:
            raise ValueError(f"Constraint order must be {expected}")
        if self.cost_limits != (0.0, 0.0):
            raise ValueError("Both canonical V2 cost limits must be exactly zero")


@dataclass(frozen=True)
class DeadlineCostConfiguration:
    name: str
    normalization_s: float
    uses_terminal_failure_cost: bool

    def __post_init__(self):
        if self.name != "optimistic_speed_limit_deadline_deficit_integral":
            raise ValueError("The preregistered V2 deadline formulation changed")
        _finite("normalization_s", self.normalization_s, positive=True)
        if self.uses_terminal_failure_cost:
            raise ValueError("V2 must not restore the V1 binary terminal task cost")


@dataclass(frozen=True)
class AcceptanceCriteria:
    minimum_material_pooled_rsr: float
    minimum_material_pooled_completion_rate: float
    minimum_improved_training_seeds: int
    maximum_material_standstill_rate: float
    minimum_credible_speed_compliance_rate: float
    maximum_peak_to_final_rsr_drop: float
    strong_minimum_rsr_per_training_seed: float
    strong_minimum_completion_rate_per_training_seed: float
    strong_minimum_speed_compliance_rate_per_training_seed: float
    strong_maximum_rsr_range: float
    strong_maximum_mean_energy_ratio_to_fast_reference: float
    standstill_dominance_rate: float
    instability_multiplier_threshold: float
    instability_critic_loss_threshold: float
    minimum_unstable_training_seeds: int

    def __post_init__(self):
        for name in (
            "minimum_material_pooled_rsr",
            "minimum_material_pooled_completion_rate",
            "maximum_material_standstill_rate",
            "minimum_credible_speed_compliance_rate",
            "maximum_peak_to_final_rsr_drop",
            "strong_minimum_rsr_per_training_seed",
            "strong_minimum_completion_rate_per_training_seed",
            "strong_minimum_speed_compliance_rate_per_training_seed",
            "strong_maximum_rsr_range",
            "standstill_dominance_rate",
        ):
            _probability(name, getattr(self, name))
        _positive_integer(
            "minimum_improved_training_seeds", self.minimum_improved_training_seeds
        )
        _positive_integer(
            "minimum_unstable_training_seeds", self.minimum_unstable_training_seeds
        )
        for name in (
            "strong_maximum_mean_energy_ratio_to_fast_reference",
            "instability_multiplier_threshold",
            "instability_critic_loss_threshold",
        ):
            _finite(name, getattr(self, name), positive=True)


@dataclass(frozen=True)
class ConstrainedRLV2Configuration:
    schema_version: int
    name: str
    environment_id: str
    max_episode_steps: int
    task: TaskSpecification
    track_splits: TrackSplits
    training_seeds: tuple[int, ...]
    simulator_step_checkpoints: tuple[int, ...]
    objective: ObjectiveConfiguration
    constraints: ConstraintConfiguration
    deadline_cost: DeadlineCostConfiguration
    algorithm: SACLagrangianConfiguration
    acceptance: AcceptanceCriteria

    def __post_init__(self):
        if self.schema_version != 1 or not self.name or not self.environment_id:
            raise ValueError("Invalid constrained V2 benchmark identity")
        _positive_integer("max_episode_steps", self.max_episode_steps)
        if self.training_seeds != (11, 29, 47):
            raise ValueError("The V2 training seeds are frozen to 11, 29, and 47")
        checkpoints = self.simulator_step_checkpoints
        if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError("simulator_step_checkpoints must be sorted and unique")
        smoke = self.name.endswith("-smoke")
        if not smoke and checkpoints[-1] != 300_000:
            raise ValueError("The V2 simulator budget must be exactly 300,000")
        if smoke and checkpoints != (2_048,):
            raise ValueError("The only permitted smoke budget is 2,048 transitions")
        if self.algorithm.action_repeat != 1:
            raise ValueError("V2 uses native 10-Hz actions")
        if self.deadline_cost.normalization_s != self.task.max_time_s:
            raise ValueError("Deadline normalization must equal the physical deadline")
        expected_development = (2000,) if smoke else tuple(range(2000, 2009))
        expected_validation = (3000,) if smoke else tuple(range(3000, 3009))
        if self.track_splits.development_calibration != expected_development:
            raise ValueError("Development tracks changed")
        if self.track_splits.validation != expected_validation:
            raise ValueError("Validation tracks changed")
        if self.track_splits.paper_final_test_reserved != tuple(range(4000, 4018)):
            raise ValueError("Sealed paper tracks changed")

    @property
    def simulator_step_budget(self) -> int:
        return self.simulator_step_checkpoints[-1]


def load_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> ConstrainedRLV2Configuration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Constrained V2 configuration root must be an object")
    splits = TrackSplits(
        **{
            name: _seeds(name, values)
            for name, values in raw["track_splits"].items()
        }
    )
    algorithm = dict(raw["algorithm"])
    for name in ("hidden_sizes", "lagrangian_pid", "initial_multipliers"):
        algorithm[name] = tuple(algorithm[name])
    constraints = dict(raw["constraints"])
    constraints["names"] = tuple(constraints["names"])
    constraints["cost_limits"] = tuple(constraints["cost_limits"])
    return ConstrainedRLV2Configuration(
        schema_version=raw["schema_version"],
        name=raw["name"],
        environment_id=raw["environment_id"],
        max_episode_steps=raw["max_episode_steps"],
        task=TaskSpecification(**raw["task"]),
        track_splits=splits,
        training_seeds=_seeds("training_seeds", raw["training_seeds"]),
        simulator_step_checkpoints=tuple(raw["simulator_step_checkpoints"]),
        objective=ObjectiveConfiguration(**raw["objective"]),
        constraints=ConstraintConfiguration(**constraints),
        deadline_cost=DeadlineCostConfiguration(**raw["deadline_cost"]),
        algorithm=SACLagrangianConfiguration(**algorithm),
        acceptance=AcceptanceCriteria(**raw["acceptance"]),
    )


def configuration_sha256(configuration: ConstrainedRLV2Configuration) -> str:
    payload = json.dumps(
        asdict(configuration), sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(payload).hexdigest()
