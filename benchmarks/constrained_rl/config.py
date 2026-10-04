"""Validated preregistration for the explicit constrained-RL study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from benchmarks.scalar_sac.v2_config import TrackSplits
from gym_longicontrol.domain.task import TaskSpecification

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")


def _finite(name: str, value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
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
class ObjectiveConfiguration:
    name: str
    energy_scale_kwh: float

    def __post_init__(self):
        if self.name != "negative_net_energy":
            raise ValueError("The study is frozen to negative net energy")
        _finite("energy_scale_kwh", self.energy_scale_kwh, minimum=1e-15)


@dataclass(frozen=True)
class ConstraintConfiguration:
    names: tuple[str, ...]
    cost_limits: tuple[float, ...]

    def __post_init__(self):
        if self.names != ("speed_integral_m", "task_failure"):
            raise ValueError("Constraint order must remain speed then task failure")
        if len(self.cost_limits) != len(self.names):
            raise ValueError("Each constraint requires one cost limit")
        for index, value in enumerate(self.cost_limits):
            _finite(f"cost_limits[{index}]", value)
        if any(self.cost_limits):
            raise ValueError("The canonical study permits no relaxed cost limit")


@dataclass(frozen=True)
class SACLagrangianConfiguration:
    name: str
    fsrl_git_commit: str
    tianshou_version: str
    actor_learning_rate: float
    critic_learning_rate: float
    hidden_sizes: tuple[int, ...]
    automatic_entropy_tuning: bool
    alpha_learning_rate: float
    initial_effective_alpha: float
    tau: float
    n_step: int
    gamma: float
    lagrangian_pid: tuple[float, float, float]
    lagrangian_rescaling: bool
    initial_multipliers: tuple[float, ...]
    deterministic_evaluation: bool
    buffer_size: int
    batch_size: int
    update_per_step: float
    episodes_per_collect: int
    action_repeat: int

    def __post_init__(self):
        if self.name != "FSRL-SACLag" or len(self.fsrl_git_commit) != 40:
            raise ValueError("A pinned FSRL-SACLag revision is required")
        if not self.tianshou_version:
            raise ValueError("tianshou_version must not be empty")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "alpha_learning_rate",
            "initial_effective_alpha",
            "update_per_step",
        ):
            _finite(name, getattr(self, name), minimum=1e-15)
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        for width in self.hidden_sizes:
            _positive_integer("hidden layer width", width)
        for name in ("n_step", "buffer_size", "batch_size"):
            _positive_integer(name, getattr(self, name))
        if self.batch_size > self.buffer_size:
            raise ValueError("batch_size must not exceed buffer_size")
        if not 0 < self.tau <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("tau and gamma are outside their valid ranges")
        if len(self.lagrangian_pid) != 3:
            raise ValueError("lagrangian_pid must contain Kp, Ki, and Kd")
        for index, value in enumerate(self.lagrangian_pid):
            _finite(f"lagrangian_pid[{index}]", value)
        if self.initial_multipliers != (0.0, 0.0):
            raise ValueError("The frozen study starts both multipliers at zero")
        if not self.automatic_entropy_tuning or not self.deterministic_evaluation:
            raise ValueError("Frozen entropy and evaluation settings changed")
        if self.episodes_per_collect != 1 or self.action_repeat != 1:
            raise ValueError("The first constrained study uses one native episode")


@dataclass(frozen=True)
class AcceptanceCriteria:
    minimum_rsr_per_training_seed: float
    minimum_completion_rate_per_training_seed: float
    minimum_speed_compliance_rate_per_training_seed: float
    maximum_rsr_range_across_training_seeds: float
    maximum_peak_to_final_rsr_drop: float
    maximum_mean_energy_ratio_to_fast_reference: float
    frozen_scalar_sac_mean_rsr: float
    minimum_material_mean_rsr_gain: float
    minimum_improved_training_seeds: int
    standstill_maximum_completion_rate: float
    standstill_minimum_speed_compliance_rate: float
    instability_multiplier_threshold: float
    instability_critic_loss_threshold: float
    minimum_unstable_training_seeds: int

    def __post_init__(self):
        probabilities = (
            "minimum_rsr_per_training_seed",
            "minimum_completion_rate_per_training_seed",
            "minimum_speed_compliance_rate_per_training_seed",
            "maximum_rsr_range_across_training_seeds",
            "maximum_peak_to_final_rsr_drop",
            "frozen_scalar_sac_mean_rsr",
            "minimum_material_mean_rsr_gain",
            "standstill_maximum_completion_rate",
            "standstill_minimum_speed_compliance_rate",
        )
        for name in probabilities:
            if _finite(name, getattr(self, name)) > 1:
                raise ValueError(f"{name} must not exceed one")
        for name in (
            "maximum_mean_energy_ratio_to_fast_reference",
            "instability_multiplier_threshold",
            "instability_critic_loss_threshold",
        ):
            _finite(name, getattr(self, name), minimum=1e-15)
        _positive_integer(
            "minimum_improved_training_seeds", self.minimum_improved_training_seeds
        )
        _positive_integer(
            "minimum_unstable_training_seeds", self.minimum_unstable_training_seeds
        )


@dataclass(frozen=True)
class ConstrainedRLConfiguration:
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
    algorithm: SACLagrangianConfiguration
    acceptance: AcceptanceCriteria

    def __post_init__(self):
        if self.schema_version != 1 or not self.name or not self.environment_id:
            raise ValueError("Invalid constrained benchmark identity")
        _positive_integer("max_episode_steps", self.max_episode_steps)
        if not self.training_seeds or len(self.training_seeds) != len(
            set(self.training_seeds)
        ):
            raise ValueError("training_seeds must be non-empty and unique")
        checkpoints = self.simulator_step_checkpoints
        if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError("simulator_step_checkpoints must be sorted and unique")
        _positive_integer("final simulator step checkpoint", checkpoints[-1])
        if set(self.training_seeds) & set(self.track_splits.validation):
            raise ValueError("Training and validation seeds must be disjoint")

    @property
    def simulator_step_budget(self) -> int:
        return self.simulator_step_checkpoints[-1]


def load_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> ConstrainedRLConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Constrained configuration root must be an object")
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
    return ConstrainedRLConfiguration(
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
        algorithm=SACLagrangianConfiguration(**algorithm),
        acceptance=AcceptanceCriteria(**raw["acceptance"]),
    )


def configuration_sha256(configuration: ConstrainedRLConfiguration) -> str:
    payload = json.dumps(
        asdict(configuration), sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(payload).hexdigest()
