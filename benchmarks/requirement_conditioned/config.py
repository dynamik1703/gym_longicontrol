"""Validated preregistration for Requirement-Conditioned RL V1."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

from benchmarks.constrained_rl.config import (
    ObjectiveConfiguration,
    SACLagrangianConfiguration,
)
from benchmarks.scalar_sac.v2_config import TrackSplits

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


@dataclass(frozen=True)
class RequirementConfiguration:
    training_margins_s: tuple[float, ...]
    interpolation_margins_s: tuple[float, ...]
    time_scale_s: float
    maximum_t_min_start_s: float
    sampling: str

    def __post_init__(self):
        if self.training_margins_s != (20.0, 40.0, 60.0):
            raise ValueError("Training margins are frozen to 20/40/60 seconds")
        if self.interpolation_margins_s != (30.0, 50.0):
            raise ValueError("Interpolation margins are frozen to 30/50 seconds")
        _finite("time_scale_s", self.time_scale_s, positive=True)
        _finite("maximum_t_min_start_s", self.maximum_t_min_start_s, positive=True)
        if self.time_scale_s != 180.0 or self.maximum_t_min_start_s != 180.0:
            raise ValueError(
                "Requirement normalization must use the 180-second horizon"
            )
        if self.sampling != "independent-shuffled-balanced-blocks":
            raise ValueError("Requirement sampling protocol changed")

    @property
    def evaluation_margins_s(self) -> tuple[float, ...]:
        return tuple(sorted(self.training_margins_s + self.interpolation_margins_s))

    @property
    def maximum_deadline_s(self) -> float:
        return self.maximum_t_min_start_s + max(self.training_margins_s)


@dataclass(frozen=True)
class ConstraintConfiguration:
    names: tuple[str, ...]
    cost_limits: tuple[float, ...]

    def __post_init__(self):
        if self.names != ("speed_integral_m", "deadline_deficit_integral_s"):
            raise ValueError("Constraint order changed from Constrained V2")
        if self.cost_limits != (0.0, 0.0):
            raise ValueError("Both constraints retain exact zero limits")


@dataclass(frozen=True)
class DeadlineCostConfiguration:
    name: str
    uses_terminal_failure_cost: bool

    def __post_init__(self):
        if self.name != "episode_requirement_optimistic_deadline_deficit_integral":
            raise ValueError("Deadline cost formulation changed")
        if self.uses_terminal_failure_cost:
            raise ValueError("The frozen V2 terminal failure cost must remain absent")


@dataclass(frozen=True)
class AcceptanceCriteria:
    minimum_overall_rsr: float
    minimum_rsr_per_requirement: float
    minimum_overall_rsr_per_training_seed: float
    travel_time_monotonicity_tolerance_s: float
    energy_monotonicity_tolerance_kwh: float
    minimum_travel_time_pairwise_monotonicity: float
    minimum_energy_pairwise_monotonicity: float
    minimum_requirement_sensitive_track_seed_rate: float
    requirement_sensitive_travel_time_range_s: float
    requirement_sensitive_energy_range_kwh: float
    requirement_sensitive_mean_action_difference: float
    maximum_interpolation_rsr_gap: float
    minimum_interpolation_directional_rate: float

    def __post_init__(self):
        probabilities = (
            "minimum_overall_rsr",
            "minimum_rsr_per_requirement",
            "minimum_overall_rsr_per_training_seed",
            "minimum_travel_time_pairwise_monotonicity",
            "minimum_energy_pairwise_monotonicity",
            "minimum_requirement_sensitive_track_seed_rate",
            "maximum_interpolation_rsr_gap",
            "minimum_interpolation_directional_rate",
        )
        for name in probabilities:
            value = _finite(name, getattr(self, name))
            if value > 1:
                raise ValueError(f"{name} must not exceed one")
        for name in (
            "travel_time_monotonicity_tolerance_s",
            "energy_monotonicity_tolerance_kwh",
            "requirement_sensitive_travel_time_range_s",
            "requirement_sensitive_energy_range_kwh",
            "requirement_sensitive_mean_action_difference",
        ):
            _finite(name, getattr(self, name), positive=True)


@dataclass(frozen=True)
class RequirementConditionedConfiguration:
    schema_version: int
    name: str
    environment_id: str
    max_episode_steps: int
    max_speed_violation_m_s: float
    track_splits: TrackSplits
    training_seeds: tuple[int, ...]
    simulator_step_checkpoints: tuple[int, ...]
    requirements: RequirementConfiguration
    objective: ObjectiveConfiguration
    constraints: ConstraintConfiguration
    deadline_cost: DeadlineCostConfiguration
    algorithm: SACLagrangianConfiguration
    acceptance: AcceptanceCriteria

    def __post_init__(self):
        if self.schema_version != 1 or not self.name or not self.environment_id:
            raise ValueError("Invalid benchmark identity")
        if self.max_episode_steps != 1800 or self.max_speed_violation_m_s != 0.0:
            raise ValueError("Public horizon and strict speed requirement changed")
        if self.training_seeds != (11, 29, 47):
            raise ValueError("Training seeds changed")
        if self.simulator_step_checkpoints != (
            50_000,
            100_000,
            150_000,
            200_000,
            250_000,
            300_000,
        ):
            raise ValueError("Training budget/checkpoints changed")
        if self.track_splits.development_calibration != tuple(range(2000, 2009)):
            raise ValueError("Development split changed")
        if self.track_splits.validation != tuple(range(3000, 3009)):
            raise ValueError("Validation split changed")
        if self.track_splits.paper_final_test_reserved != tuple(range(4000, 4018)):
            raise ValueError("Reserved paper split changed")
        if self.algorithm.action_repeat != 1:
            raise ValueError("Native 10-Hz actions are required")

    @property
    def simulator_step_budget(self) -> int:
        return self.simulator_step_checkpoints[-1]


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = TrackSplits(
        **{key: tuple(value) for key, value in raw["track_splits"].items()}
    )
    requirements = dict(raw["requirements"])
    requirements["training_margins_s"] = tuple(requirements["training_margins_s"])
    requirements["interpolation_margins_s"] = tuple(
        requirements["interpolation_margins_s"]
    )
    constraints = dict(raw["constraints"])
    constraints["names"] = tuple(constraints["names"])
    constraints["cost_limits"] = tuple(constraints["cost_limits"])
    algorithm = dict(raw["algorithm"])
    for name in ("hidden_sizes", "lagrangian_pid", "initial_multipliers"):
        algorithm[name] = tuple(algorithm[name])
    return RequirementConditionedConfiguration(
        schema_version=raw["schema_version"],
        name=raw["name"],
        environment_id=raw["environment_id"],
        max_episode_steps=raw["max_episode_steps"],
        max_speed_violation_m_s=raw["max_speed_violation_m_s"],
        track_splits=splits,
        training_seeds=tuple(raw["training_seeds"]),
        simulator_step_checkpoints=tuple(raw["simulator_step_checkpoints"]),
        requirements=RequirementConfiguration(**requirements),
        objective=ObjectiveConfiguration(**raw["objective"]),
        constraints=ConstraintConfiguration(**constraints),
        deadline_cost=DeadlineCostConfiguration(**raw["deadline_cost"]),
        algorithm=SACLagrangianConfiguration(**algorithm),
        acceptance=AcceptanceCriteria(**raw["acceptance"]),
    )


def configuration_sha256(configuration: RequirementConditionedConfiguration) -> str:
    payload = json.dumps(asdict(configuration), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()
