"""Validated preregistration for the scalar credit-assignment study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from benchmarks.scalar_sac.v2_config import (
    BaselineAcceptanceCriteria,
    TrackSplits,
    V2RewardParameters,
)
from benchmarks.scalar_sb3.config import SACConfiguration
from gym_longicontrol.domain.task import TaskSpecification

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite(name: str, value: Any, *, positive: bool = False) -> float:
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
    result = tuple(int(value) for value in raw)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class CreditCondition:
    condition_id: str
    label: str
    action_repeat: int
    gamma: float

    def __post_init__(self):
        if not self.condition_id or not self.label:
            raise ValueError("Condition identifiers and labels must not be empty")
        _positive_integer("action_repeat", self.action_repeat)
        gamma = _finite("gamma", self.gamma)
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1)")


@dataclass(frozen=True)
class CreditAssignmentConfiguration:
    schema_version: int
    name: str
    environment_id: str
    max_episode_steps: int
    simulator_dt_s: float
    task: TaskSpecification
    track_splits: TrackSplits
    training_seeds: tuple[int, ...]
    simulator_step_checkpoints: tuple[int, ...]
    energy_normalization_kwh: float
    speed_violation_normalization_m: float
    reward: V2RewardParameters
    acceptance: BaselineAcceptanceCriteria
    material_mean_rsr_improvement: float
    conditions: tuple[CreditCondition, ...]
    sac: SACConfiguration

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Only credit-assignment schema_version 1 is supported")
        if not self.name or not self.environment_id:
            raise ValueError("Benchmark identifiers must not be empty")
        _positive_integer("max_episode_steps", self.max_episode_steps)
        _finite("simulator_dt_s", self.simulator_dt_s, positive=True)
        if not self.training_seeds or len(self.training_seeds) != len(
            set(self.training_seeds)
        ):
            raise ValueError("training_seeds must be non-empty and unique")
        checkpoints = self.simulator_step_checkpoints
        if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
            raise ValueError("simulator checkpoints must be sorted and unique")
        if any(value <= 0 for value in checkpoints):
            raise ValueError("simulator checkpoints must be positive")
        _finite(
            "energy_normalization_kwh",
            self.energy_normalization_kwh,
            positive=True,
        )
        _finite(
            "speed_violation_normalization_m",
            self.speed_violation_normalization_m,
            positive=True,
        )
        material = _finite(
            "material_mean_rsr_improvement", self.material_mean_rsr_improvement
        )
        if material > 1:
            raise ValueError("material_mean_rsr_improvement must not exceed one")
        if self.reward.configuration_id != "v2b-balanced":
            raise ValueError("The study is frozen to the V2-B Balanced reward")
        expected = (
            ("a-baseline", 1, 0.99),
            ("b-discount-only", 1, 0.999),
            ("c-horizon-only", 5, 0.99),
            ("d-horizon-discount", 5, 0.999),
        )
        observed = tuple(
            (item.condition_id, item.action_repeat, item.gamma)
            for item in self.conditions
        )
        if observed != expected:
            raise ValueError("The preregistered 2x2 condition matrix must not change")
        if self.sac.gamma != 0.99:
            raise ValueError("The common SAC template must retain baseline gamma 0.99")

    @property
    def simulator_step_budget(self) -> int:
        return self.simulator_step_checkpoints[-1]


def load_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> CreditAssignmentConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Credit-assignment configuration root must be an object")
    split_raw = raw["track_splits"]
    normalization = raw["reward_normalization"]
    sac_raw = dict(raw["sac"])
    sac_raw["policy_network"] = tuple(sac_raw["policy_network"])
    return CreditAssignmentConfiguration(
        schema_version=raw["schema_version"],
        name=raw["name"],
        environment_id=raw["environment_id"],
        max_episode_steps=raw["max_episode_steps"],
        simulator_dt_s=raw["simulator_dt_s"],
        task=TaskSpecification(**raw["task"]),
        track_splits=TrackSplits(
            **{name: _seeds(name, values) for name, values in split_raw.items()}
        ),
        training_seeds=_seeds("training_seeds", raw["training_seeds"]),
        simulator_step_checkpoints=tuple(raw["simulator_step_checkpoints"]),
        energy_normalization_kwh=normalization["energy_kwh"],
        speed_violation_normalization_m=normalization[
            "integrated_speed_violation_m"
        ],
        reward=V2RewardParameters(**raw["reward"]),
        acceptance=BaselineAcceptanceCriteria(**raw["acceptance"]),
        material_mean_rsr_improvement=raw["material_mean_rsr_improvement"],
        conditions=tuple(CreditCondition(**item) for item in raw["conditions"]),
        sac=SACConfiguration(**sac_raw),
    )


def configuration_sha256(configuration: CreditAssignmentConfiguration) -> str:
    payload = json.dumps(
        asdict(configuration), sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(payload).hexdigest()
