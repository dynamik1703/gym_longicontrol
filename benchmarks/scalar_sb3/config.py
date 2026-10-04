"""Validated, immutable protocol for the SB3 scalar baseline study."""

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
from gym_longicontrol.domain.task import TaskSpecification

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite(name: str, value: Any, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
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


def _network(name: str, raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(_positive_integer(f"{name} width", value) for value in raw)


@dataclass(frozen=True)
class SACConfiguration:
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

    def __post_init__(self):
        _finite("SAC learning_rate", self.learning_rate, minimum=1e-15)
        for name in (
            "buffer_size",
            "learning_starts",
            "batch_size",
            "train_freq",
            "gradient_steps",
        ):
            _positive_integer(f"SAC {name}", getattr(self, name))
        if self.learning_starts >= self.buffer_size:
            raise ValueError("SAC learning_starts must be below buffer_size")
        if self.batch_size > self.buffer_size:
            raise ValueError("SAC batch_size must not exceed buffer_size")
        if not 0 < self.tau <= 1:
            raise ValueError("SAC tau must be in (0, 1]")
        if not 0 <= self.gamma <= 1:
            raise ValueError("SAC gamma must be in [0, 1]")
        if self.ent_coef != "auto":
            raise ValueError("The frozen SAC study requires learned entropy ('auto')")
        if not self.policy_network:
            raise ValueError("SAC policy_network must not be empty")


@dataclass(frozen=True)
class PPOConfiguration:
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    policy_network: tuple[int, ...]

    def __post_init__(self):
        _finite("PPO learning_rate", self.learning_rate, minimum=1e-15)
        for name in ("n_steps", "batch_size", "n_epochs"):
            _positive_integer(f"PPO {name}", getattr(self, name))
        if self.batch_size > self.n_steps:
            raise ValueError("PPO batch_size must not exceed n_steps for one env")
        for name in ("gamma", "gae_lambda"):
            value = _finite(f"PPO {name}", getattr(self, name))
            if value > 1:
                raise ValueError(f"PPO {name} must not exceed one")
        _finite("PPO clip_range", self.clip_range, minimum=1e-15)
        _finite("PPO ent_coef", self.ent_coef)
        _finite("PPO vf_coef", self.vf_coef)
        if not self.policy_network:
            raise ValueError("PPO policy_network must not be empty")


@dataclass(frozen=True)
class ScalarSB3Configuration:
    schema_version: int
    name: str
    environment_id: str
    max_episode_steps: int
    task: TaskSpecification
    track_splits: TrackSplits
    training_seeds: tuple[int, ...]
    learning_curve_steps: tuple[int, ...]
    energy_normalization_kwh: float
    speed_violation_normalization_m: float
    reward: V2RewardParameters
    acceptance: BaselineAcceptanceCriteria
    sac: SACConfiguration
    ppo: PPOConfiguration

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Only SB3 benchmark schema_version 1 is supported")
        if not self.name or not self.environment_id:
            raise ValueError("Benchmark identifiers must not be empty")
        _positive_integer("max_episode_steps", self.max_episode_steps)
        if not self.training_seeds or len(self.training_seeds) != len(
            set(self.training_seeds)
        ):
            raise ValueError("training_seeds must be non-empty and unique")
        if (
            not self.learning_curve_steps
            or tuple(sorted(set(self.learning_curve_steps)))
            != self.learning_curve_steps
        ):
            raise ValueError("learning_curve_steps must be sorted and unique")
        if self.learning_curve_steps[-1] <= 0:
            raise ValueError("The final learning-curve step must be positive")
        _finite(
            "energy_normalization_kwh",
            self.energy_normalization_kwh,
            minimum=1e-15,
        )
        _finite(
            "speed_violation_normalization_m",
            self.speed_violation_normalization_m,
            minimum=1e-15,
        )
        if self.reward.configuration_id != "v2b-balanced":
            raise ValueError("The SB3 study is frozen to V2-B Balanced")

    @property
    def total_training_steps(self) -> int:
        return self.learning_curve_steps[-1]

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        """Compatibility view used by the shared reference-controller code."""

        return self.track_splits.validation

    @property
    def evaluation_set_id(self) -> str:
        return "validation-3000-3008-v1"


def load_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> ScalarSB3Configuration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("SB3 configuration root must be an object")
    split_raw = raw["track_splits"]
    normalization = raw["reward_normalization"]
    sac_raw = dict(raw["sac"])
    ppo_raw = dict(raw["ppo"])
    sac_raw["policy_network"] = _network(
        "SAC policy_network", sac_raw["policy_network"]
    )
    ppo_raw["policy_network"] = _network(
        "PPO policy_network", ppo_raw["policy_network"]
    )
    return ScalarSB3Configuration(
        schema_version=raw["schema_version"],
        name=raw["name"],
        environment_id=raw["environment_id"],
        max_episode_steps=raw["max_episode_steps"],
        task=TaskSpecification(**raw["task"]),
        track_splits=TrackSplits(
            **{name: _seeds(name, values) for name, values in split_raw.items()}
        ),
        training_seeds=_seeds("training_seeds", raw["training_seeds"]),
        learning_curve_steps=tuple(raw["learning_curve_steps"]),
        energy_normalization_kwh=normalization["energy_kwh"],
        speed_violation_normalization_m=normalization[
            "integrated_speed_violation_m"
        ],
        reward=V2RewardParameters(**raw["reward"]),
        acceptance=BaselineAcceptanceCriteria(**raw["acceptance"]),
        sac=SACConfiguration(**sac_raw),
        ppo=PPOConfiguration(**ppo_raw),
    )


def configuration_sha256(configuration: ScalarSB3Configuration) -> str:
    payload = json.dumps(
        asdict(configuration), sort_keys=True, separators=(",", ":")
    ).encode()
    return sha256(payload).hexdigest()
