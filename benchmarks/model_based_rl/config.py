"""Validated preregistration for the matched constrained MBRL study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from benchmarks.constrained_rl_v2.config import (
    configuration_sha256 as v2_configuration_sha256,
)
from benchmarks.constrained_rl_v2.config import load_configuration as load_v2

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")
MODEL_CONDITIONS = ("learned", "physics")
RESERVED_TRACKS = frozenset(
    (*range(1000, 1009), *range(2000, 2009), *range(3000, 3009), *range(4000, 4018))
)


@dataclass(frozen=True)
class ModelConfig:
    ensemble_size: int
    elite_size: int
    hidden_sizes: tuple[int, ...]
    activation: str
    learning_rate: float
    batch_size: int
    holdout_fraction: float
    maximum_holdout_size: int
    maximum_epochs: int
    early_stop_patience: int
    relative_improvement_threshold: float
    minimum_log_variance: float
    maximum_log_variance: float
    input_fields: tuple[str, ...]
    target_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ensemble_size != 7 or self.elite_size != 5:
            raise ValueError("MBRL V1 freezes a seven-member, five-elite ensemble")
        if self.hidden_sizes != (200, 200, 200, 200) or self.activation != "swish":
            raise ValueError("The source-audited MBPO-style architecture changed")
        if not 0 < self.holdout_fraction < 1 or self.elite_size > self.ensemble_size:
            raise ValueError("Invalid ensemble split")
        if self.minimum_log_variance >= self.maximum_log_variance:
            raise ValueError("Invalid log-variance bounds")


@dataclass(frozen=True)
class ImaginationConfig:
    horizon: int
    warmup_real_transitions: int
    refresh_real_transitions: int
    synthetic_transitions_per_refresh: int
    real_fraction: float
    batch_size: int
    synthetic_replay_capacity: int
    source_states: str
    action_source: str

    def __post_init__(self) -> None:
        if self.horizon != 1:
            raise ValueError("MBRL V1 rollout horizon is exactly one")
        if (
            self.warmup_real_transitions != 10_000
            or self.refresh_real_transitions != 250
        ):
            raise ValueError("The preregistered model cadence changed")
        if self.real_fraction != 0.5 or self.batch_size != 256:
            raise ValueError("The frozen update batch is 128 real plus 128 model")
        if self.source_states != "real_replay_only":
            raise ValueError("Synthetic rollouts may branch only from real replay")

    @property
    def real_batch_size(self) -> int:
        return int(self.batch_size * self.real_fraction)

    @property
    def model_batch_size(self) -> int:
        return self.batch_size - self.real_batch_size


@dataclass(frozen=True)
class MBRLConfiguration:
    raw: dict[str, Any]
    model: ModelConfig
    imagination: ImaginationConfig

    def __post_init__(self) -> None:
        if self.raw["conditions"] != list(MODEL_CONDITIONS):
            raise ValueError("Only learned and physics conditions are authorized")
        if self.raw["training_seeds"] != [11, 29, 47]:
            raise ValueError("Training seeds changed")
        if self.raw["real_transition_checkpoints"][-1] != 300_000:
            raise ValueError("Each policy must receive exactly 300k real transitions")
        auth = self.raw["authorization"]
        if any(auth.values()):
            raise ValueError("Preparation configuration must keep all gates closed")
        if self.raw["track_splits"]["paper_test"] != list(range(4000, 4018)):
            raise ValueError("Paper-test tracks changed")
        if self.raw["preparation"]["maximum_simulator_transitions"] > 5000:
            raise ValueError("Preparation interaction cap exceeded")
        expected_v2 = self.raw["frozen_v2"]["configuration_sha256"]
        if v2_configuration_sha256(load_v2()) != expected_v2:
            raise RuntimeError("Frozen Constrained V2 configuration hash changed")

    @property
    def v2(self):
        return load_v2()

    @property
    def configuration_sha256(self) -> str:
        return sha256(
            json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH) -> MBRLConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    model_raw = dict(raw["model"])
    for key in ("hidden_sizes", "input_fields", "target_fields"):
        model_raw[key] = tuple(model_raw[key])
    return MBRLConfiguration(
        raw=raw,
        model=ModelConfig(**model_raw),
        imagination=ImaginationConfig(**raw["imagination"]),
    )


def training_track_seed(rng) -> int:
    """Draw a reproducible non-reserved uint32 training seed."""

    while True:
        value = int(rng.integers(0, 2**31 - 1))
        if value not in RESERVED_TRACKS:
            return value
