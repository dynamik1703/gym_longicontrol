"""Frozen scientific configuration and fail-closed authorization gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "canonical.json"
TRAINING_SEEDS = (11, 29, 47)
CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
PAPER_SEEDS = tuple(range(4000, 4018))
CANONICAL_GOAL = (1.0, 1.0, 1.0)
NATIVE_DT_S = 0.1


@dataclass(frozen=True)
class GCSLConfig:
    state_dim: int = 12
    goal_dim: int = 3
    action_dim: int = 1
    width: int = 256
    depth: int = 4
    learning_rate: float = 5e-4
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_epsilon: float = 1e-6
    future_discount: float = 0.99
    batch_size: int = 256
    replay_capacity: int = 300_000
    horizon_conditioning: bool = False

    def __post_init__(self) -> None:
        if self.depth <= 0 or self.depth % 4:
            raise ValueError("depth must be a positive multiple of four")
        if not 0.0 < self.future_discount < 1.0:
            raise ValueError("future_discount must lie strictly between zero and one")
        if self.horizon_conditioning:
            raise ValueError("Canonical GCSL freezes horizon conditioning off")


@dataclass(frozen=True)
class StudyConfiguration:
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        raw = self.raw
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported canonical GCSL configuration")
        if tuple(raw.get("training_seeds", ())) != TRAINING_SEEDS:
            raise ValueError("Training seeds changed")
        if tuple(raw.get("diagnostic_checkpoints", ())) != CHECKPOINTS:
            raise ValueError("Scientific checkpoints changed")
        if raw.get("collector") != "TRANSITIONAL":
            raise ValueError("Collector convention changed")
        if raw.get("real_transition_budget") != 300_000:
            raise ValueError("Transition budget changed")
        if raw.get("minimum_physically_possible_episode_length") != 347:
            raise ValueError("Physical episode lower bound changed")
        if raw.get("maximum_exposed_tasks") != 865:
            raise ValueError("Maximum exposed-task bound changed")
        budget = raw.get("update_budget", {})
        expected_budget = {
            "selected": "A_CRL_MATCHED",
            "prefill_transitions": 10_000,
            "update_interval_transitions": 40,
            "updates_per_policy": 7_250,
            "sampled_hindsight_tuples_per_policy": 1_856_000,
            "rejected_candidate_B_updates_per_policy": 30_000,
        }
        if budget != expected_budget:
            raise ValueError("Frozen supervised update budget changed")
        auth = raw.get("authorization", {})
        if set(auth) != {
            "main_training_authorized",
            "main_training_enabled",
            "same_attempt_exact_resume_authorized",
            "fresh_restart_authorized",
            "validation_authorized",
            "paper_test_authorized",
        } or any(not isinstance(value, bool) for value in auth.values()):
            raise ValueError("Authorization flags must be complete booleans")

    @property
    def algorithm(self) -> GCSLConfig:
        return GCSLConfig()

    @property
    def configuration_sha256(self) -> str:
        scientific = {
            key: value for key, value in self.raw.items() if key != "authorization"
        }
        encoded = json.dumps(
            scientific, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def updates_per_policy(self) -> int:
        return int(self.raw["update_budget"]["updates_per_policy"])


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH) -> StudyConfiguration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be an object")
    return StudyConfiguration(payload)


def should_update(transition_count: int) -> bool:
    raw = load_configuration().raw
    prefill = int(raw["update_budget"]["prefill_transitions"])
    interval = int(raw["update_budget"]["update_interval_transitions"])
    return bool(
        transition_count > prefill
        and (transition_count - prefill) % interval == 0
        and transition_count <= int(raw["real_transition_budget"])
    )


def scheduled_update_transitions() -> tuple[int, ...]:
    raw = load_configuration().raw
    prefill = int(raw["update_budget"]["prefill_transitions"])
    interval = int(raw["update_budget"]["update_interval_transitions"])
    total = int(raw["real_transition_budget"])
    return tuple(range(prefill + interval, total + 1, interval))


def require_main_training_authorized(configuration: StudyConfiguration) -> None:
    auth = configuration.raw["authorization"]
    if not (auth["main_training_authorized"] and auth["main_training_enabled"]):
        raise PermissionError("MAIN_TRAINING_NOT_AUTHORIZED")


def require_validation_authorized(configuration: StudyConfiguration) -> None:
    if not configuration.raw["authorization"]["validation_authorized"]:
        raise PermissionError("GCSL_VALIDATION_V1_FROZEN_UNOPENED")


def forbid_paper(*_args, **_kwargs):
    raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
