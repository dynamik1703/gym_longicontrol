"""Frozen configuration and fail-closed authorization checks."""

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


@dataclass(frozen=True)
class StudyConfiguration:
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        raw = self.raw
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported same-information configuration")
        if tuple(raw.get("training_seeds", ())) != TRAINING_SEEDS:
            raise ValueError("Training seeds changed")
        if tuple(raw.get("diagnostic_checkpoints", ())) != CHECKPOINTS:
            raise ValueError("Scientific checkpoints changed")
        if raw.get("collector") != "TRANSITIONAL":
            raise ValueError("Collector convention changed")
        if raw.get("real_transition_budget") != 300_000:
            raise ValueError("Transition budget changed")
        if raw.get("gradient_updates_per_policy") != 30_000:
            raise ValueError("Update budget changed")
        expected = {
            "gamma": 0.99,
            "n_step": 2,
            "batch_size": 256,
            "replay_capacity": 100_000,
            "hidden_sizes": [128, 128],
            "actor_learning_rate": 0.0005,
            "critic_learning_rate": 0.001,
            "automatic_alpha": True,
            "initial_alpha": 1.0,
            "target_entropy": -1.0,
            "alpha_learning_rate": 0.0003,
            "tau": 0.05,
            "action_repeat": 1,
            "native_dt_s": 0.1,
            "max_episode_steps": 1800,
        }
        if raw.get("algorithm") != expected:
            raise ValueError("Matched algorithm settings changed")
        scalar = raw.get("scalarization", {})
        expected_scalar = {
            "selected_candidate": "U",
            "convention": "unnormalized-fixed-lagrangian",
            "lambda_speed_star": 1.4009421299572349,
            "lambda_deadline_star": 1.5752363435900516,
            "constrained_actor_normalizer": 3.9761784735472867,
            "w_reward": 1.0,
            "w_speed": 1.4009421299572349,
            "w_deadline": 1.5752363435900516,
        }
        if any(scalar.get(name) != value for name, value in expected_scalar.items()):
            raise ValueError("Frozen Candidate U scalarization changed")
        auth = raw.get("authorization", {})
        if any(not isinstance(value, bool) for value in auth.values()):
            raise ValueError("Authorization flags must be explicit booleans")

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
    def constrained_v2(self):
        from benchmarks.constrained_rl_v2.config import load_configuration

        return load_configuration()


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH) -> StudyConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be an object")
    return StudyConfiguration(raw)


def require_main_training_authorized(configuration: StudyConfiguration) -> None:
    auth = configuration.raw["authorization"]
    if not (auth["main_training_authorized"] and auth["main_training_enabled"]):
        raise PermissionError("MAIN_TRAINING_NOT_AUTHORIZED")


def require_validation_authorized(configuration: StudyConfiguration) -> None:
    if not configuration.raw["authorization"]["validation_authorized"]:
        raise PermissionError("SAMEINFO_VALIDATION_V1_SEALED")


def forbid_paper(*_args, **_kwargs):
    raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
