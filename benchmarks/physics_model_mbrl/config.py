"""Validated protocol identity for the Physics-only constrained MBRL study."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from benchmarks.model_based_rl.config import load_configuration as load_parent

DEFAULT_CONFIG_PATH = Path(__file__).with_name("canonical.json")
EXPECTED_SOURCE_COMMIT = "e0a6b7a2068eb0e0f9917831ff420fd7bd10aa9a"
EXPECTED_PARENT_CONFIGURATION_SHA256 = (
    "c60560198ef3ba4b9ce4cc6b9e75c30e04760d7d57a324812646e9ea32d66533"
)
PHYSICS_CONDITION = "physics"
TRAINING_SEEDS = (11, 29, 47)
SCIENTIFIC_CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)


@dataclass(frozen=True)
class PhysicsStudyConfiguration:
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        if self.raw["schema_version"] != 1:
            raise ValueError("Unsupported Physics-only study schema")
        if self.raw["name"] != "physics-model-constrained-rl-v1":
            raise ValueError("Physics-only study identity changed")
        if self.raw["source_commit"] != EXPECTED_SOURCE_COMMIT:
            raise ValueError("Physics-only source commit changed")
        if self.raw["condition"] != PHYSICS_CONDITION:
            raise ValueError("Only the Physics condition is authorized")
        if tuple(self.raw["training_seeds"]) != TRAINING_SEEDS:
            raise ValueError("Physics-only training seeds changed")
        if self.raw["real_transitions_per_policy"] != 300_000:
            raise ValueError("Each Physics policy must receive exactly 300k real steps")
        if self.raw["total_real_transitions"] != 900_000:
            raise ValueError("The Physics-only matrix must consume 900k real steps")
        if tuple(self.raw["real_transition_checkpoints"]) != SCIENTIFIC_CHECKPOINTS:
            raise ValueError("Development checkpoint schedule changed")
        if self.raw["recovery_checkpoint_interval"] != 10_000:
            raise ValueError("Recovery checkpoint cadence changed")
        splits = self.raw["track_splits"]
        expected_excluded = tuple(
            (
                *range(1000, 1009),
                *range(2000, 2009),
                *range(3000, 3009),
                *range(4000, 4018),
            )
        )
        if tuple(splits["excluded"]) != expected_excluded:
            raise ValueError("Training exclusions changed")
        if tuple(splits["development"]) != tuple(range(2000, 2009)):
            raise ValueError("Development tracks changed")
        if tuple(splits["validation"]) != tuple(range(3000, 3009)):
            raise ValueError("Validation tracks changed")
        if tuple(splits["paper_test"]) != tuple(range(4000, 4018)):
            raise ValueError("Paper-test tracks changed")
        auth = self.raw["authorization"]
        if auth != {
            "main_training_authorized": True,
            "main_training_enabled": True,
            "validation_authorized": True,
            "paper_test_authorized": False,
            "automatic_fresh_restart_authorized": False,
        }:
            raise ValueError("Physics-only execution authorization changed")
        parent = load_parent()
        if parent.configuration_sha256 != EXPECTED_PARENT_CONFIGURATION_SHA256:
            raise RuntimeError("Frozen parent MBRL configuration changed")
        if self.raw["parent_configuration_sha256"] != parent.configuration_sha256:
            raise RuntimeError(
                "Physics-only study does not reference the frozen parent"
            )
        if parent.raw["conditions"] != ["learned", "physics"]:
            raise RuntimeError("Frozen parent condition matrix changed")
        if parent.raw["training_seeds"] != list(TRAINING_SEEDS):
            raise RuntimeError("Frozen parent seeds changed")
        if tuple(parent.raw["real_transition_checkpoints"]) != SCIENTIFIC_CHECKPOINTS:
            raise RuntimeError("Frozen parent checkpoint schedule changed")
        imagination = parent.imagination
        if (
            imagination.horizon,
            imagination.warmup_real_transitions,
            imagination.refresh_real_transitions,
            imagination.synthetic_transitions_per_refresh,
            imagination.real_batch_size,
            imagination.model_batch_size,
        ) != (1, 10_000, 250, 2_500, 128, 128):
            raise RuntimeError("Frozen Physics imagination schedule changed")
        if parent.v2.algorithm.n_step != 2:
            raise RuntimeError("Frozen real-data n-step setting changed")

    @property
    def configuration_sha256(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return sha256(payload).hexdigest()

    @property
    def parent(self):
        return load_parent()


def load_configuration(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> PhysicsStudyConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Physics-only configuration root must be an object")
    return PhysicsStudyConfiguration(raw)
