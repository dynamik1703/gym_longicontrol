"""Frozen preregistration for the three-seed collector replication."""

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

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "canonical.json"
SOURCE_COMMIT = "7d4b63aa021a8694bf51cca2ed319d55045bb7f8"
TRAINING_SEEDS = (11, 29, 47)
COLLECTORS = ("EPISODIC", "TRANSITIONAL")
CONDITION_NAMES = tuple(
    f"{collector[0]}{seed}_{collector.lower()}_seed{seed}"
    for seed in TRAINING_SEEDS
    for collector in COLLECTORS
)
CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
CANONICAL_NAMESPACE = 0x43414E4F


@dataclass(frozen=True)
class Condition:
    id: str
    name: str
    training_seed: int
    collector: str


@dataclass(frozen=True)
class StudyConfiguration:
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        raw = self.raw
        if raw.get("schema_version") != 1 or raw.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("Collector-replication identity or lineage changed")
        conditions = tuple(Condition(**row) for row in raw.get("conditions", ()))
        if tuple(row.name for row in conditions) != CONDITION_NAMES:
            raise ValueError("Paired E/T condition order changed")
        if tuple(row.id for row in conditions) != (
            "E11",
            "T11",
            "E29",
            "T29",
            "E47",
            "T47",
        ):
            raise ValueError("Condition IDs changed")
        if tuple(row.training_seed for row in conditions) != (11, 11, 29, 29, 47, 47):
            raise ValueError("Training-seed pairing changed")
        if tuple(row.collector for row in conditions) != (
            "EPISODIC",
            "TRANSITIONAL",
            "EPISODIC",
            "TRANSITIONAL",
            "EPISODIC",
            "TRANSITIONAL",
        ):
            raise ValueError("Collector pairing changed")
        if raw.get("canonical_namespace") != CANONICAL_NAMESPACE:
            raise ValueError("CANO namespace changed")
        if raw.get("canonical_seed_sequence") != "SeedSequence([seed, 0x43414E4F])":
            raise ValueError("Canonical SeedSequence definition changed")
        if raw.get("real_transition_budget") != 300_000:
            raise ValueError("Real-transition budget changed")
        if raw.get("minimum_physically_possible_episode_length") != 347:
            raise ValueError("Physical lower episode-length bound changed")
        exposures = (
            raw["real_transition_budget"]
            // raw["minimum_physically_possible_episode_length"]
            + 1
        )
        if exposures != raw.get("expected_max_exposed_tasks") or exposures != 865:
            raise ValueError("Reachable stream horizon changed")
        if raw.get("gradient_updates_per_policy") != 30_000:
            raise ValueError("Gradient-update budget changed")
        if raw.get("update_per_transition") != 0.1:
            raise ValueError("Update ratio changed")
        if tuple(raw.get("diagnostic_checkpoints", ())) != CHECKPOINTS:
            raise ValueError("Diagnostic checkpoint schedule changed")
        if raw.get("recovery_checkpoint_interval") != 10_000:
            raise ValueError("Recovery cadence changed")
        selections = raw.get("evaluation_selections", {})
        if selections.get("diagnostic", {}).get("candidate_start") != 9_200_000:
            raise ValueError("Diagnostic namespace changed")
        if selections.get("validation", {}).get("candidate_start") != 9_300_000:
            raise ValueError("Validation namespace changed")
        if any(
            selections[name].get("accepted_count") != 18
            for name in ("diagnostic", "validation")
        ):
            raise ValueError("Evaluation-set size changed")
        if set(raw.get("expected_evaluation_fingerprints", {})) != {
            "diagnostic",
            "validation",
        }:
            raise ValueError("Evaluation fingerprints are not frozen")
        if set(raw.get("expected_stream_fingerprints", {})) != {"11", "29", "47"}:
            raise ValueError("Canonical stream fingerprints are not frozen")
        v2 = load_v2()
        if v2_configuration_sha256(v2) != raw["frozen_v2_configuration_sha256"]:
            raise RuntimeError("Frozen Constrained-V2 science changed")
        algorithm = v2.algorithm
        expected = (
            algorithm.gamma == 0.99
            and algorithm.n_step == 2
            and algorithm.batch_size == 256
            and algorithm.buffer_size == 100_000
            and algorithm.hidden_sizes == (128, 128)
            and algorithm.actor_learning_rate == 0.0005
            and algorithm.critic_learning_rate == 0.001
            and algorithm.initial_effective_alpha == 1.0
            and algorithm.alpha_learning_rate == 0.0003
            and algorithm.tau == 0.05
            and algorithm.lagrangian_pid == (0.05, 0.0005, 0.1)
            and algorithm.initial_multipliers == (0.0, 0.0)
        )
        if not expected:
            raise RuntimeError("Frozen SACLag configuration changed")

    @property
    def conditions(self) -> tuple[Condition, ...]:
        return tuple(Condition(**row) for row in self.raw["conditions"])

    @property
    def v2(self):
        return load_v2()

    @property
    def configuration_sha256(self) -> str:
        encoded = json.dumps(
            self.raw, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode()
        return sha256(encoded).hexdigest()


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH) -> StudyConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Collector-replication configuration root must be an object")
    return StudyConfiguration(raw)
