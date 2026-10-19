"""Frozen V3 preregistration and validation."""

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
SOURCE_COMMIT = "ad2ad65df655ef1797e52e9e9f8f82b70b1c5d75"
CONDITION_NAMES = (
    "A_histTrack_histCollector_seed11",
    "B_histTrack_currentCollector_seed11",
    "C_currentTrack_histCollector_seed11",
    "D_currentTrack_currentCollector_seed11",
)
CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
EXPECTED_STREAM_FINGERPRINTS = {
    "T_HIST": "1f0504f969a55bc0c1af144fc2d228a81798a8a52e61d52edec406ae6b3582ed",
    "T_CURRENT": "3278239196a40db9f42c4c613dcfa93c979636e20cfd3f30326a3000a0635159",
}


@dataclass(frozen=True)
class Condition:
    id: str
    name: str
    track_stream: str
    collector: str


@dataclass(frozen=True)
class StudyConfiguration:
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        raw = self.raw
        if raw.get("schema_version") != 1 or raw.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("V3 identity or source lineage changed")
        conditions = tuple(Condition(**row) for row in raw.get("conditions", ()))
        if tuple(row.name for row in conditions) != CONDITION_NAMES:
            raise ValueError("Factorial condition order changed")
        if tuple(row.id for row in conditions) != ("A", "B", "C", "D"):
            raise ValueError("Factorial IDs changed")
        if {(row.track_stream, row.collector) for row in conditions} != {
            ("T_HIST", "C_HIST"),
            ("T_HIST", "C_CURRENT"),
            ("T_CURRENT", "C_HIST"),
            ("T_CURRENT", "C_CURRENT"),
        }:
            raise ValueError("The frozen 2x2 treatment matrix changed")
        if raw.get("training_seed") != 11:
            raise ValueError("Training seed changed")
        if raw.get("real_transition_budget") != 300_000:
            raise ValueError("Real-transition budget changed")
        if raw.get("minimum_physically_possible_episode_length") != 347:
            raise ValueError("Physical lower episode-length bound changed")
        expected_exposures = (
            raw["real_transition_budget"]
            // raw["minimum_physically_possible_episode_length"]
            + 1
        )
        if expected_exposures != raw.get("expected_max_exposed_tasks") or (
            expected_exposures != 865
        ):
            raise ValueError("Reachable stream horizon changed")
        if raw.get("gradient_updates_per_policy") != 30_000:
            raise ValueError("Gradient-update budget changed")
        if raw.get("update_per_transition") != 0.1:
            raise ValueError("Update ratio changed")
        if tuple(raw.get("diagnostic_checkpoints", ())) != CHECKPOINTS:
            raise ValueError("Diagnostic checkpoint schedule changed")
        if raw.get("recovery_checkpoint_interval") != 10_000:
            raise ValueError("Recovery cadence changed")
        selection = raw["diagnostic_selection"]
        if selection["candidate_start"] != 9_100_000:
            raise ValueError("V3 candidate namespace changed")
        if selection["accepted_count"] != 18:
            raise ValueError("Diagnostic-set size changed")
        if raw["expected_stream_fingerprints"] != EXPECTED_STREAM_FINGERPRINTS:
            raise ValueError("Expected stream fingerprints changed")
        v2 = load_v2()
        if (
            v2_configuration_sha256(v2)
            != raw["frozen_v2_configuration_sha256"]
        ):
            raise RuntimeError("Frozen Constrained-V2 science changed")
        algorithm = v2.algorithm
        expected_algorithm = (
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
        if not expected_algorithm:
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
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH) -> StudyConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V3 configuration root must be an object")
    return StudyConfiguration(raw)
