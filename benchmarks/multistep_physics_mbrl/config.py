"""Frozen configuration and validation for multi-step physics MVE."""

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
SOURCE_COMMIT = "8e5ebf6fb241f434781c0148c122a84c1ab4c35b"
CONDITIONS = ("matched_real_only", "physics_mve")
TRAINING_SEEDS = (11, 29, 47)
CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
MODEL_EXPANSION_STEPS = 60


@dataclass(frozen=True)
class StudyConfiguration:
    raw: dict[str, Any]

    def __post_init__(self) -> None:
        raw = self.raw
        if raw.get("schema_version") != 1:
            raise ValueError("Unsupported multi-step MVE schema")
        if raw.get("source_commit") != SOURCE_COMMIT:
            raise ValueError("The frozen H=1 source commit changed")
        if tuple(raw.get("conditions", ())) != CONDITIONS:
            raise ValueError("The matched two-condition matrix changed")
        if tuple(raw.get("training_seeds", ())) != TRAINING_SEEDS:
            raise ValueError("Training seeds changed")
        if raw.get("real_transitions_per_policy") != 300_000:
            raise ValueError("Per-policy real budget changed")
        if raw.get("total_real_transitions") != 1_800_000:
            raise ValueError("Six-policy real budget changed")
        if raw.get("rl_updates_per_policy") != 30_000:
            raise ValueError("Optimizer budget changed")
        if tuple(raw.get("real_transition_checkpoints", ())) != CHECKPOINTS:
            raise ValueError("Development checkpoint schedule changed")
        if raw.get("recovery_checkpoint_interval") != 10_000:
            raise ValueError("Recovery checkpoint cadence changed")

        target = raw["target"]
        expected_target = {
            "gamma": 0.99,
            "real_prefix_steps": 2,
            "model_expansion_steps": MODEL_EXPANSION_STEPS,
            "native_dt_s": 0.1,
            "real_prefix_time_s": 0.2,
            "model_expansion_time_s": 6.0,
            "total_nominal_horizon_steps": 62,
            "total_nominal_horizon_s": 6.2,
            "entropy_on_every_policy_value": True,
            "entropy_applies_to_cost_critics": True,
            "synthetic_replay": False,
        }
        if target != expected_target:
            raise ValueError("Frozen value-expansion target changed")

        physical = raw["physical_horizon"]
        if (
            physical["maximum_single_transition_drop_km_h"] != 40.0
            or physical["braking_case_start_km_h"] != 100.0
            or physical["braking_case_target_km_h"] != 60.0
            or physical["exact_steps"] != MODEL_EXPANSION_STEPS
            or physical["sensor_range_m"] != 150.0
        ):
            raise ValueError("Physical horizon derivation changed")

        splits = raw["track_splits"]
        expected_excluded = tuple(
            (
                *range(1000, 1009),
                *range(2000, 2009),
                *range(3000, 3009),
                *range(4000, 4018),
            )
        )
        if tuple(splits["excluded"]) != expected_excluded:
            raise ValueError("Reserved training exclusions changed")
        if tuple(splits["development"]) != tuple(range(2000, 2009)):
            raise ValueError("Development split changed")
        if tuple(splits["validation"]) != tuple(range(3000, 3009)):
            raise ValueError("Validation split changed")
        if tuple(splits["paper_test"]) != tuple(range(4000, 4018)):
            raise ValueError("Paper-final split changed")

        authorization = raw["authorization"]
        expected_authorization = {
            "main_training_authorized": True,
            "main_training_enabled": True,
            "validation_authorized": False,
            "paper_test_authorized": False,
            "automatic_fresh_restart_authorized": False,
        }
        if authorization != expected_authorization:
            raise PermissionError("Only frozen main-matrix training is authorized")

        v2 = load_v2()
        if v2_configuration_sha256(v2) != raw["frozen_v2"]["configuration_sha256"]:
            raise RuntimeError("Frozen Constrained-V2 configuration changed")
        algorithm = v2.algorithm
        if (
            algorithm.n_step != 2
            or algorithm.gamma != 0.99
            or algorithm.batch_size != 256
            or algorithm.update_per_step != 0.1
        ):
            raise RuntimeError("Frozen SACLag semantics changed")

    @property
    def configuration_sha256(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return sha256(payload).hexdigest()

    @property
    def v2(self):
        return load_v2()

    @property
    def model_expansion_steps(self) -> int:
        return int(self.raw["target"]["model_expansion_steps"])


def load_configuration(path: str | Path = DEFAULT_CONFIG_PATH) -> StudyConfiguration:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be an object")
    return StudyConfiguration(raw)
