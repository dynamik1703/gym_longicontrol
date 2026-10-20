"""Verification-only access to immutable prior Constrained checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_v2_collector_replication.checkpointing import (
    file_sha256,
    load_verified,
)

from .config import CHECKPOINTS, TRAINING_SEEDS, StudyConfiguration


def reference_checkpoint(
    configuration: StudyConfiguration, seed: int, step: int
) -> Path:
    condition = configuration.raw["reference_conditions"][str(seed)]
    return (
        Path(configuration.raw["reference_root"])
        / condition
        / f"step-{step:06d}"
        / "checkpoint.pt"
    )


def verify_references(configuration: StudyConfiguration) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        checkpoints: dict[str, Any] = {}
        for step in CHECKPOINTS:
            path = reference_checkpoint(configuration, seed, step)
            metadata_path = path.with_name("checkpoint.pt.metadata.json")
            if not path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Missing reference checkpoint: {path}")
            expected = configuration.raw["reference_checkpoints"][str(seed)][str(step)]
            actual = file_sha256(path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if actual != expected or metadata["sha256"] != expected:
                raise RuntimeError(f"Reference checkpoint hash changed: C{seed}@{step}")
            if int(metadata["transition_count"]) != step:
                raise RuntimeError("Reference checkpoint transition count changed")
            checkpoints[str(step)] = {
                "path": str(path),
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        final = load_verified(
            reference_checkpoint(configuration, seed, 300_000),
            expected_sha256=checkpoints["300000"]["sha256"],
        )
        if final["counters"] != {
            "real_transitions": 300_000,
            "rl_gradient_updates": 30_000,
        }:
            raise RuntimeError("Reference final budget changed")
        lambdas = [float(row["lagrangian"]) for row in final["lagrangian_states"]]
        expected_lambdas = (
            configuration.raw["scalarization"]["lambda_speed_values"][
                TRAINING_SEEDS.index(seed)
            ],
            configuration.raw["scalarization"]["lambda_deadline_values"][
                TRAINING_SEEDS.index(seed)
            ],
        )
        if not np.allclose(lambdas, expected_lambdas, rtol=0, atol=1e-15):
            raise RuntimeError("Frozen final multipliers changed")
        result[str(seed)] = {
            "condition": configuration.raw["reference_conditions"][str(seed)],
            "checkpoints": checkpoints,
            "final_lambdas": {"speed": lambdas[0], "deadline": lambdas[1]},
            "counters": final["counters"],
            "diagnostic_learning_curve_reusable": True,
        }
    return {
        "reference_constrained_verified": True,
        "matched_retraining_required": False,
        "seeds": result,
    }
