"""Read-only verification and future loading of immutable Constrained actors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_rl_v2.config import load_configuration as load_v2
from benchmarks.constrained_v2_collector_replication.checkpointing import (
    file_sha256,
    load_verified,
    restore_policy,
)
from benchmarks.scalar_sac.experiment import _base_environment

from .config import TRAINING_SEEDS, StudyConfiguration


def reference_checkpoint(configuration: StudyConfiguration, seed: int) -> Path:
    return (
        Path(configuration.raw["reference_root"])
        / configuration.raw["reference_conditions"][str(seed)]
        / "step-300000"
        / "checkpoint.pt"
    )


def verify_references(configuration: StudyConfiguration) -> dict[str, Any]:
    root = Path(configuration.raw["reference_root"])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        path = reference_checkpoint(configuration, seed)
        metadata_path = path.with_name("checkpoint.pt.metadata.json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Constrained reference: {path}")
        expected = configuration.raw["reference_final_checkpoints"][str(seed)]
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Constrained reference hash changed: C{seed}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        condition = configuration.raw["reference_conditions"][str(seed)]
        record = manifest["runs"][condition]
        counters = {
            "real_transitions": int(record["real_transitions"]),
            "rl_gradient_updates": int(record["rl_gradient_updates"]),
        }
        if metadata["sha256"] != expected or metadata["transition_count"] != 300_000:
            raise RuntimeError("Constrained reference metadata changed")
        if record["state"] != "COMPLETED" or counters != {
            "real_transitions": 300_000,
            "rl_gradient_updates": 30_000,
        }:
            raise RuntimeError("Constrained reference final budget changed")
        rows[str(seed)] = {
            "condition": condition,
            "path": str(path),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "counters": counters,
            "verification_mode": "checkpoint bytes + sidecar + completed manifest",
        }
    return {
        "reference_constrained_verified": True,
        "strict_algorithmic_control": False,
        "reason": "GCSL uses 12D state and a different objective/update budget",
        "seeds": rows,
    }


@dataclass(frozen=True)
class ReferenceConfigurationView:
    v2: Any


def load_reference_policy(configuration: StudyConfiguration, seed: int):
    """Future-only adapter; construction does not evaluate or retrain."""

    v2 = load_v2()
    environment = _base_environment(v2)
    try:
        agent, _ = build_agent(
            v2, environment, training_seed=seed, device="cpu", threads=4
        )
        expected = configuration.raw["reference_final_checkpoints"][str(seed)]
        payload = load_verified(
            reference_checkpoint(configuration, seed), expected_sha256=expected
        )
        restore_policy(agent.policy, payload)
        return agent.policy, environment, ReferenceConfigurationView(v2=v2)
    except Exception:
        environment.close()
        raise
