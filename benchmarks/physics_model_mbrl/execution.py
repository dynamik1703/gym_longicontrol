"""Isolated manifest and gates for the Physics-only MBRL matrix."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.model_based_rl.execution import atomic_json, scientific_hashes, utc_now

from .config import EXPECTED_SOURCE_COMMIT, PHYSICS_CONDITION, load_configuration

DEFAULT_OUTPUT_ROOT = Path("runs/physics-model-mbrl-v1")


def assert_source_ancestor(repository: str | Path = ".") -> None:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_SOURCE_COMMIT, "HEAD"],
        cwd=repository,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "Study branch does not descend from the frozen source commit"
        )


def initial_manifest() -> dict[str, Any]:
    configuration = load_configuration()
    runs = {
        f"{PHYSICS_CONDITION}-seed-{seed}": {
            "condition": PHYSICS_CONDITION,
            "training_seed": seed,
            "state": "NOT_STARTED",
            "attempts": [],
            "real_transitions": 0,
            "synthetic_transitions": 0,
            "rl_gradient_updates": 0,
            "model_gradient_updates": 0,
            "final_checkpoint_sha256": None,
            "development_complete": False,
        }
        for seed in configuration.raw["training_seeds"]
    }
    return {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "configuration_sha256": configuration.configuration_sha256,
        "parent_configuration_sha256": configuration.raw["parent_configuration_sha256"],
        "source_commit": configuration.raw["source_commit"],
        "created_at": utc_now(),
        "runs": runs,
        "validation": {"state": "SEALED", "opened_at": None},
        "paper_test": {"state": "SEALED", "opened_at": None},
    }


def load_or_create_manifest(output_root: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(output_root) / "manifest.json"
    configuration = load_configuration()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["configuration_sha256"] != configuration.configuration_sha256:
            raise RuntimeError("Existing manifest uses a different study configuration")
        expected_runs = {
            f"{PHYSICS_CONDITION}-seed-{seed}"
            for seed in configuration.raw["training_seeds"]
        }
        if set(payload["runs"]) != expected_runs:
            raise RuntimeError("Existing manifest is not the Physics-only matrix")
        return path, payload
    payload = initial_manifest()
    atomic_json(path, payload)
    return path, payload


def validation_ready(manifest: dict[str, Any]) -> bool:
    runs = tuple(manifest["runs"].values())
    return len(runs) == 3 and all(
        run["condition"] == PHYSICS_CONDITION
        and run["state"] == "COMPLETED"
        and run["real_transitions"] == 300_000
        and run["rl_gradient_updates"] == 30_000
        and bool(run["final_checkpoint_sha256"])
        and run["development_complete"] is True
        for run in runs
    )


def assert_validation_authorized(manifest: dict[str, Any]) -> None:
    configuration = load_configuration()
    if not configuration.raw["authorization"]["validation_authorized"]:
        raise PermissionError("Validation is not authorized")
    if not validation_ready(manifest):
        raise RuntimeError("All three Physics policies must freeze before Validation")
    if manifest["validation"] != {"state": "SEALED", "opened_at": None}:
        raise RuntimeError("Validation may be opened exactly once")


def assert_paper_test_blocked() -> None:
    raise PermissionError("Paper-test tracks 4000-4017 remain sealed")


def preflight_payload(repository: str | Path = ".") -> dict[str, Any]:
    configuration = load_configuration()
    parent = configuration.parent
    return {
        "study": configuration.raw["name"],
        "source_commit": configuration.raw["source_commit"],
        "configuration_sha256": configuration.configuration_sha256,
        "parent_configuration_sha256": parent.configuration_sha256,
        "condition": PHYSICS_CONDITION,
        "matrix": [
            [PHYSICS_CONDITION, seed] for seed in configuration.raw["training_seeds"]
        ],
        "total_real_transitions": configuration.raw["total_real_transitions"],
        "runtime_estimate": configuration.raw["runtime_estimate"],
        "authorization": configuration.raw["authorization"],
        "scientific_source_sha256": scientific_hashes(repository),
    }
