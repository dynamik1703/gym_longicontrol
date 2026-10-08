"""Study locking, provenance, scheduling, duplicate protection, and gates."""

from __future__ import annotations

import json
import os
import socket
import subprocess
from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .checkpointing import file_sha256, runtime_versions
from .config import (
    DEVELOPMENT_CHECKPOINTS,
    DEVELOPMENT_TRACKS,
    PLANNED_SEEDS,
    PLANNED_TRANSITIONS_PER_POLICY,
    PLANNED_UPDATE_CYCLES,
    SEALED_PAPER_TRACKS,
    VALIDATION_TRACKS,
    should_update,
)

PREPARATION_COMMIT = "8c9a61ea545d339faaa12a201dbdca7120706d44"
EXPECTED_BRANCH = "research/gcsl"
STUDY_ID = "longicontrol-gcsl-v1"
DEFAULT_OUTPUT_ROOT = Path("runs/gcsl-v1")
STATUS_PATH = Path("benchmarks/gcsl/preparation_status.json")
SCIENTIFIC_SOURCE_FILES = (
    "benchmarks/gcsl/canonical.json",
    "benchmarks/gcsl/config.py",
    "benchmarks/gcsl/goal_adapter.py",
    "benchmarks/gcsl/learner.py",
    "benchmarks/gcsl/policy.py",
    "benchmarks/gcsl/replay.py",
    "benchmarks/gcsl/sampling.py",
    "benchmarks/gcsl/upstream.json",
)
EXECUTION_SOURCE_FILES = (
    "benchmarks/gcsl/checkpointing.py",
    "benchmarks/gcsl/diagnostics.py",
    "benchmarks/gcsl/evaluation.py",
    "benchmarks/gcsl/execution.py",
    "benchmarks/gcsl/runner.py",
)
_RESERVED_TRACKS = frozenset(
    (*range(1000, 1009), *DEVELOPMENT_TRACKS, *VALIDATION_TRACKS, *SEALED_PAPER_TRACKS)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


class TrainingTrackStream:
    """Dedicated PCG64 stream that rejects every reserved research split."""

    DOMAIN_TAG = 0x4743534C

    def __init__(self, training_seed: int, state: Mapping[str, Any] | None = None):
        self.training_seed = int(training_seed)
        sequence = np.random.SeedSequence([self.training_seed, self.DOMAIN_TAG])
        self._rng = np.random.Generator(np.random.PCG64(sequence))
        if state is not None:
            self._rng.bit_generator.state = dict(state)

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._rng.bit_generator.state)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "algorithm": "numpy.random.PCG64",
            "seed_sequence_entropy": [self.training_seed, self.DOMAIN_TAG],
            "excluded_tracks": sorted(_RESERVED_TRACKS),
        }

    def next_seed(self) -> int:
        while True:
            candidate = int(self._rng.integers(0, np.iinfo(np.uint32).max))
            if candidate not in _RESERVED_TRACKS:
                return candidate


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_historical_hashes(root: Path) -> int:
    manifest = root / "benchmarks/llm_reward/frozen_artifacts.sha256"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip(" *")
        if file_sha256(root / relative) != expected:
            raise RuntimeError(f"historical frozen artifact changed: {relative}")
        checked += 1
    return checked


def verify_preflight(
    repo_root: str | Path,
    *,
    require_clean: bool = True,
    require_training_authorization: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if _git(root, "branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"runner must execute on {EXPECTED_BRANCH}")
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", PREPARATION_COMMIT, "HEAD"),
        cwd=root,
        check=True,
    )
    if require_clean and _git(root, "status", "--porcelain"):
        raise RuntimeError("working-tree changes block GCSL study execution")
    status = json.loads((root / STATUS_PATH).read_text(encoding="utf-8"))
    if status.get("execution_infrastructure_ready") is not True:
        raise RuntimeError("execution infrastructure is not ready")
    actual_science = {
        relative: file_sha256(root / relative) for relative in SCIENTIFIC_SOURCE_FILES
    }
    if actual_science != status.get("scientific_source_sha256"):
        raise RuntimeError("frozen GCSL scientific source hashes changed")
    if require_training_authorization and not (
        status.get("ready_for_main_training") is True
        and status.get("main_training_authorized") is True
        and status.get("main_training_enabled") is True
    ):
        raise PermissionError("GCSL main training is not separately authorized")
    execution_hashes = {
        relative: file_sha256(root / relative) for relative in EXECUTION_SOURCE_FILES
    }
    return {
        "study_id": STUDY_ID,
        "preparation_commit": PREPARATION_COMMIT,
        "execution_commit": _git(root, "rev-parse", "HEAD"),
        "branch": EXPECTED_BRANCH,
        "configuration_sha256": actual_science[
            "benchmarks/gcsl/canonical.json"
        ],
        "scientific_source_sha256": actual_science,
        "execution_source_sha256": execution_hashes,
        "runtime_versions": runtime_versions(),
        "historical_frozen_artifact_count_verified": _verify_historical_hashes(root),
    }


def policy_key(seed: int) -> str:
    return f"seed-{seed}"


def new_manifest(provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "NOT_STARTED",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "validation_opened": False,
        "paper_tracks_opened": False,
        "provenance": dict(provenance),
        "execution_order": list(PLANNED_SEEDS),
        "policies": {
            policy_key(seed): {
                "training_seed": seed,
                "status": "NOT_STARTED",
                "attempts": [],
                "native_transitions": 0,
                "complete_update_cycles": 0,
                "development_checkpoints": [],
            }
            for seed in PLANNED_SEEDS
        },
        "actual_resources_across_attempts": {
            "native_simulator_transitions": 0,
            "complete_update_cycles": 0,
            "development_simulator_transitions": 0,
            "validation_simulator_transitions": 0,
        },
        "planned_primary_budget": {
            "native_simulator_transitions": 3 * PLANNED_TRANSITIONS_PER_POLICY,
            "complete_update_cycles": 3 * PLANNED_UPDATE_CYCLES,
            "replay_samples": 3 * PLANNED_UPDATE_CYCLES * 256,
        },
    }


class ExclusiveStudyLock(AbstractContextManager):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.path = self.root / "ACTIVE.lock"
        self.acquired = False

    def __enter__(self):
        payload = {
            "study_id": STUDY_ID,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": utc_now(),
        }
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            message = "another GCSL study process holds ACTIVE.lock"
            raise RuntimeError(message) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.acquired and self.path.exists():
            self.path.unlink()
        self.acquired = False
        return False


def initialize_study_root(
    root: str | Path, provenance: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    destination = Path(root)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            "fixed GCSL study root already exists; duplicate/free reruns are forbidden"
        ) from error
    manifest = new_manifest(provenance)
    atomic_json(destination / "manifest.json", manifest)
    return destination, manifest


def update_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    atomic_json(Path(root) / "manifest.json", manifest)


def start_policy(manifest: dict[str, Any], seed: int) -> dict[str, Any]:
    item = manifest["policies"][policy_key(seed)]
    if item["status"] != "NOT_STARTED":
        raise RuntimeError(f"policy cannot start from {item['status']}")
    item["status"] = "RUNNING"
    item["started_at"] = utc_now()
    item["attempts"].append(
        {
            "attempt": len(item["attempts"]) + 1,
            "status": "RUNNING",
            "started_at": item["started_at"],
            "native_transitions": 0,
            "complete_update_cycles": 0,
        }
    )
    return item


def interrupt_policy(
    item: dict[str, Any], *, transition_count: int, update_cycles: int, reason: str
) -> None:
    if item["status"] != "RUNNING":
        raise RuntimeError("only a RUNNING policy can be interrupted")
    item["status"] = "INTERRUPTED"
    item["native_transitions"] = int(transition_count)
    item["complete_update_cycles"] = int(update_cycles)
    item["attempts"][-1].update(
        {
            "status": "INTERRUPTED",
            "interrupted_at": utc_now(),
            "native_transitions": int(transition_count),
            "complete_update_cycles": int(update_cycles),
            "reason": reason,
        }
    )


def permit_exact_resume(item: dict[str, Any], checkpoint_sha256: str) -> None:
    if item["status"] != "INTERRUPTED" or len(checkpoint_sha256) != 64:
        raise RuntimeError("exact resume requires the recorded interrupted checkpoint")
    item["status"] = "RUNNING"
    item["resumed_at"] = utc_now()
    item["resume_checkpoint_sha256"] = checkpoint_sha256


def authorize_fresh_restart(
    item: dict[str, Any], authorization: Mapping[str, Any]
) -> None:
    required = {"authorized_by", "authorized_at", "reason", "prior_attempt_hash"}
    if item["status"] != "INTERRUPTED" or required - authorization.keys():
        raise RuntimeError("fresh restart requires explicit complete authorization")
    item.setdefault("restart_authorizations", []).append(dict(authorization))
    item["status"] = "NOT_STARTED"
    item["native_transitions"] = 0
    item["complete_update_cycles"] = 0


def validate_resume_provenance(
    manifest: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    recorded = manifest.get("provenance", {})
    for field in (
        "configuration_sha256",
        "scientific_source_sha256",
        "execution_source_sha256",
    ):
        if recorded.get(field) != current.get(field):
            raise RuntimeError(f"resume provenance changed: {field}")


def validate_validation_gate(root: str | Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("validation_opened"):
        raise RuntimeError("Validation has already been opened")
    if manifest.get("paper_tracks_opened"):
        raise RuntimeError("paper-final tracks must remain sealed")
    policies = manifest.get("policies", {})
    if len(policies) != len(PLANNED_SEEDS):
        raise RuntimeError("Validation requires exactly three final policies")
    for seed in PLANNED_SEEDS:
        item = policies[policy_key(seed)]
        if item.get("status") != "COMPLETED":
            raise RuntimeError("Validation is sealed until all policies complete")
        if item.get("native_transitions") != PLANNED_TRANSITIONS_PER_POLICY:
            raise RuntimeError("policy transition budget is incomplete")
        if item.get("complete_update_cycles") != PLANNED_UPDATE_CYCLES:
            raise RuntimeError("policy update budget is incomplete")
        if [row["transition_count"] for row in item["development_checkpoints"]] != list(
            DEVELOPMENT_CHECKPOINTS
        ):
            raise RuntimeError("Development checkpoint matrix is incomplete")
        checkpoint = Path(root) / item["final_checkpoint_path"]
        if file_sha256(checkpoint) != item["final_checkpoint_sha256"]:
            raise RuntimeError("final checkpoint hash changed")


def assert_paper_tracks_sealed(*_args, **_kwargs) -> None:
    raise PermissionError("paper-final tracks 4000-4017 are sealed for GCSL V1")


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "ExclusiveStudyLock",
    "TrainingTrackStream",
    "assert_paper_tracks_sealed",
    "atomic_json",
    "authorize_fresh_restart",
    "initialize_study_root",
    "interrupt_policy",
    "new_manifest",
    "permit_exact_resume",
    "policy_key",
    "should_update",
    "start_policy",
    "update_manifest",
    "validate_resume_provenance",
    "validate_validation_gate",
    "verify_preflight",
]
