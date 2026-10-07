"""Frozen scheduling, provenance, run protection, and evaluation gates."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import uuid
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
    PLANNED_COMPLETE_UPDATE_CYCLES,
    PLANNED_DEPTHS,
    PLANNED_SEEDS,
    PLANNED_TRANSITIONS_PER_POLICY,
    PREFILL_TRANSITIONS,
    SEALED_PAPER_TRACKS,
    UPDATE_INTERVAL_TRANSITIONS,
    VALIDATION_TRACKS,
)

PREPARATION_COMMIT = "8c9a61ea545d339faaa12a201dbdca7120706d44"
EXPECTED_CONFIGURATION_SHA256 = (
    "659e139034d9f3aed25a07d5244bc6ac4c86ce63dd0394f315ca5341ae0a78fc"
)
DEFAULT_OUTPUT_ROOT = Path("runs/contrastive-rl-projected-depth-v1")
STUDY_SCHEMA_VERSION = 1
POLICY_STATES = frozenset({"NOT_STARTED", "RUNNING", "INTERRUPTED", "COMPLETED"})
SCIENTIFIC_SHA256 = {
    "benchmarks/contrastive_rl/canonical.json": EXPECTED_CONFIGURATION_SHA256,
    "benchmarks/contrastive_rl/config.py": (
        "3cf2224f724cf544a2654c56487f2800ac595758913ea106117f959e6083d2c9"
    ),
    "benchmarks/contrastive_rl/learner.py": (
        "b159aabe57c02f67469d75a9b91ec5104be47f806fa1119a69c0fd26ea776ce0"
    ),
    "benchmarks/contrastive_rl/losses.py": (
        "347d60100b841ea7f029cfcde4fb88164ac74e87283e6cdd92088d24c13e5aca"
    ),
    "benchmarks/contrastive_rl/networks.py": (
        "e927bf9ab87b8a7433777741deb66b4826a7c7ae7c7b60edbabdc40d841e028a"
    ),
    "benchmarks/contrastive_rl/goal_adapter.py": (
        "d4bb6a029298368cf4fcc79112e9b405116ed6f8bc52341849640b1588230630"
    ),
    "benchmarks/contrastive_rl/projected_adapter.py": (
        "0daa4ae6fa82b06297a6f96864ff5fe7adf8985a14ea8636e32a1f25f9c89932"
    ),
    "benchmarks/contrastive_rl/sampling.py": (
        "6a019d1af9eed291fe3982e56f9ecc81b3ec7754abb9d2d37ec27ea407dcaf34"
    ),
    "benchmarks/contrastive_rl/upstream.json": (
        "932aaafacd62339e8f05834be563f55067deabdca7379997168a28a10bfe1958"
    ),
}
EXECUTION_SOURCE_FILES = (
    "benchmarks/contrastive_rl/checkpointing.py",
    "benchmarks/contrastive_rl/diagnostics.py",
    "benchmarks/contrastive_rl/evaluation.py",
    "benchmarks/contrastive_rl/execution.py",
    "benchmarks/contrastive_rl/replay.py",
    "benchmarks/contrastive_rl/runner.py",
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
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def should_update(transition_count: int) -> bool:
    return bool(
        transition_count > PREFILL_TRANSITIONS
        and (transition_count - PREFILL_TRANSITIONS) % UPDATE_INTERVAL_TRANSITIONS == 0
        and transition_count <= PLANNED_TRANSITIONS_PER_POLICY
    )


def scheduled_update_transitions() -> tuple[int, ...]:
    return tuple(
        range(
            PREFILL_TRANSITIONS + UPDATE_INTERVAL_TRANSITIONS,
            PLANNED_TRANSITIONS_PER_POLICY + 1,
            UPDATE_INTERVAL_TRANSITIONS,
        )
    )


class TrainingTrackStream:
    """Dedicated PCG64 stream whose domain excludes every research split."""

    DOMAIN_TAG = 0x43524C31

    def __init__(self, training_seed: int, state: Mapping[str, Any] | None = None):
        self.training_seed = int(training_seed)
        seed_sequence = np.random.SeedSequence([self.training_seed, self.DOMAIN_TAG])
        self._rng = np.random.Generator(np.random.PCG64(seed_sequence))
        if state is not None:
            self._rng.bit_generator.state = dict(state)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "algorithm": "numpy.random.PCG64",
            "seed_sequence_entropy": [self.training_seed, self.DOMAIN_TAG],
            "excluded_tracks": sorted(_RESERVED_TRACKS),
            "claim_identical_realized_tracks_across_policies": False,
        }

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._rng.bit_generator.state)

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
    """Verify frozen science and, optionally, the separate execution gate."""

    root = Path(repo_root).resolve()
    if _git(root, "branch", "--show-current") != "research/contrastive-rl":
        raise RuntimeError("runner must execute on research/contrastive-rl")
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", PREPARATION_COMMIT, "HEAD"),
        cwd=root,
        check=True,
    )
    if require_clean and _git(root, "status", "--porcelain"):
        raise RuntimeError("working-tree changes block main-study execution")
    if file_sha256(root / "benchmarks/contrastive_rl/canonical.json") != (
        EXPECTED_CONFIGURATION_SHA256
    ):
        raise RuntimeError("frozen scientific configuration hash changed")
    actual_science = {
        relative: file_sha256(root / relative) for relative in SCIENTIFIC_SHA256
    }
    if actual_science != SCIENTIFIC_SHA256:
        raise RuntimeError("frozen CRL scientific sources changed")
    preparation = json.loads(
        (root / "benchmarks/contrastive_rl/preparation_status.json").read_text()
    )
    for field in ("task_mapping_verified", "technical_semantic_readiness"):
        if preparation.get(field) is not True:
            raise RuntimeError(f"preparation gate is false: {field}")
    if preparation.get("execution_infrastructure_ready") is not True:
        raise RuntimeError("execution infrastructure is not marked ready")
    if require_training_authorization and not (
        preparation.get("main_training_authorized") is True
        and preparation.get("main_training_enabled") is True
    ):
        raise PermissionError("CRL main training is not separately authorized")
    execution_hashes = {
        relative: file_sha256(root / relative) for relative in EXECUTION_SOURCE_FILES
    }
    return {
        "preparation_commit": PREPARATION_COMMIT,
        "execution_commit": _git(root, "rev-parse", "HEAD"),
        "branch": "research/contrastive-rl",
        "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "scientific_source_sha256": actual_science,
        "execution_source_sha256": execution_hashes,
        "runtime_versions": runtime_versions(),
        "historical_frozen_artifact_count_verified": _verify_historical_hashes(root),
    }


def policy_key(depth: int, seed: int) -> str:
    return f"depth-{depth}:seed-{seed}"


def new_manifest(provenance: Mapping[str, Any]) -> dict[str, Any]:
    order = [
        {"depth": depth, "training_seed": seed}
        for depth in PLANNED_DEPTHS
        for seed in PLANNED_SEEDS
    ]
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_id": str(uuid.uuid4()),
        "status": "NOT_STARTED",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "validation_opened": False,
        "paper_tracks_opened": False,
        "provenance": dict(provenance),
        "execution_order": order,
        "policies": {
            policy_key(item["depth"], item["training_seed"]): {
                **item,
                "status": "NOT_STARTED",
                "attempts": [],
                "native_transitions": 0,
                "complete_update_cycles": 0,
                "development_checkpoints": [],
            }
            for item in order
        },
        "actual_resources_across_attempts": {
            "native_simulator_transitions": 0,
            "complete_update_cycles": 0,
            "development_simulator_transitions": 0,
        },
        "planned_primary_budget": {
            "native_simulator_transitions": 6 * PLANNED_TRANSITIONS_PER_POLICY,
            "complete_update_cycles": 6 * PLANNED_COMPLETE_UPDATE_CYCLES,
        },
    }


def validate_resume_provenance(
    manifest: Mapping[str, Any], current_provenance: Mapping[str, Any]
) -> None:
    """Refuse continuation under different scientific or execution sources."""

    recorded = manifest.get("provenance", {})
    for field in (
        "configuration_sha256",
        "scientific_source_sha256",
        "execution_source_sha256",
    ):
        if recorded.get(field) != current_provenance.get(field):
            raise RuntimeError(f"resume provenance changed: {field}")


class ExclusiveStudyLock(AbstractContextManager):
    def __init__(self, root: str | Path, study_id: str):
        self.root = Path(root)
        self.study_id = study_id
        self.path = self.root / "ACTIVE.lock"
        self.acquired = False

    def __enter__(self):
        payload = {
            "study_id": self.study_id,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": utc_now(),
        }
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            raise RuntimeError("another CRL study process holds ACTIVE.lock") from error
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
            "CRL study root already exists; a new directory cannot evade duplicate "
            "or partial-run protection"
        ) from error
    manifest = new_manifest(provenance)
    atomic_json(destination / "manifest.json", manifest)
    return destination, manifest


def update_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    atomic_json(Path(root) / "manifest.json", manifest)


def start_policy(manifest: dict[str, Any], depth: int, seed: int) -> dict[str, Any]:
    item = manifest["policies"][policy_key(depth, seed)]
    if item["status"] == "COMPLETED":
        raise RuntimeError("completed CRL policies are immutable")
    if item["status"] == "RUNNING":
        raise RuntimeError("a second copy of a RUNNING CRL policy is forbidden")
    if item["status"] == "INTERRUPTED":
        raise RuntimeError("INTERRUPTED policy requires explicit resume or restart")
    if item["status"] != "NOT_STARTED":
        raise RuntimeError(f"unsupported policy status: {item['status']}")
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
    attempt = item["attempts"][-1]
    attempt.update(
        {
            "status": "INTERRUPTED",
            "interrupted_at": utc_now(),
            "native_transitions": int(transition_count),
            "complete_update_cycles": int(update_cycles),
            "reason": reason,
        }
    )


def permit_exact_resume(item: dict[str, Any], checkpoint_sha256: str) -> None:
    if item["status"] != "INTERRUPTED":
        raise RuntimeError("exact resume requires an INTERRUPTED policy")
    if len(checkpoint_sha256) != 64:
        raise ValueError("exact resume requires the recorded checkpoint hash")
    item["status"] = "RUNNING"
    item["resumed_at"] = utc_now()
    item["resume_checkpoint_sha256"] = checkpoint_sha256


def authorize_fresh_restart(
    item: dict[str, Any], authorization: Mapping[str, Any]
) -> None:
    if item["status"] != "INTERRUPTED":
        raise RuntimeError("fresh restart requires a preserved INTERRUPTED attempt")
    required = {"authorized_by", "authorized_at", "reason", "prior_attempt_hash"}
    if required - authorization.keys():
        raise ValueError("fresh restart authorization is incomplete")
    item.setdefault("restart_authorizations", []).append(dict(authorization))
    item["status"] = "NOT_STARTED"
    item["native_transitions"] = 0
    item["complete_update_cycles"] = 0


def validate_validation_gate(root: str | Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("validation_opened"):
        raise RuntimeError("Validation has already been opened")
    if manifest.get("paper_tracks_opened"):
        raise RuntimeError("paper-final tracks must remain sealed")
    policies = manifest.get("policies", {})
    if len(policies) != 6:
        raise RuntimeError("Validation is sealed until all six policies exist")
    for depth in PLANNED_DEPTHS:
        for seed in PLANNED_SEEDS:
            item = policies[policy_key(depth, seed)]
            if item.get("status") != "COMPLETED":
                raise RuntimeError(
                    "Validation is sealed until all six policies complete"
                )
            if item.get("native_transitions") != PLANNED_TRANSITIONS_PER_POLICY:
                raise RuntimeError("final policy transition count is not 300,000")
            if item.get("complete_update_cycles") != (PLANNED_COMPLETE_UPDATE_CYCLES):
                raise RuntimeError("final policy update count is not 7,250")
            recorded_checkpoints = [
                row["transition_count"] for row in item["development_checkpoints"]
            ]
            if recorded_checkpoints != list(DEVELOPMENT_CHECKPOINTS):
                raise RuntimeError("Development artifacts are incomplete")
            checkpoint = Path(root) / item["final_checkpoint_path"]
            if file_sha256(checkpoint) != item["final_checkpoint_sha256"]:
                raise RuntimeError("final checkpoint hash changed")
            if item.get("configuration_sha256") != EXPECTED_CONFIGURATION_SHA256:
                raise RuntimeError("final policy configuration hash changed")
            if item.get("scientific_source_sha256") != SCIENTIFIC_SHA256:
                raise RuntimeError("final policy source hashes changed")
            if item.get("execution_source_sha256") != manifest["provenance"].get(
                "execution_source_sha256"
            ):
                raise RuntimeError("final policy execution source hashes changed")


def assert_paper_tracks_sealed(*_args, **_kwargs) -> None:
    raise PermissionError("paper-final tracks 4000-4017 are sealed for CRL V1")
