"""Repository, source, authorization, and protected-boundary gates."""

from __future__ import annotations

import subprocess
from typing import Any

from benchmarks.content_split_audit.streams import PAPER_SEEDS

from .checkpoint_audit import production_identity, verify_v1_incident_immutable
from .constants import (
    AUTHORIZATION,
    BASE_COMMIT,
    BASE_TREE,
    CANONICAL,
    EXPECTED_BRANCH,
    HERE,
    PAIR_LEDGER,
    PREPARATION_STATUS,
    ROOT,
    SPLIT_LEDGER,
    VERIFIED_REMOTE,
    VERIFIED_REMOTE_URL,
)
from .ledger import validate_pair_ledger
from .utils import canonical_sha256, file_sha256, load_json

FROZEN_EXTERNAL_SOURCES = (
    "src/gym_longicontrol/domain/dynamics.py",
    "src/gym_longicontrol/domain/metrics.py",
    "src/gym_longicontrol/domain/task.py",
    "src/gym_longicontrol/domain/vehicle.py",
    "src/gym_longicontrol/envs/longicontrol.py",
    "src/gym_longicontrol/assets/vehicle/BMW_electric_i3_2014.npz",
    "benchmarks/physical_feasibility_reference/causal_controller.py",
    "benchmarks/hybrid_rl_design/controller.py",
    "benchmarks/hybrid_rl_design/viability.py",
    "benchmarks/causal_mpc_v2/controller.py",
    "benchmarks/causal_mpc_v2/planner.py",
    "benchmarks/feedback_chunk_rl_v1/chunk_decoder.py",
    "benchmarks/feedback_chunk_rl_v1/residual_controller.py",
    "benchmarks/feedback_chunk_rl_v1/sac_agent.py",
    "benchmarks/feedback_chunk_rl_v1/checkpointing.py",
    "benchmarks/feedback_chunk_rl_v1/evaluation.py",
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_identity() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    remote_url = _git("remote", "get-url", VERIFIED_REMOTE)
    base_tree = _git("show", "-s", "--format=%T", BASE_COMMIT)
    ancestor = (
        subprocess.run(
            ("git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"),
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )
    checks = {
        "branch": branch == EXPECTED_BRANCH,
        "verified_remote": remote_url == VERIFIED_REMOTE_URL,
        "base_tree": base_tree == BASE_TREE,
        "base_is_ancestor": ancestor,
    }
    if not all(checks.values()):
        raise RuntimeError(f"repository identity gate failed: {checks}")
    return {
        "branch": branch,
        "head": _git("rev-parse", "HEAD"),
        "base_commit": BASE_COMMIT,
        "base_tree": base_tree,
        "verified_remote": VERIFIED_REMOTE,
        "verified_remote_url": remote_url,
        "checks": checks,
    }


def scientific_source_hashes() -> dict[str, str]:
    paths = [ROOT / relative for relative in FROZEN_EXTERNAL_SOURCES]
    paths.extend(sorted(HERE.glob("*.py")))
    return {
        str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(set(paths))
    }


def source_bundle() -> dict[str, Any]:
    hashes = scientific_source_hashes()
    return {"files": hashes, "sha256": canonical_sha256(hashes)}


def verify_source_bundle(expected: dict[str, Any]) -> None:
    observed = source_bundle()
    if observed != expected:
        changed = sorted(set(observed["files"]) | set(expected.get("files", {})))
        changed = [
            path
            for path in changed
            if observed["files"].get(path) != expected.get("files", {}).get(path)
        ]
        raise RuntimeError(f"frozen scientific source changed: {changed}")


def verify_protected_boundaries() -> dict[str, Any]:
    incident = verify_v1_incident_immutable()
    split = load_json(SPLIT_LEDGER)
    accepted = split["panel"]["accepted_tasks"]
    paper_overlap = sorted(
        set(PAPER_SEEDS) & {int(row["numeric_seed"]) for row in accepted}
    )
    checks = {
        "v1_counts_immutable": incident["pair_counts"]
        == {"COMPLETED": 252, "RUNNING": 1, "NOT_STARTED": 647},
        "validation_unopened": split["validation"]["performance_episode_count"] == 0,
        "paper_unopened": split["paper"]["performance_episode_count"] == 0,
        "paper_fingerprints_absent": split["paper"]["content_fingerprints_constructed"]
        == 0,
        "paper_numeric_disjoint": not paper_overlap,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protected evaluation boundary failed: {checks}")
    return {"checks": checks, "v1_incident": incident, "paper_overlap": paper_overlap}


def require_authorized() -> dict[str, Any]:
    authorization = load_json(AUTHORIZATION)
    preparation = load_json(PREPARATION_STATUS)
    ledger = load_json(PAIR_LEDGER)
    validate_pair_ledger(ledger)
    if authorization.get("study") != "FEEDBACK_CHUNK_RUNTIME_ISOLATED_DIAGNOSTIC_V2":
        raise PermissionError("authorization belongs to another study")
    required = {
        "preparation_lock": False,
        "diagnostic_v2_authorized": True,
        "training_authorized": False,
        "validation_authorized": False,
        "paper_authorized": False,
    }
    for key, value in required.items():
        if authorization.get(key) != value:
            raise PermissionError(f"authorization gate failed: {key}")
    if not preparation.get("all_mandatory_gates_passed"):
        raise PermissionError("mandatory V2 preparation gates did not pass")
    if authorization.get("preparation_status_sha256") != file_sha256(
        PREPARATION_STATUS
    ):
        raise PermissionError("authorized preparation status changed")
    if authorization.get("pair_schedule_sha256") != ledger["schedule_sha256"]:
        raise PermissionError("authorized pair schedule changed")
    return authorization


def full_pre_execution_gate(expected_source: dict[str, Any]) -> dict[str, Any]:
    return {
        "repository": repository_identity(),
        "canonical_sha256": file_sha256(CANONICAL),
        "production_identity": production_identity(),
        "protected_boundaries": verify_protected_boundaries(),
        "authorization": require_authorized(),
        "source": (verify_source_bundle(expected_source) or expected_source),
    }
