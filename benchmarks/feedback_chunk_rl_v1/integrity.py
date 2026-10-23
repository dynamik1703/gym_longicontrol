"""Fail-closed study provenance and authorization helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.content_split_audit.streams import PAPER_SEEDS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "canonical.json"
SPLIT_LEDGER = HERE / "split_ledger.json"
AUTHORIZATION = HERE / "authorization.json"
EXPECTED_BRANCH = "research/feedback-chunk-rl-v1"
BASE_COMMIT = "b1613e62ec381932cca3fb43ae698d2f75ffaa5e"
EXPECTED_TREE = "c1bc3f4eab0eb3a6d96966df25fe99a518493967"


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_repository(*, require_clean: bool) -> dict[str, str]:
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError(f"study must run on {EXPECTED_BRANCH}")
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"),
        cwd=ROOT,
        check=True,
    )
    if _git("show", "-s", "--format=%T", BASE_COMMIT) != EXPECTED_TREE:
        raise RuntimeError("frozen source tree changed")
    if require_clean and _git("status", "--porcelain"):
        raise RuntimeError("canonical execution requires a clean worktree")
    remotes = _git("remote", "-v")
    expected = (
        "gym_longicontrol_origin\thttps://github.com/dynamik1703/gym_longicontrol.git"
    )
    if expected not in remotes:
        raise RuntimeError("verified LongiControl remote is missing")
    return {
        "branch": EXPECTED_BRANCH,
        "base_commit": BASE_COMMIT,
        "base_tree": EXPECTED_TREE,
        "head": _git("rev-parse", "HEAD"),
    }


def require_not_paper(numeric_seed: int) -> None:
    if int(numeric_seed) in PAPER_SEEDS:
        raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")


def require_diagnostic_authorized() -> None:
    payload = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if not payload.get("diagnostic_authorized"):
        raise PermissionError("Diagnostic execution is not authorized")


def require_main_training_authorized() -> None:
    payload = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    if not payload.get("main_training_enabled"):
        raise PermissionError("main training gates have not enabled execution")


def require_validation_authorized() -> None:
    raise PermissionError("CHUNK_RL_VALIDATION_V1 is FROZEN_UNOPENED")


def require_paper_authorized() -> None:
    raise PermissionError("Paper is SEALED_UNINSPECTED")


def canonical_hashes() -> dict[str, str]:
    return {
        "canonical_sha256": sha256_file(CANONICAL),
        "split_ledger_sha256": sha256_file(SPLIT_LEDGER),
    }


def read_split() -> dict[str, Any]:
    return json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
