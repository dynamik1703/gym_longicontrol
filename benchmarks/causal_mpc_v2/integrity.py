"""Fail-closed provenance and execution-integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.content_split_audit.streams import PAPER_SEEDS

HERE = Path(__file__).resolve().parent
SPLIT_LEDGER = HERE / "split_ledger.json"
CANONICAL = HERE / "canonical.json"
METHOD_FILES = (
    "planner.py",
    "terminal_value.py",
    "controller.py",
    "fixed_price.py",
    "adaptive_price.py",
    "evaluation.py",
    "integrity.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_hashes() -> dict[str, str]:
    return {name: sha256_file(HERE / name) for name in METHOD_FILES}


def require_not_paper(numeric_seed: int) -> None:
    if int(numeric_seed) in PAPER_SEEDS:
        raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")


def verify_paper_seal() -> dict[str, Any]:
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    paper = ledger["paper"]
    if (
        paper["state"] != "SEALED_UNINSPECTED"
        or tuple(paper["numeric_exclusion_only"]) != tuple(PAPER_SEEDS)
        or paper["content_fingerprints_constructed"] != 0
        or paper["episode_count"] != 0
    ):
        raise RuntimeError("Paper seal changed")
    return dict(paper)


def verify_frozen_method() -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    expected = canonical.get("frozen_method_sha256")
    if not expected:
        raise RuntimeError("method has not been frozen after calibration")
    observed = method_hashes()
    if observed != expected:
        raise RuntimeError("controller parameter/source change after method freeze")


def verify_capability_unopened() -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if canonical["execution"]["capability_opened"]:
        raise PermissionError("MPC_V2_CAPABILITY is an exactly-once panel")
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    if ledger["capability_panel"]["performance_episode_count"] != 0:
        raise RuntimeError("capability ledger is inconsistent with unopened state")

