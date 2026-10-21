"""Fail-closed provenance, authorization and Paper/Validation boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.content_split_audit.streams import PAPER_SEEDS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "canonical.json"
SPLIT_LEDGER = HERE / "split_ledger.json"
PAIR_LEDGER = HERE / "pair_ledger.json"

METHOD_FILES = (
    "option_state.py",
    "feedback_executor.py",
    "option_selector.py",
    "controller.py",
    "observation_adapter.py",
    "checkpointing.py",
    "semantics_audit.py",
    "split.py",
    "evaluation.py",
    "execution.py",
    "integrity.py",
    "report.py",
)
INHERITED_FILES = (
    "benchmarks/causal_mpc_v2/planner.py",
    "benchmarks/causal_mpc_v2/controller.py",
    "benchmarks/causal_mpc_v2/terminal_value.py",
    "benchmarks/causal_mpc_v2/fixed_price.py",
    "benchmarks/causal_mpc_v2/adaptive_price.py",
    "benchmarks/physical_feasibility_reference/causal_controller.py",
    "benchmarks/physical_feasibility_reference/oracle_controller.py",
    "benchmarks/hybrid_rl_design/controller.py",
    "benchmarks/hybrid_rl_design/viability.py",
    "src/gym_longicontrol/domain/dynamics.py",
    "src/gym_longicontrol/domain/vehicle.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_hashes() -> dict[str, str]:
    return {name: sha256_file(HERE / name) for name in METHOD_FILES}


def inherited_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in INHERITED_FILES}


def require_not_paper(numeric_seed: int) -> None:
    if int(numeric_seed) in PAPER_SEEDS:
        raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")


def verify_paper_and_validation_seals() -> dict[str, Any]:
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    paper = ledger["paper"]
    if (
        paper["state"] != "SEALED_UNINSPECTED"
        or tuple(paper["numeric_exclusion_only"]) != tuple(PAPER_SEEDS)
        or paper["content_fingerprints_constructed"] != 0
        or paper["episode_count"] != 0
    ):
        raise RuntimeError("Paper seal changed")
    if ledger["historical_validation_panels_opened"] != 0:
        raise RuntimeError("an old Validation panel was opened")
    return {"paper": paper, "historical_validation_panels_opened": 0}


def verify_frozen_method() -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if not canonical["authorization"]["capability_authorized"]:
        raise PermissionError("OPTIONS_CAPABILITY_V1 is not authorized")
    if canonical["authorization"]["rl_training_authorized"]:
        raise RuntimeError("RL authorization boundary changed")
    if method_hashes() != canonical["frozen_method_sha256"]:
        raise RuntimeError("method changed after capability freeze")
    if inherited_hashes() != canonical["inherited_source_sha256"]:
        raise RuntimeError("inherited scientific source changed")
    verify_paper_and_validation_seals()
