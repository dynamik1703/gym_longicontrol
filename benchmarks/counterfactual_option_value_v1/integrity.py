"""Frozen-source, authorization, split and Paper boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.content_split_audit.streams import PAPER_SEEDS
from benchmarks.hybrid_rl_design.controller import (
    _endpoint_limit,
    _unknown_preview_safe,
)
from benchmarks.hybrid_rl_design.viability import pessimistic_completion_bound
from benchmarks.physical_feasibility_reference.oracle_controller import (
    VELOCITY_SAFETY_MARGIN_M_S,
    Transition,
    braking_safe,
)
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "canonical.json"
SPLIT_LEDGER = HERE / "split_ledger.json"
BRANCH_LEDGER = HERE / "branch_ledger.json"

METHOD_FILES = (
    "split.py",
    "execution.py",
    "integrity.py",
    "instrumented_controller.py",
    "snapshot.py",
    "branch_runner.py",
    "alternative_selector.py",
    "dataset.py",
    "opportunity_analysis.py",
    "predictor.py",
    "predictor_evaluation.py",
)
INHERITED_FILES = (
    "benchmarks/temporal_options_v1/controller.py",
    "benchmarks/temporal_options_v1/feedback_executor.py",
    "benchmarks/temporal_options_v1/option_selector.py",
    "benchmarks/temporal_options_v1/option_state.py",
    "benchmarks/temporal_options_v1/observation_adapter.py",
    "benchmarks/temporal_options_v1/checkpointing.py",
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
    "src/gym_longicontrol/envs/longicontrol.py",
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


def verify_seals() -> dict[str, Any]:
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    paper = ledger["paper"]
    if (
        paper["state"] != "SEALED_UNINSPECTED"
        or tuple(paper["numeric_exclusion_only"]) != tuple(PAPER_SEEDS)
        or paper["content_fingerprints_constructed"] != 0
        or paper["episode_count"] != 0
        or ledger["historical_validation_panels_opened"] != 0
    ):
        raise RuntimeError("Paper or historical Validation seal changed")
    return {"paper": paper, "historical_validation_panels_opened": 0}


def verify_frozen_method(*, require_labels_authorized: bool = False) -> None:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if (
        require_labels_authorized
        and not canonical["authorization"]["scientific_labels_authorized"]
    ):
        raise PermissionError("scientific counterfactual labels are not authorized")
    if canonical["authorization"]["full_rl_training_authorized"]:
        raise RuntimeError("full RL authorization boundary changed")
    if method_hashes() != canonical["frozen_method_sha256"]:
        raise RuntimeError("counterfactual method changed after freeze")
    if inherited_hashes() != canonical["inherited_source_sha256"]:
        raise RuntimeError("inherited scientific source changed")
    verify_seals()


def _close(left: float, right: float, tolerance: float = 1.0e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def verify_native_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently reconstruct every registered native integrity quantity.

    Deadline-bound falsification is frozen as disagreement between the retained
    ``completion_bound_s``/``deadline_viable`` pair and a fresh invocation of
    the inherited heuristic on the proposed-action successor.  It is an audit
    of a falsifiable heuristic, not a claim of global viability.
    """

    if not records:
        raise RuntimeError("empty records cannot certify an empty falsification list")
    vehicle = VehicleModel()
    config = SimulationConfig()
    required = {
        "native_step",
        "causal_state_before",
        "known_transitions",
        "current_limit_m_s",
        "proposed_native_action",
        "executed_action",
        "fallback_action",
        "accepted",
        "proposal_source",
        "completion_bound_s",
        "deadline_viable",
        "endpoint_speed_safe",
        "known_transition_braking_safe",
        "unknown_preview_safe",
        "actual_next_state",
        "signed_step_energy_kwh",
    }
    falsifications: list[dict[str, Any]] = []
    energy = 0.0
    expected_step = int(records[0]["native_step"])
    previous_next: dict[str, Any] | None = None
    for row in records:
        missing = required - row.keys()
        if missing:
            raise RuntimeError(f"native record missing fields: {sorted(missing)}")
        if int(row["native_step"]) != expected_step:
            raise RuntimeError("native step sequence is not contiguous")
        expected_step += 1
        state = VehicleState(**row["causal_state_before"])
        if previous_next is not None and row["causal_state_before"] != previous_next:
            raise RuntimeError("native state chain is discontinuous")
        transitions = tuple(Transition(**value) for value in row["known_transitions"])
        proposal = float(row["proposed_native_action"])
        proposed_next, _, _ = advance(state, proposal, vehicle, config)
        endpoint = _endpoint_limit(
            float(row["current_limit_m_s"]), proposed_next, transitions
        )
        endpoint_safe = (
            proposed_next.velocity_m_s
            <= endpoint - VELOCITY_SAFETY_MARGIN_M_S + 1.0e-12
        )
        known_safe = braking_safe(
            proposed_next, transitions, vehicle=vehicle, config=config
        )
        preview_safe, _ = _unknown_preview_safe(
            proposed_next, vehicle=vehicle, config=config
        )
        viability = pessimistic_completion_bound(
            proposed_next, vehicle=vehicle, config=config
        )
        comparisons = {
            "completion_bound_s": _close(
                row["completion_bound_s"], viability.completion_bound_s
            ),
            "deadline_viable": bool(row["deadline_viable"]) == viability.viable,
            "endpoint_speed_safe": bool(row["endpoint_speed_safe"]) == endpoint_safe,
            "known_transition_braking_safe": bool(row["known_transition_braking_safe"])
            == known_safe,
            "unknown_preview_safe": bool(row["unknown_preview_safe"]) == preview_safe,
        }
        if not all(comparisons.values()):
            falsifications.append(
                {
                    "native_step": int(row["native_step"]),
                    "failed_fields": [
                        key for key, passed in comparisons.items() if not passed
                    ],
                }
            )
        identical = _close(proposal, row["fallback_action"], 1.0e-15)
        expected_acceptance = (
            True
            if row["proposal_source"] == "FALLBACK_ONLY"
            else all(
                (endpoint_safe, known_safe, preview_safe, viability.viable or identical)
            )
        )
        if bool(row["accepted"]) != expected_acceptance:
            raise RuntimeError("incorrect retained acceptance flag")
        expected_action = (
            proposal
            if expected_acceptance or row["proposal_source"] == "FALLBACK_ONLY"
            else float(row["fallback_action"])
        )
        if not _close(row["executed_action"], expected_action, 1.0e-15):
            raise RuntimeError("executed action does not follow acceptance rule")
        actual_next, _, step_energy = advance(state, expected_action, vehicle, config)
        observed_next = VehicleState(**row["actual_next_state"])
        for key, value in asdict(actual_next).items():
            if not _close(value, getattr(observed_next, key)):
                raise RuntimeError("actual next state failed exact native replay")
        if not _close(row["signed_step_energy_kwh"], step_energy):
            raise RuntimeError("signed native energy failed replay")
        energy += float(row["signed_step_energy_kwh"])
        previous_next = row["actual_next_state"]
    if falsifications:
        raise RuntimeError(f"deadline-bound audit falsifications: {falsifications[:3]}")
    return {
        "native_record_count": len(records),
        "first_native_step": int(records[0]["native_step"]),
        "last_native_step": int(records[-1]["native_step"]),
        "signed_energy_kwh": energy,
        "deadline_bound_falsifications": [],
        "record_completeness_verified": True,
        "exact_native_replay_verified": True,
        "acceptance_flags_verified": True,
    }
