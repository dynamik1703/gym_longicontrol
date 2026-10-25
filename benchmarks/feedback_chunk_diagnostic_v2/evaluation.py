"""Native evaluation adapters and the normalized V2 evidence contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.feedback_chunk_rl_v1.evaluation import (
    EvaluationArtifact,
    evaluate_frozen_seed,
    evaluate_learned_seed,
)
from benchmarks.physical_feasibility_reference.causal_controller import (
    decode_observation,
)

from .constants import C_IDENTITIES, NONLEARNING_CONTROLLERS

FROZEN_ID = {
    "CAUSAL60": "CAUSAL60",
    "TIME_AWARE": "TIME_AWARE",
    "FIXED_PRICE_MPC_V2": "FIXED_PRICE_V2",
    "ADAPTIVE_PRICE_MPC_V2": "ADAPTIVE_PRICE_V2",
    "HYBRID_H11": "H11",
    "HYBRID_H29": "H29",
    "HYBRID_H47": "H47",
}


def applicable(value: Any) -> dict[str, Any]:
    """Represent a value which the controller actually computed."""

    return {"status": "APPLICABLE", "value": value}


def not_applicable(reason: str) -> dict[str, Any]:
    """Represent an inapplicable quantity without inventing a zero."""

    return {"status": "NOT_APPLICABLE", "reason": reason, "value": None}


def _safety_value(decision: dict[str, Any], name: str, reason: str) -> dict[str, Any]:
    if not decision or name not in decision or decision[name] is None:
        return not_applicable(reason)
    return applicable(decision[name])


def _normalized_row(
    row: dict[str, Any], *, controller_id: str, learned: bool
) -> dict[str, Any]:
    observation = np.asarray(row["ordinary_observation"], dtype=np.float64)
    decoded = decode_observation(observation)
    if learned:
        decision = dict(row["residual_decision"])
        proposed_controller_action = float(decision["combined_proposal"])
        proposed_residual = applicable(float(decision["requested_residual"]))
        executed_residual = applicable(
            float(decision["executed_action"] - decision["mpc_executed_action"])
        )
        intervention = applicable(bool(not decision["residual_accepted"]))
        cancellation = applicable(decision["cancellation_reason"] or "NONE")
    else:
        decision = dict(row.get("controller_decision", {}))
        proposed_controller_action = float(
            decision.get("proposed_native_action", row["executed_action"])
        )
        proposed_residual = not_applicable("controller_has_no_residual_action")
        executed_residual = not_applicable("controller_has_no_residual_action")
        if decision:
            intervention = applicable(bool(decision.get("overridden", False)))
            cancellation = applicable(
                decision.get("intervention_reason", "NONE") or "NONE"
            )
        else:
            intervention = not_applicable("controller_has_no_shield_decision")
            cancellation = not_applicable("controller_has_no_shield_decision")
    completion = _safety_value(
        decision, "completion_bound_s", "controller_did_not_compute_completion_bound"
    )
    return {
        "schema_version": 1,
        "controller_id": controller_id,
        "native_step": int(row["native_step"]),
        "causal_pre_state": dict(row["pre_state"]),
        "ordinary_observation": row["ordinary_observation"],
        "visible_speed_limits": {
            "current_m_s": float(decoded["current_limit_m_s"]),
            "future_m_s": [float(value) for value in decoded["future_limits_m_s"]],
            "transition_distances_m": [
                float(value) for value in decoded["distances_m"]
            ],
        },
        "proposed_controller_action": proposed_controller_action,
        "proposed_residual": proposed_residual,
        "executed_residual": executed_residual,
        "final_actuator_action": float(row["executed_action"]),
        "native_action_influence": float(row["native_action_influence"]),
        "signed_step_energy_kwh": float(row["step_energy_kwh"]),
        "endpoint_speed_check": _safety_value(
            decision,
            "endpoint_speed_safe",
            "controller_did_not_compute_endpoint_speed_check",
        ),
        "known_transition_braking_check": _safety_value(
            decision,
            "known_transition_braking_safe",
            "controller_did_not_compute_known_transition_braking_check",
        ),
        "unknown_preview_safety_check": _safety_value(
            decision,
            "unknown_preview_safe",
            "controller_did_not_compute_unknown_preview_check",
        ),
        "deadline_completion_bound_s": completion,
        "viability_flag": _safety_value(
            decision,
            "deadline_viable",
            "controller_did_not_compute_deadline_viability",
        ),
        "shield_intervention": intervention,
        "intervention_or_cancellation_reason": cancellation,
        "actual_successor_state": dict(row["post_state"]),
        "terminated": bool(row["terminated"]),
        "truncated": bool(row["truncated"]),
        "native_decision": decision,
    }


def normalize_artifact(
    artifact: EvaluationArtifact, *, controller_id: str
) -> dict[str, Any]:
    learned = controller_id in C_IDENTITIES
    evidence = [
        _normalized_row(row, controller_id=controller_id, learned=learned)
        for row in artifact.native_evidence
    ]
    return {"outcome": dict(artifact.outcome), "native_evidence": evidence}


def evaluate_controller(
    *,
    numeric_seed: int,
    controller_id: str,
    c_agents: dict[str, Any] | None = None,
    h_agents: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one assignment inside the controller's native worker runtime."""

    if controller_id in C_IDENTITIES:
        if c_agents is None or controller_id not in c_agents:
            raise PermissionError("C checkpoint was not loaded in this worker")
        method, _ = C_IDENTITIES[controller_id]
        artifact = evaluate_learned_seed(
            numeric_seed=numeric_seed,
            method=method,
            agent=c_agents[controller_id],
        )
    elif controller_id in NONLEARNING_CONTROLLERS:
        artifact = evaluate_frozen_seed(
            numeric_seed=numeric_seed,
            controller_id=FROZEN_ID[controller_id],
            hybrid_agents={},
        )
    elif controller_id.startswith("HYBRID_H"):
        if h_agents is None or controller_id not in h_agents:
            raise PermissionError("H checkpoint was not loaded in this worker")
        short = FROZEN_ID[controller_id]
        artifact = evaluate_frozen_seed(
            numeric_seed=numeric_seed,
            controller_id=short,
            hybrid_agents={short: h_agents[controller_id]},
        )
    else:
        raise PermissionError(f"controller is not registered: {controller_id}")
    return normalize_artifact(artifact, controller_id=controller_id)
