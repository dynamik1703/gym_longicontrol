"""Exclusive, exactly-once executor for the frozen V2 transaction."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmarks.hybrid_rl_execution.evaluation import evaluate_seed

from .constants import (
    AUTHORIZATION,
    EXPECTED_ACTOR_SHA256,
    PAIR_LEDGER,
    POLICY_SEEDS,
    SPLIT_LEDGER,
    TRANSACTION_LEASE,
    V2_ID,
    V2_ORDERED_SHA256,
    V2_TRANSACTION_ID,
)
from .integrity import (
    actor_parameter_sha256,
    controller_identities,
    load_frozen_policies,
    verify_frozen_scientific_sources,
    verify_incident_evidence,
)
from .lease import TransactionLease
from .ledger import (
    atomic_json,
    canonical_sha256,
    load_json,
    mark_completed,
    mark_interrupted,
    mark_started,
    validate_pair_ledger,
    validate_registration,
)
from .split import verify_regeneration


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_v2_authorized(path: Path = AUTHORIZATION) -> dict[str, Any]:
    authorization = load_json(path)
    if authorization.get("study") != V2_ID:
        raise PermissionError("only the distinct V2 authorization contract is valid")
    if authorization.get("v1_state") != "INCIDENT_ABORTED":
        raise PermissionError("V1 must remain permanently incident-aborted")
    if authorization.get("preparation_lock", True):
        raise PermissionError("V2 preparation lock blocks evaluation before reset")
    if not authorization.get("v2_validation_authorized"):
        raise PermissionError("V2 final evaluation is not authorized")
    if not authorization.get("v2_validation_enabled"):
        raise PermissionError("V2 final evaluation is not enabled")
    return authorization


def _exclusive_failure_mode(outcome: Mapping[str, Any]) -> str:
    if outcome["feasible"]:
        return "feasible"
    if not outcome["completed"]:
        return "incomplete"
    deadline = not outcome["deadline_compliant"]
    speed = not outcome["speed_compliant"]
    if deadline and speed:
        return "deadline_and_speed"
    if deadline:
        return "deadline_only"
    if speed:
        return "speed_only"
    return "other"


def validate_and_enrich_outcome(
    identity: Mapping[str, Any], outcome: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate frozen evaluator output and add provenance-only derived fields."""

    required = {
        "completed",
        "travel_time_s",
        "energy_kwh",
        "speed_violation_count",
        "max_speed_violation_m_s",
        "integrated_speed_violation_m",
        "feasible",
        "feasible_energy_kwh",
        "deadline_compliant",
        "speed_compliant",
        "macro_decision_count",
        "target_action",
        "target_speed_m_s",
        "proposal_count",
        "override_count",
        "override_fraction",
        "native_action_influence_count",
        "native_action_influence_fraction",
        "intervention_reasons",
        "deadline_bound_falsifications",
    }
    missing = sorted(required - set(outcome))
    if missing:
        raise RuntimeError(f"V2 evaluator result is incomplete: {missing}")
    normalized = dict(outcome)
    completed = bool(normalized["completed"])
    deadline = float(normalized["travel_time_s"]) <= 140.0
    speed = float(normalized["max_speed_violation_m_s"]) <= 0.0
    feasible = completed and deadline and speed
    if (
        normalized["deadline_compliant"] != deadline
        or normalized["speed_compliant"] != speed
        or normalized["feasible"] != feasible
    ):
        raise RuntimeError("V2 feasibility fields are internally inconsistent")
    expected_energy = normalized["energy_kwh"] if feasible else None
    if normalized["feasible_energy_kwh"] != expected_energy:
        raise RuntimeError("V2 feasible-only energy field is inconsistent")
    targets = list(normalized["target_speed_m_s"])
    actions = list(normalized["target_action"])
    if len(targets) != len(actions):
        raise RuntimeError("V2 target action and speed traces differ in length")
    proposals = int(normalized["proposal_count"])
    overrides = int(normalized["override_count"])
    influenced = int(normalized["native_action_influence_count"])
    if proposals < 0 or not (
        0 <= overrides <= proposals and 0 <= influenced <= proposals
    ):
        raise RuntimeError("V2 Shield counts are invalid")
    expected_override = overrides / proposals if proposals else 0.0
    expected_influence = influenced / proposals if proposals else 0.0
    if (
        float(normalized["override_fraction"]) != expected_override
        or float(normalized["native_action_influence_fraction"]) != expected_influence
    ):
        raise RuntimeError("V2 Shield fractions are inconsistent")
    normalized.update(
        {
            "controller_id": identity["controller_id"],
            "numeric_seed": int(identity["numeric_seed"]),
            "route_sha256": identity["route_sha256"],
            "full_task_sha256": identity["full_task_sha256"],
            "signed_net_energy_kwh": normalized["energy_kwh"],
            "exclusive_failure_mode": _exclusive_failure_mode(normalized),
            # Completed boundary crossings are clipped to exactly 1000 m.
            "final_position_m": 1000.0 if completed else None,
            "final_position_source": (
                "completed_route_boundary_invariant"
                if completed
                else "unavailable_from_frozen_evaluator"
            ),
            "target_speed_distribution_m_s": (
                {
                    "count": len(targets),
                    "minimum": min(targets),
                    "mean": fmean(targets),
                    "maximum": max(targets),
                }
                if targets
                else {"count": 0, "minimum": None, "mean": None, "maximum": None}
            ),
            "result_validation": "COMPLETE_FINITE_AND_INTERNALLY_CONSISTENT",
        }
    )
    canonical_sha256(normalized)  # Reject NaN/Infinity and ensure serialization.
    return normalized


def execute_next_registered_pair(
    *,
    ledger_path: Path,
    authorization_path: Path,
    lease: TransactionLease | None,
    context: Any,
    evaluator: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
) -> dict[str, Any]:
    """Execute one registered pair under an already-owned transaction lease."""

    require_v2_authorized(authorization_path)
    if lease is None:
        raise PermissionError("direct V2 runner entry requires lease ownership")
    lease.assert_owned()
    ledger = load_json(ledger_path)
    validate_pair_ledger(ledger)
    if ledger["run_transaction_id"] != lease.transaction_id:
        raise PermissionError("V2 lease does not own this registered transaction")
    if any(
        pair["state"] in {"STARTED", "INTERRUPTED_UNRECOVERABLE"}
        for pair in ledger["pairs"]
    ):
        raise RuntimeError("V2 integrity stop: opened pair cannot be repeated")
    pending = [pair for pair in ledger["pairs"] if pair["state"] == "PENDING"]
    if not pending:
        raise RuntimeError("no untouched V2 pair remains")
    pair = pending[0]
    pair_hash = pair["pair_sha256"]
    lease.assert_owned()
    mark_started(ledger_path, pair_hash, started_at_utc=utc_now())
    lease.assert_owned()
    try:
        outcome = evaluator(pair["identity"], context)
        normalized = validate_and_enrich_outcome(pair["identity"], outcome)
    except BaseException as error:
        mark_interrupted(
            ledger_path,
            pair_hash,
            reason=f"{type(error).__name__}: {error}",
            interrupted_at_utc=utc_now(),
        )
        raise
    lease.assert_owned()
    completed = mark_completed(
        ledger_path,
        pair_hash,
        result=normalized,
        completed_at_utc=utc_now(),
    )
    persisted = load_json(ledger_path)["pairs"][pair["identity"]["pair_ordinal"]]
    if persisted != completed or persisted["state"] != "COMPLETED":
        raise RuntimeError("V2 COMPLETED state did not survive readback")
    return completed


def execute_registered_transaction(
    *,
    ledger_path: Path,
    authorization_path: Path,
    lease_path: Path,
    preflight: Callable[[], Any],
    evaluator: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
    on_pairs_completed: Callable[[TransactionLease], Any] | None = None,
    progress: Callable[[Mapping[str, int]], Any] | None = None,
) -> dict[str, Any]:
    """Run exactly the pre-registered transaction with one permanent lease."""

    require_v2_authorized(authorization_path)
    context = preflight()
    with TransactionLease(lease_path, V2_TRANSACTION_ID) as lease:
        while True:
            lease.assert_owned()
            ledger = load_json(ledger_path)
            validate_pair_ledger(ledger)
            counts = Counter(pair["state"] for pair in ledger["pairs"])
            if counts["INTERRUPTED_UNRECOVERABLE"] or counts["STARTED"]:
                raise RuntimeError("V2 transaction contains an unrecoverable open pair")
            if counts["PENDING"] == 0:
                break
            execute_next_registered_pair(
                ledger_path=ledger_path,
                authorization_path=authorization_path,
                lease=lease,
                context=context,
                evaluator=evaluator,
            )
            if progress is not None:
                persisted = load_json(ledger_path)
                progress(Counter(pair["state"] for pair in persisted["pairs"]))
        final = load_json(ledger_path)
        validate_pair_ledger(final)
        final_counts = Counter(pair["state"] for pair in final["pairs"])
        if final["state"] != "COMPLETED_SEALED" or final_counts != {"COMPLETED": 180}:
            raise RuntimeError("V2 transaction did not reach its exact terminal state")
        if on_pairs_completed is not None:
            on_pairs_completed(lease)
        lease.seal(completed_pair_count=180)
    return load_json(ledger_path)


def _execution_preflight() -> dict[str, Any]:
    incident = verify_incident_evidence()
    sources = verify_frozen_scientific_sources()
    if not incident["passed"] or not sources["passed"]:
        raise RuntimeError("frozen incident or scientific source integrity failed")
    split = verify_regeneration(SPLIT_LEDGER)
    if split["ordered_sha256"] != V2_ORDERED_SHA256:
        raise RuntimeError("V2 split hash changed")
    split_payload = load_json(SPLIT_LEDGER)
    ledger = load_json(PAIR_LEDGER)
    validate_pair_ledger(ledger)
    if ledger["split_sha256"] != split["ordered_sha256"]:
        raise RuntimeError("V2 pair ledger is bound to another split")
    if ledger["run_transaction_id"] != V2_TRANSACTION_ID:
        raise RuntimeError("V2 transaction ID changed")
    if Counter(pair["state"] for pair in ledger["pairs"]) != {"PENDING": 180}:
        raise RuntimeError("V2 must begin from exactly 180 pending pairs")
    policies, agents = load_frozen_policies(return_agents=True)
    validate_registration(
        ledger,
        split_payload,
        controller_identities=controller_identities(policies),
    )
    return {
        "policies": policies,
        "agents": agents,
        "incident": incident,
        "scientific_sources": sources,
        "split": split,
    }


def _single_arm_evaluator(
    identity: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate exactly one already-registered pair with frozen semantics."""

    controller_id = str(identity["controller_id"])
    numeric_seed = int(identity["numeric_seed"])
    if controller_id == "CAUSAL60":
        return evaluate_seed(numeric_seed=numeric_seed, arm="causal")
    if controller_id == "TIME_AWARE":
        return evaluate_seed(numeric_seed=numeric_seed, arm="time_aware")
    if controller_id not in POLICY_SEEDS:
        raise RuntimeError("unregistered controller identity")
    agent = context["agents"][controller_id]
    before_updates = agent.update_count
    before_actor = actor_parameter_sha256(agent.state_dict())
    outcome = evaluate_seed(
        numeric_seed=numeric_seed,
        arm="hybrid_rl",
        agent=agent,
    )
    after_actor = actor_parameter_sha256(agent.state_dict())
    if agent.update_count != before_updates:
        raise RuntimeError("evaluation modified learner update count")
    if (
        before_actor != after_actor
        or after_actor != EXPECTED_ACTOR_SHA256[controller_id]
    ):
        raise RuntimeError("evaluation modified frozen actor")
    return outcome


def seal_v2_authorization(lease: TransactionLease) -> dict[str, Any]:
    lease.assert_owned()
    authorization = require_v2_authorized(AUTHORIZATION)
    authorization.update(
        {
            "v2_state": "COMPLETED_SEALED",
            "v2_validation_authorized": False,
            "v2_validation_enabled": False,
            "sealed_at_utc": utc_now(),
            "sealed_transaction_id": V2_TRANSACTION_ID,
            "paper_authorized": False,
            "paper_state": "SEALED_UNINSPECTED",
            "paper_episodes": 0,
            "paper_content_fingerprints": 0,
        }
    )
    atomic_json(AUTHORIZATION, authorization)
    persisted = load_json(AUTHORIZATION)
    if persisted != authorization:
        raise RuntimeError("V2 authorization seal did not survive readback")
    return persisted


def _print_progress(counts: Mapping[str, int]) -> None:
    print(
        "HYBRID_RL_VALIDATION_V2 "
        f"completed={counts.get('COMPLETED', 0)}/180 "
        f"pending={counts.get('PENDING', 0)} "
        f"started={counts.get('STARTED', 0)} "
        f"unrecoverable={counts.get('INTERRUPTED_UNRECOVERABLE', 0)} "
        "paper=SEALED_UNINSPECTED",
        flush=True,
    )


def execute_v2_transaction() -> dict[str, Any]:
    """The only production entry point for the complete V2 matrix."""

    ledger = execute_registered_transaction(
        ledger_path=PAIR_LEDGER,
        authorization_path=AUTHORIZATION,
        lease_path=TRANSACTION_LEASE,
        preflight=_execution_preflight,
        evaluator=_single_arm_evaluator,
        on_pairs_completed=seal_v2_authorization,
        progress=_print_progress,
    )
    from .analysis import write_final_artifacts

    write_final_artifacts()
    return ledger


def main() -> None:
    result = execute_v2_transaction()
    print(
        json.dumps(
            {
                "study": V2_ID,
                "state": result["state"],
                "performance_episode_count": result["performance_episode_count"],
                "transaction_id": result["run_transaction_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
