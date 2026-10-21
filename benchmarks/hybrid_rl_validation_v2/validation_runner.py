"""Future V2 executor; current preparation authorization always fails closed."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.hybrid_rl_execution.evaluation import evaluate_seed

from .constants import (
    AUTHORIZATION,
    EXPECTED_ACTOR_SHA256,
    PAIR_LEDGER,
    POLICY_SEEDS,
    SPLIT_LEDGER,
    V2_ID,
)
from .integrity import (
    actor_parameter_sha256,
    controller_identities,
    load_frozen_policies,
)
from .ledger import (
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


def execute_next_registered_pair(
    *,
    ledger_path: Path,
    authorization_path: Path,
    preflight: Callable[[], Any],
    evaluator: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
) -> dict[str, Any]:
    """Execute one pair with durable STARTED written before evaluator entry."""

    require_v2_authorized(authorization_path)
    ledger = load_json(ledger_path)
    validate_pair_ledger(ledger)
    if any(
        pair["state"] in {"STARTED", "INTERRUPTED_UNRECOVERABLE"}
        for pair in ledger["pairs"]
    ):
        raise RuntimeError("V2 integrity stop: opened pair cannot be repeated")
    context = preflight()
    pending = [pair for pair in ledger["pairs"] if pair["state"] == "PENDING"]
    if not pending:
        raise RuntimeError("no untouched V2 pair remains")
    pair = pending[0]
    pair_hash = pair["pair_sha256"]
    mark_started(ledger_path, pair_hash, started_at_utc=utc_now())
    try:
        outcome = evaluator(pair["identity"], context)
    except BaseException as error:
        mark_interrupted(
            ledger_path,
            pair_hash,
            reason=f"{type(error).__name__}: {error}",
            interrupted_at_utc=utc_now(),
        )
        raise
    return mark_completed(
        ledger_path,
        pair_hash,
        result=outcome,
        completed_at_utc=utc_now(),
    )


def _future_execution_preflight() -> dict[str, Any]:
    split = verify_regeneration(SPLIT_LEDGER)
    split_payload = load_json(SPLIT_LEDGER)
    ledger = load_json(PAIR_LEDGER)
    validate_pair_ledger(ledger)
    if ledger["split_sha256"] != split["ordered_sha256"]:
        raise RuntimeError("V2 pair ledger is bound to another split")
    policies, agents = load_frozen_policies(return_agents=True)
    validate_registration(
        ledger,
        split_payload,
        controller_identities=controller_identities(policies),
    )
    return {"policies": policies, "agents": agents}


def _future_evaluator(
    identity: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate exactly one already-registered pair; never the old convenience path."""

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


def execute_next_v2_pair() -> dict[str, Any]:
    """Future entry point. The committed preparation contract blocks this call."""

    return execute_next_registered_pair(
        ledger_path=PAIR_LEDGER,
        authorization_path=AUTHORIZATION,
        preflight=_future_execution_preflight,
        evaluator=_future_evaluator,
    )
