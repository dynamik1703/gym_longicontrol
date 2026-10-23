"""Exactly-once Diagnostic evaluation; Validation and Paper are absent."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.causal_mpc_v2.controller import PriceMPCController
from benchmarks.causal_mpc_v2.fixed_price import FixedPrice
from benchmarks.hybrid_rl_design.baseline import TimeAwareEnergyStrategy
from benchmarks.hybrid_rl_design.controller import HybridTargetSpeedController
from benchmarks.hybrid_rl_execution.agent import ConstrainedSAC as HistoricalHybridSAC
from benchmarks.hybrid_rl_validation_v2.integrity import load_frozen_policies
from benchmarks.physical_feasibility_reference.causal_controller import (
    ObservationLimitedController,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .checkpointing import file_sha256, load_checkpoint
from .chunk_decoder import method_spec
from .execution import DEFAULT_OUTPUT_ROOT, RunLease, atomic_json, utc_now
from .integrity import (
    ROOT,
    read_split,
    require_diagnostic_authorized,
    verify_execution_preconditions,
)
from .residual_controller import FeedbackResidualController
from .sac_agent import ConstrainedResidualSAC
from .split import verify_runtime_row

CANONICAL_TASK = TaskSpecification(140.0, 0.0)
COMPARATOR_ORDER = (
    "CAUSAL60",
    "TIME_AWARE",
    "FIXED_PRICE_V2",
    "ADAPTIVE_PRICE_V2",
    "H11",
    "H29",
    "H47",
)
LEARNED_STEPS = (100_000, 200_000, 300_000)


@dataclass(frozen=True)
class EvaluationArtifact:
    outcome: dict[str, Any]
    native_evidence: list[dict[str, Any]]


def _environment() -> gym.Env:
    return gym.wrappers.TimeLimit(StochasticTrack(), max_episode_steps=1_800)


def _common_outcome(
    *,
    info: dict[str, Any],
    native_evidence: list[dict[str, Any]],
    actor_actions: list[list[float]],
    decisions: list[dict[str, Any]],
    decision_kind: str,
) -> dict[str, Any]:
    metrics = EpisodeMetrics(**info["episode_metrics"])
    feasible = is_feasible(metrics, CANONICAL_TASK)
    if decision_kind == "residual":
        overrides = sum(not row["residual_accepted"] for row in decisions)
        influenced = sum(row["residual_influence"] > 1.0e-15 for row in decisions)
        reasons = Counter(
            row["cancellation_reason"] or "RESIDUAL_ACCEPTED" for row in decisions
        )
        cancelled_chunk_ids = {
            row["chunk_id"] for row in decisions if row["chunk_cancelled"]
        }
        chunk_ids = {row["chunk_id"] for row in decisions}
        full_chunk_ids = {
            chunk_id
            for chunk_id in chunk_ids
            if chunk_id not in cancelled_chunk_ids
            and sum(row["chunk_id"] == chunk_id for row in decisions)
            == len(actor_actions[chunk_id]) * 10
        }
        cancellation_reasons = dict(sorted(reasons.items()))
    else:
        overrides = sum(bool(row.get("overridden", False)) for row in decisions)
        influenced = sum(
            float(row.get("native_action_influence", 0.0)) > 1.0e-15
            for row in decisions
        )
        cancellation_reasons = dict(
            sorted(
                Counter(
                    row.get("intervention_reason", "NONE") for row in decisions
                ).items()
            )
        )
        cancelled_chunk_ids = set()
        full_chunk_ids = set()
        chunk_ids = set()
    executed_actions = [float(row["executed_action"]) for row in native_evidence]
    influenced_energy = sum(
        float(row["step_energy_kwh"])
        for row in native_evidence
        if float(row.get("native_action_influence", 0.0)) > 1.0e-15
    )
    actor_array = np.asarray(actor_actions, dtype=np.float64)
    correlations: list[list[float]] | None = None
    if (
        actor_array.ndim == 2
        and actor_array.shape[1] == 3
        and actor_array.shape[0] >= 2
    ):
        matrix = np.corrcoef(actor_array, rowvar=False)
        if np.isfinite(matrix).all():
            correlations = matrix.tolist()
    failure_reasons = []
    if not metrics.completed:
        failure_reasons.append("INCOMPLETE")
    if metrics.travel_time_s > 140.0:
        failure_reasons.append("DEADLINE")
    if metrics.max_speed_violation_m_s > 0.0:
        failure_reasons.append("SPEED")
    return {
        **asdict(metrics),
        "feasible": feasible,
        "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
        "deadline_compliant": metrics.travel_time_s <= 140.0,
        "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
        "failure_reasons": failure_reasons,
        "actor_decision_count": len(actor_actions),
        "actor_actions": actor_actions,
        "proposal_count": len(decisions),
        "override_count": overrides,
        "native_action_influence_count": influenced,
        "intervention_or_cancellation_reasons": cancellation_reasons,
        "chunk_count": len(chunk_ids),
        "full_chunk_count": len(full_chunk_ids),
        "cancelled_chunk_count": len(cancelled_chunk_ids),
        "full_chunk_fraction": (
            len(full_chunk_ids) / len(chunk_ids) if chunk_ids else None
        ),
        "actor_component_correlation": correlations,
        "executed_action_distribution": {
            "braking": sum(value < -1.0e-12 for value in executed_actions),
            "coasting": sum(abs(value) <= 1.0e-12 for value in executed_actions),
            "traction": sum(value > 1.0e-12 for value in executed_actions),
        },
        "signed_energy_on_influenced_steps_kwh": influenced_energy,
    }


def evaluate_learned_seed(
    *,
    numeric_seed: int,
    method: str,
    agent: ConstrainedResidualSAC,
) -> EvaluationArtifact:
    spec = method_spec(method)
    environment = _environment()
    controller = FeedbackResidualController(method)
    evidence: list[dict[str, Any]] = []
    actor_actions: list[list[float]] = []
    decisions: list[dict[str, Any]] = []
    try:
        observation, _ = environment.reset(seed=int(numeric_seed))
        controller.reset()
        ended = False
        step = 0
        while not ended:
            actor_observation = controller.prepare_native_step(observation)
            if actor_observation is not None:
                actor_action = agent.act(actor_observation, deterministic=True)
                if actor_action.shape != (spec.action_dim,):
                    raise RuntimeError("deterministic actor dimension changed")
                actor_actions.append(actor_action.tolist())
                controller.begin_chunk(actor_action)
            pre_state = asdict(controller.state)
            native_action = controller.action(observation)
            decision = controller.last_decision
            if decision is None:
                raise RuntimeError("residual decision is missing")
            next_observation, _, terminated, truncated, info = environment.step(
                [native_action]
            )
            controller.observe_executed_action(native_action)
            decision_row = decision.as_dict()
            decisions.append(decision_row)
            evidence.append(
                {
                    "native_step": step,
                    "pre_state": pre_state,
                    "ordinary_observation": np.asarray(observation).tolist(),
                    "actor_action": list(decision.actor_action),
                    "mpc_proposed_action": decision.mpc_proposed_action,
                    "requested_residual": decision.requested_residual,
                    "combined_proposal": decision.combined_proposal,
                    "executed_action": float(native_action),
                    "native_action_influence": decision.residual_influence,
                    "step_energy_kwh": float(info["step_energy_kwh"]),
                    "post_state": {
                        "position_m": float(info["position_m"]),
                        "velocity_m_s": float(info["velocity_m_s"]),
                        "acceleration_m_s2": float(info["acceleration_m_s2"]),
                        "elapsed_time_s": float(info["elapsed_time_s"]),
                        "total_energy_kwh": float(info["total_energy_kwh"]),
                        "speed_limit_m_s": float(info["speed_limit_m_s"]),
                    },
                    "residual_decision": decision_row,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            observation = np.asarray(next_observation, dtype=np.float64)
            step += 1
            ended = bool(terminated or truncated)
        outcome = _common_outcome(
            info=info,
            native_evidence=evidence,
            actor_actions=actor_actions,
            decisions=decisions,
            decision_kind="residual",
        )
        outcome["actual_macro_transition_lengths"] = [
            min(spec.actor_interval_native_steps, len(evidence) - start)
            for start in range(0, len(evidence), spec.actor_interval_native_steps)
        ]
        return EvaluationArtifact(outcome, evidence)
    finally:
        environment.close()


def evaluate_frozen_seed(
    *,
    numeric_seed: int,
    controller_id: str,
    hybrid_agents: dict[str, HistoricalHybridSAC],
) -> EvaluationArtifact:
    environment = _environment()
    evidence: list[dict[str, Any]] = []
    actor_actions: list[list[float]] = []
    decisions: list[dict[str, Any]] = []
    if controller_id == "CAUSAL60":
        controller: Any = ObservationLimitedController()
        strategy: Any = None
        kind = "causal"
    elif controller_id in ("TIME_AWARE", "H11", "H29", "H47"):
        controller = HybridTargetSpeedController(strategy_enabled=True)
        strategy = (
            TimeAwareEnergyStrategy()
            if controller_id == "TIME_AWARE"
            else hybrid_agents[controller_id]
        )
        kind = "hybrid"
    elif controller_id in ("FIXED_PRICE_V2", "ADAPTIVE_PRICE_V2"):
        controller = PriceMPCController()
        strategy = (
            FixedPrice(0.0)
            if controller_id == "FIXED_PRICE_V2"
            else AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
        )
        kind = "price"
    else:
        raise ValueError(f"unknown frozen controller: {controller_id}")
    try:
        observation, _ = environment.reset(seed=int(numeric_seed))
        controller.reset()
        if strategy is not None and hasattr(strategy, "reset"):
            strategy.reset()
        held: float | None = None
        step = 0
        ended = False
        while not ended:
            if kind in ("hybrid", "price") and step % 10 == 0:
                strategy_observation = controller.prepare_strategy_step(observation)
                if isinstance(strategy, TimeAwareEnergyStrategy):
                    held = strategy.action(
                        controller.state,
                        step,
                        route_length_m=controller.config.track_length_m,
                    )
                elif isinstance(strategy, HistoricalHybridSAC):
                    held = strategy.act(strategy_observation, deterministic=True)
                else:
                    held = strategy.action(strategy_observation)
                actor_actions.append([float(held)])
            pre_state = asdict(controller.state)
            if kind == "causal":
                native_action = controller.action(observation)
                decision_row: dict[str, Any] = {}
            else:
                if held is None:
                    raise RuntimeError("frozen strategy action is missing")
                native_action = controller.action(observation, held)
                decision_row = controller.last_decision.as_dict()
                decisions.append(decision_row)
            next_observation, _, terminated, truncated, info = environment.step(
                [native_action]
            )
            controller.observe_executed_action(native_action)
            influence = float(decision_row.get("native_action_influence", 0.0))
            evidence.append(
                {
                    "native_step": step,
                    "pre_state": pre_state,
                    "ordinary_observation": np.asarray(observation).tolist(),
                    "strategy_action": held,
                    "executed_action": float(native_action),
                    "native_action_influence": influence,
                    "step_energy_kwh": float(info["step_energy_kwh"]),
                    "post_state": {
                        "position_m": float(info["position_m"]),
                        "velocity_m_s": float(info["velocity_m_s"]),
                        "acceleration_m_s2": float(info["acceleration_m_s2"]),
                        "elapsed_time_s": float(info["elapsed_time_s"]),
                        "total_energy_kwh": float(info["total_energy_kwh"]),
                        "speed_limit_m_s": float(info["speed_limit_m_s"]),
                    },
                    "controller_decision": decision_row,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            observation = np.asarray(next_observation, dtype=np.float64)
            step += 1
            ended = bool(terminated or truncated)
        outcome = _common_outcome(
            info=info,
            native_evidence=evidence,
            actor_actions=actor_actions,
            decisions=decisions,
            decision_kind=kind,
        )
        return EvaluationArtifact(outcome, evidence)
    finally:
        environment.close()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_evidence(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
            for row in rows:
                stream.write(
                    json.dumps(
                        row,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "row_count": len(rows),
        "size_bytes": path.stat().st_size,
    }


def _diagnostic_rows() -> list[dict[str, Any]]:
    split = read_split()["diagnostic_panel"]
    if split["state"] not in ("FROZEN_UNOPENED", "OPENED_DIAGNOSTIC"):
        raise RuntimeError("Diagnostic split state is invalid")
    return [dict(row) for row in split["accepted_tasks"]]


def _plan(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = _diagnostic_rows()
    plan: list[dict[str, Any]] = []
    for controller_id in COMPARATOR_ORDER:
        for task in tasks:
            plan.append(
                {
                    "pair_id": _canonical_sha256(
                        {
                            "controller_id": controller_id,
                            "checkpoint_step": None,
                            "numeric_seed": task["numeric_seed"],
                        }
                    ),
                    "controller_id": controller_id,
                    "kind": "FROZEN_COMPARATOR",
                    "method": None,
                    "training_seed": None,
                    "checkpoint_step": None,
                    "numeric_seed": task["numeric_seed"],
                    "route_sha256": task["route_sha256"],
                    "full_task_sha256": task["full_task_sha256"],
                    "state": "NOT_STARTED",
                }
            )
    for method in ("CHUNK3_RESIDUAL_SAC", "RESIDUAL1S_SAC"):
        for training_seed in (11, 29, 47):
            run = manifest["runs"][f"{method}__seed-{training_seed}"]
            checkpoints = {
                row["native_transition_count"]: row
                for row in run["scientific_checkpoints"]
            }
            for checkpoint_step in LEARNED_STEPS:
                checkpoint = checkpoints[checkpoint_step]
                controller_id = f"{method}__S{training_seed}__N{checkpoint_step}"
                for task in tasks:
                    plan.append(
                        {
                            "pair_id": _canonical_sha256(
                                {
                                    "controller_id": controller_id,
                                    "checkpoint_sha256": checkpoint["sha256"],
                                    "numeric_seed": task["numeric_seed"],
                                }
                            ),
                            "controller_id": controller_id,
                            "kind": "LEARNED_TREATMENT",
                            "method": method,
                            "training_seed": training_seed,
                            "checkpoint_step": checkpoint_step,
                            "checkpoint_path": checkpoint["path"],
                            "checkpoint_sha256": checkpoint["sha256"],
                            "numeric_seed": task["numeric_seed"],
                            "route_sha256": task["route_sha256"],
                            "full_task_sha256": task["full_task_sha256"],
                            "state": "NOT_STARTED",
                        }
                    )
    return plan


def _load_learned_agent(
    identity: dict[str, Any], manifest: dict[str, Any]
) -> ConstrainedResidualSAC:
    provenance = manifest["provenance"]
    run = manifest["runs"][f"{identity['method']}__seed-{identity['training_seed']}"]
    attempt_id = run["attempts"][0]["attempt_id"]
    loaded = load_checkpoint(
        identity["checkpoint_path"],
        expected_sha256=identity["checkpoint_sha256"],
        expected_configuration_sha256=provenance["canonical_sha256"],
        expected_source_hashes=provenance["source_hashes"],
        expected_attempt_id=attempt_id,
    )
    if loaded.runtime["native_transition_count"] != identity["checkpoint_step"]:
        raise RuntimeError("learned checkpoint step changed")
    agent = ConstrainedResidualSAC(identity["method"], identity["training_seed"])
    agent.load_state_dict(loaded.agent_state)
    loaded.environment.close()
    return agent


def execute_diagnostic(repository: Path, output_root: Path) -> Path:
    if repository.resolve() != ROOT.resolve():
        raise RuntimeError("Diagnostic repository differs from study root")
    require_diagnostic_authorized()
    provenance = verify_execution_preconditions()
    root = output_root if output_root.is_absolute() else repository / output_root
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["provenance"] != provenance:
        raise RuntimeError("training provenance changed before Diagnostic")
    if manifest["state"] not in (
        "TRAINING_COMPLETED_DIAGNOSTIC_PENDING",
        "DIAGNOSTIC_RUNNING",
    ):
        raise RuntimeError("all six training runs must complete before Diagnostic")
    diagnostic_root = root / "diagnostic"
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    ledger_path = diagnostic_root / "pair-ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if any(row["state"] == "RUNNING" for row in ledger["pairs"]):
            raise RuntimeError("ambiguous interrupted Diagnostic pair; integrity stop")
    else:
        ledger = {
            "schema_version": 1,
            "study": "CHUNK_RL_DIAGNOSTIC_V1",
            "state": "NOT_STARTED",
            "created_at": utc_now(),
            "pairs": _plan(manifest),
            "validation_performance_episodes": 0,
            "paper_performance_episodes": 0,
        }
        atomic_json(ledger_path, ledger)
    _, hybrid_agents = load_frozen_policies(return_agents=True)
    learned_cache: dict[str, ConstrainedResidualSAC] = {}
    with RunLease(diagnostic_root):
        ledger["state"] = "RUNNING"
        manifest["state"] = "DIAGNOSTIC_RUNNING"
        atomic_json(ledger_path, ledger)
        atomic_json(manifest_path, manifest)
        for identity in ledger["pairs"]:
            if identity["state"] == "COMPLETED":
                result_path = Path(identity["result_path"])
                if file_sha256(result_path) != identity["result_sha256"]:
                    raise RuntimeError("completed Diagnostic pair record changed")
                continue
            if identity["state"] != "NOT_STARTED":
                raise RuntimeError("Diagnostic pair state is not exactly-once safe")
            task_row = {
                key: identity[key]
                for key in ("numeric_seed", "route_sha256", "full_task_sha256")
            }
            verify_runtime_row(task_row)
            identity["state"] = "RUNNING"
            identity["started_at"] = utc_now()
            atomic_json(ledger_path, ledger)
            if identity["kind"] == "FROZEN_COMPARATOR":
                artifact = evaluate_frozen_seed(
                    numeric_seed=identity["numeric_seed"],
                    controller_id=identity["controller_id"],
                    hybrid_agents=hybrid_agents,
                )
            else:
                controller_id = identity["controller_id"]
                if controller_id not in learned_cache:
                    learned_cache[controller_id] = _load_learned_agent(
                        identity, manifest
                    )
                artifact = evaluate_learned_seed(
                    numeric_seed=identity["numeric_seed"],
                    method=identity["method"],
                    agent=learned_cache[controller_id],
                )
            pair_directory = diagnostic_root / "pairs" / identity["pair_id"]
            pair_directory.mkdir(parents=True, exist_ok=False)
            evidence = _atomic_evidence(
                pair_directory / "native-evidence.jsonl.gz",
                artifact.native_evidence,
            )
            result = {
                "identity": {
                    key: value
                    for key, value in identity.items()
                    if key not in ("state", "started_at")
                },
                "outcome": artifact.outcome,
                "native_evidence": evidence,
            }
            result_path = atomic_json(pair_directory / "result.json", result)
            identity.update(
                {
                    "state": "COMPLETED",
                    "completed_at": utc_now(),
                    "result_path": str(result_path),
                    "result_sha256": file_sha256(result_path),
                    "evidence_sha256": evidence["sha256"],
                    "evidence_rows": evidence["row_count"],
                }
            )
            atomic_json(ledger_path, ledger)
        ledger["state"] = "COMPLETED"
        ledger["completed_at"] = utc_now()
        manifest["state"] = "COMPLETED"
        manifest["diagnostic"] = {
            "state": "COMPLETED",
            "episodes": len(ledger["pairs"]),
            "pair_ledger": str(ledger_path),
            "completed_at": utc_now(),
        }
        atomic_json(ledger_path, ledger)
        atomic_json(manifest_path, manifest)
    return ledger_path


def _load_completed_outcomes(ledger_path: Path) -> list[dict[str, Any]]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger["state"] != "COMPLETED":
        raise RuntimeError("Diagnostic pair ledger is incomplete")
    outcomes = []
    for identity in ledger["pairs"]:
        if identity["state"] != "COMPLETED":
            raise RuntimeError("Diagnostic pair is incomplete")
        result_path = Path(identity["result_path"])
        if file_sha256(result_path) != identity["result_sha256"]:
            raise RuntimeError("Diagnostic result hash changed")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        outcomes.append({**identity, **result["outcome"]})
    return outcomes


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["feasible"]]
    energies = [float(row["energy_kwh"]) for row in feasible]
    return {
        "episodes": len(rows),
        "rsr": len(feasible),
        "completion_count": sum(bool(row["completed"]) for row in rows),
        "deadline_compliant_count": sum(row["deadline_compliant"] for row in rows),
        "speed_compliant_count": sum(row["speed_compliant"] for row in rows),
        "failure_identities": [
            {
                "numeric_seed": row["numeric_seed"],
                "reasons": row["failure_reasons"],
            }
            for row in rows
            if not row["feasible"]
        ],
        "mean_feasible_energy_kwh": statistics.fmean(energies) if energies else None,
        "median_feasible_energy_kwh": statistics.median(energies) if energies else None,
        "mean_travel_time_s": statistics.fmean(
            float(row["travel_time_s"]) for row in rows
        ),
        "proposal_count": sum(int(row["proposal_count"]) for row in rows),
        "native_action_influence_count": sum(
            int(row["native_action_influence_count"]) for row in rows
        ),
        "cancelled_chunk_count": sum(int(row["cancelled_chunk_count"]) for row in rows),
    }


def _paired(
    baseline: list[dict[str, Any]], treatment: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_by_seed = {row["numeric_seed"]: row for row in baseline}
    pairs = [
        (baseline_by_seed[row["numeric_seed"]], row)
        for row in treatment
        if row["numeric_seed"] in baseline_by_seed
        and baseline_by_seed[row["numeric_seed"]]["feasible"]
        and row["feasible"]
    ]
    savings = [
        100.0 * (base["energy_kwh"] - target["energy_kwh"]) / base["energy_kwh"]
        for base, target in pairs
    ]
    differences = [base["energy_kwh"] - target["energy_kwh"] for base, target in pairs]
    return {
        "pair_count": len(pairs),
        "mean_percent_saving": statistics.fmean(savings) if savings else None,
        "median_percent_saving": statistics.median(savings) if savings else None,
        "range_percent_saving": (
            [min(savings), max(savings)] if savings else [None, None]
        ),
        "mean_absolute_energy_difference_kwh": (
            statistics.fmean(differences) if differences else None
        ),
        "per_track_percent_saving": savings,
    }


def _finite_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return statistics.fmean(finite) if len(finite) == len(values) and finite else None


def analyze_diagnostic(ledger_path: Path, destination: Path) -> Path:
    outcomes = _load_completed_outcomes(ledger_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in outcomes:
        grouped.setdefault(row["controller_id"], []).append(row)
    summaries = {name: _summary(rows) for name, rows in grouped.items()}
    comparisons: dict[str, Any] = {}
    adaptive = grouped["ADAPTIVE_PRICE_V2"]
    chunk_seed_means = []
    residual_seed_means = []
    chunk_vs_residual_seed_means = []
    chunk_vs_adaptive_seed_means = []
    for seed in (11, 29, 47):
        chunk_id = f"CHUNK3_RESIDUAL_SAC__S{seed}__N300000"
        residual_id = f"RESIDUAL1S_SAC__S{seed}__N300000"
        chunk_rows = grouped[chunk_id]
        residual_rows = grouped[residual_id]
        chunk_summary = summaries[chunk_id]
        residual_summary = summaries[residual_id]
        chunk_seed_means.append(chunk_summary["mean_feasible_energy_kwh"])
        residual_seed_means.append(residual_summary["mean_feasible_energy_kwh"])
        versus_residual = _paired(residual_rows, chunk_rows)
        versus_adaptive = _paired(adaptive, chunk_rows)
        comparisons[f"chunk_seed_{seed}_vs_residual_seed_{seed}"] = versus_residual
        comparisons[f"chunk_seed_{seed}_vs_adaptive"] = versus_adaptive
        chunk_vs_residual_seed_means.append(versus_residual["mean_percent_saving"])
        chunk_vs_adaptive_seed_means.append(versus_adaptive["mean_percent_saving"])
    chunk_failures = {
        seed: {
            row["numeric_seed"]
            for row in grouped[f"CHUNK3_RESIDUAL_SAC__S{seed}__N300000"]
            if not row["feasible"]
        }
        for seed in (11, 29, 47)
    }
    residual_failures = {
        seed: {
            row["numeric_seed"]
            for row in grouped[f"RESIDUAL1S_SAC__S{seed}__N300000"]
            if not row["feasible"]
        }
        for seed in (11, 29, 47)
    }
    adaptive_failures = {row["numeric_seed"] for row in adaptive if not row["feasible"]}
    level_b = bool(
        all(not (chunk_failures[s] - residual_failures[s]) for s in (11, 29, 47))
        and all(value is not None for value in chunk_vs_residual_seed_means)
        and statistics.fmean(chunk_vs_residual_seed_means) >= 1.0
    )
    level_c = bool(
        all(not (chunk_failures[s] - adaptive_failures) for s in (11, 29, 47))
        and all(
            value is not None and value >= 3.0 for value in chunk_vs_adaptive_seed_means
        )
    )
    feasibility_regression = any(
        chunk_failures[s] - (residual_failures[s] | adaptive_failures)
        for s in (11, 29, 47)
    )
    if feasibility_regression:
        decision = "CHUNK_RL_FEASIBILITY_REGRESSION"
    elif level_b or level_c:
        decision = "CHUNK_RL_PROMISING_FOR_INDEPENDENT_VALIDATION"
    elif all(
        summaries[f"CHUNK3_RESIDUAL_SAC__S{s}__N300000"]["rsr"] == 36
        for s in (11, 29, 47)
    ):
        decision = "CHUNK_RL_FEASIBLE_WITHOUT_INCREMENTAL_GAIN"
    else:
        decision = "CHUNK_RL_NEGATIVE_RESULT"
    result = {
        "schema_version": 1,
        "panel": "CHUNK_RL_DIAGNOSTIC_V1",
        "episode_count": len(outcomes),
        "summaries": summaries,
        "paired_comparisons": comparisons,
        "method_level": {
            "chunk_mean_feasible_energy_across_seeds_kwh": _finite_mean(
                chunk_seed_means
            ),
            "residual_mean_feasible_energy_across_seeds_kwh": _finite_mean(
                residual_seed_means
            ),
            "chunk_vs_residual_mean_percent_saving_across_seeds": _finite_mean(
                chunk_vs_residual_seed_means
            ),
            "chunk_vs_adaptive_mean_percent_saving_across_seeds": _finite_mean(
                chunk_vs_adaptive_seed_means
            ),
        },
        "success_levels": {"B": level_b, "C": level_c},
        "decision": decision,
        "validation_performance_episodes": 0,
        "paper_performance_episodes": 0,
        "paper_content_fingerprints_constructed": 0,
    }
    atomic_json(destination, result)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("diagnostic")
    execute.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args(argv)
    root = (
        arguments.output_root
        if arguments.output_root.is_absolute()
        else ROOT / arguments.output_root
    )
    if arguments.command == "diagnostic":
        print(execute_diagnostic(ROOT, root))
    else:
        print(
            analyze_diagnostic(
                root / "diagnostic/pair-ledger.json",
                root / "diagnostic/analysis.json",
            )
        )


if __name__ == "__main__":
    main()
