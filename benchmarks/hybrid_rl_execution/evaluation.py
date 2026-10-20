"""Actor-only evaluation with strict Diagnostic/Validation/Paper boundaries."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym

from benchmarks.content_split_audit.streams import PAPER_SEEDS
from benchmarks.hybrid_rl_design.baseline import TimeAwareEnergyStrategy
from benchmarks.hybrid_rl_design.controller import (
    HybridTargetSpeedController,
    normalized_action_to_target_speed,
)
from benchmarks.hybrid_rl_design.split import verify_runtime_row
from benchmarks.physical_feasibility_reference.causal_controller import (
    ObservationLimitedController,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .agent import ConstrainedSAC
from .macro_transition import DECISION_INTERVAL_NATIVE_STEPS

HERE = Path(__file__).resolve().parent
DESIGN_SPLIT = HERE.parent / "hybrid_rl_design/split_ledger.json"
CANONICAL = HERE / "canonical.json"
CANONICAL_TASK = TaskSpecification(140.0, 0.0)
EXPECTED_ORDERED_HASHES = {
    "diagnostic": "c8fe41d3ae100d108a6c5175ee180e023cbd431ab3a2e709d815fef66547b784",
    "validation": "f3864fed8725874dc315004375af0d2850dc2d2d56957d52c64d043ca6b3352b",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_frozen_splits_without_opening() -> dict[str, Any]:
    ledger = json.loads(DESIGN_SPLIT.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for name, expected in EXPECTED_ORDERED_HASHES.items():
        split = ledger["evaluation_sets"][name]
        frozen = [
            {
                key: row[key]
                for key in (
                    "numeric_seed",
                    "route_sha256",
                    "full_task_sha256",
                    "canonical_task_serialization",
                )
            }
            for row in split["accepted_tasks"]
        ]
        observed = _canonical_sha256(frozen)
        if split["ordered_sha256"] != expected or observed != expected:
            raise RuntimeError(f"frozen {name} split hash changed")
        results[name] = {
            "id": split["id"],
            "count": len(frozen),
            "ordered_sha256": observed,
            "performance_episodes": 0,
        }
    paper = ledger["paper"]
    if (
        paper["state"] != "SEALED_UNINSPECTED"
        or paper["content_fingerprints_constructed"] != 0
        or tuple(paper["numeric_exclusion_only"]) != tuple(PAPER_SEEDS)
    ):
        raise RuntimeError("Paper seal changed")
    results["paper"] = dict(paper)
    return results


def _environment() -> gym.Env:
    return gym.wrappers.TimeLimit(StochasticTrack(), max_episode_steps=1_800)


def _outcome(
    info: dict[str, Any],
    *,
    target_actions: list[float],
    decisions: list[dict[str, Any]],
    macro_decisions: int,
) -> dict[str, Any]:
    metrics = EpisodeMetrics(**info["episode_metrics"])
    feasible = is_feasible(metrics, CANONICAL_TASK)
    overrides = sum(bool(row["overridden"]) for row in decisions)
    influenced = sum(float(row["native_action_influence"]) > 0.0 for row in decisions)
    viability_falsifications = [
        {
            "native_step": index,
            "completion_bound_s": row["completion_bound_s"],
            "observed_completion_s": metrics.travel_time_s,
        }
        for index, row in enumerate(decisions)
        if row["accepted"]
        and row["deadline_viable"]
        and row["completion_bound_s"] is not None
        and metrics.travel_time_s > float(row["completion_bound_s"]) + 0.2
    ]
    return {
        **asdict(metrics),
        "feasible": feasible,
        "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
        "deadline_compliant": metrics.travel_time_s <= 140.0,
        "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
        "macro_decision_count": macro_decisions,
        "target_action": target_actions,
        "target_speed_m_s": [
            normalized_action_to_target_speed(value) for value in target_actions
        ],
        "proposal_count": len(decisions),
        "override_count": overrides,
        "override_fraction": overrides / len(decisions) if decisions else 0.0,
        "native_action_influence_count": influenced,
        "native_action_influence_fraction": (
            influenced / len(decisions) if decisions else 0.0
        ),
        "intervention_reasons": dict(
            sorted(Counter(row["intervention_reason"] for row in decisions).items())
        ),
        "deadline_bound_falsifications": viability_falsifications,
    }


def evaluate_seed(
    *,
    numeric_seed: int,
    arm: Literal["causal", "time_aware", "hybrid_rl"],
    agent: ConstrainedSAC | None = None,
) -> dict[str, Any]:
    if numeric_seed in PAPER_SEEDS:
        raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
    environment = _environment()
    target_actions: list[float] = []
    decisions: list[dict[str, Any]] = []
    macro_decisions = 0
    try:
        observation, info = environment.reset(seed=int(numeric_seed))
        if arm == "causal":
            controller: ObservationLimitedController | HybridTargetSpeedController = (
                ObservationLimitedController()
            )
            strategy = None
        else:
            controller = HybridTargetSpeedController(strategy_enabled=True)
            strategy = TimeAwareEnergyStrategy() if arm == "time_aware" else agent
            if strategy is None:
                raise ValueError("Hybrid RL evaluation requires an agent")
        controller.reset()
        if isinstance(strategy, TimeAwareEnergyStrategy):
            strategy.reset()
        held_target: float | None = None
        step = 0
        ended = False
        while not ended:
            if isinstance(controller, HybridTargetSpeedController):
                if step % DECISION_INTERVAL_NATIVE_STEPS == 0:
                    strategy_observation = controller.prepare_strategy_step(observation)
                    if isinstance(strategy, TimeAwareEnergyStrategy):
                        held_target = strategy.action(
                            controller.state,
                            step,
                            route_length_m=controller.config.track_length_m,
                        )
                    else:
                        assert isinstance(strategy, ConstrainedSAC)
                        held_target = strategy.act(
                            strategy_observation, deterministic=True
                        )
                    macro_decisions += 1
                    target_actions.append(float(held_target))
                assert held_target is not None
                native_action = controller.action(observation, held_target)
            else:
                native_action = controller.action(observation)
            observation, _, terminated, truncated, info = environment.step(
                [native_action]
            )
            controller.observe_executed_action(native_action)
            if isinstance(controller, HybridTargetSpeedController):
                assert controller.last_decision is not None
                decisions.append(controller.last_decision.as_dict())
            step += 1
            ended = bool(terminated or truncated)
        return _outcome(
            info,
            target_actions=target_actions,
            decisions=decisions,
            macro_decisions=macro_decisions,
        )
    finally:
        environment.close()


def _split_rows(name: Literal["diagnostic", "validation"]) -> list[dict[str, Any]]:
    ledger = json.loads(DESIGN_SPLIT.read_text(encoding="utf-8"))
    return [dict(row) for row in ledger["evaluation_sets"][name]["accepted_tasks"]]


def evaluate_diagnostic(
    agent: ConstrainedSAC,
    *,
    training_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verify_frozen_splits_without_opening()
    rows = _split_rows("diagnostic")
    outcomes = []
    for row in rows:
        verify_runtime_row(row)
        outcomes.append(
            {
                "numeric_seed": row["numeric_seed"],
                **evaluate_seed(
                    numeric_seed=int(row["numeric_seed"]),
                    arm="hybrid_rl",
                    agent=agent,
                ),
            }
        )
    feasible = [row for row in outcomes if row["feasible"]]
    targets = [target for row in outcomes for target in row["target_speed_m_s"]]
    return {
        "split": "HYBRID_RL_DIAGNOSTIC_V1",
        "episode_count": len(outcomes),
        "rsr": len(feasible),
        "completion_count": sum(row["completed"] for row in outcomes),
        "deadline_compliant_count": sum(row["deadline_compliant"] for row in outcomes),
        "speed_compliant_count": sum(row["speed_compliant"] for row in outcomes),
        "feasible_only_energy_kwh": [row["feasible_energy_kwh"] for row in feasible],
        "target_speed_distribution_m_s": {
            "count": len(targets),
            "minimum": min(targets),
            "mean": sum(targets) / len(targets),
            "maximum": max(targets),
        },
        "proposal_count": sum(row["proposal_count"] for row in outcomes),
        "override_count": sum(row["override_count"] for row in outcomes),
        "native_action_influence_count": sum(
            row["native_action_influence_count"] for row in outcomes
        ),
        "actual_macro_decision_count": sum(
            row["macro_decision_count"] for row in outcomes
        ),
        "deadline_bound_falsifications": [
            {"numeric_seed": row["numeric_seed"], **failure}
            for row in outcomes
            for failure in row["deadline_bound_falsifications"]
        ],
        "learner": {
            "gradient_update_count": agent.update_count,
            "alpha": agent.alpha,
            "speed_multiplier": float(agent.speed_multiplier.detach()),
            "terminal_multiplier": float(agent.terminal_multiplier.detach()),
        },
        "training_context": training_context or {},
        "outcomes": outcomes,
    }


def evaluate_three_arm_row(
    row: dict[str, Any], agent: ConstrainedSAC
) -> dict[str, Any]:
    """Run the exact information-matched comparison on one authorized row."""

    verify_runtime_row(row)
    seed = int(row["numeric_seed"])
    return {
        "numeric_seed": seed,
        "causal": evaluate_seed(numeric_seed=seed, arm="causal"),
        "time_aware": evaluate_seed(numeric_seed=seed, arm="time_aware"),
        "hybrid_rl": evaluate_seed(numeric_seed=seed, arm="hybrid_rl", agent=agent),
    }


def paired_three_arm_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen feasible-only pairing and acceptance hierarchy."""

    causal_pairs = [
        row
        for row in outcomes
        if row["causal"]["feasible"] and row["hybrid_rl"]["feasible"]
    ]
    nonlearning_pairs = [
        row
        for row in outcomes
        if row["causal"]["feasible"]
        and row["time_aware"]["feasible"]
        and row["hybrid_rl"]["feasible"]
    ]
    vs_causal = [
        100.0
        * (row["causal"]["energy_kwh"] - row["hybrid_rl"]["energy_kwh"])
        / row["causal"]["energy_kwh"]
        for row in causal_pairs
    ]
    vs_nonlearning = [
        100.0
        * (row["time_aware"]["energy_kwh"] - row["hybrid_rl"]["energy_kwh"])
        / row["time_aware"]["energy_kwh"]
        for row in nonlearning_pairs
    ]
    rsr = sum(row["hybrid_rl"]["feasible"] for row in outcomes)
    mean_causal = sum(vs_causal) / len(vs_causal) if vs_causal else None
    mean_nonlearning = (
        sum(vs_nonlearning) / len(vs_nonlearning) if vs_nonlearning else None
    )
    return {
        "hybrid_rsr": rsr,
        "paired_feasible_count_vs_causal": len(causal_pairs),
        "paired_feasible_count_vs_nonlearning": len(nonlearning_pairs),
        "paired_reduction_vs_causal_percent": vs_causal,
        "paired_reduction_vs_nonlearning_percent": vs_nonlearning,
        "mean_reduction_vs_causal_percent": mean_causal,
        "mean_reduction_vs_nonlearning_percent": mean_nonlearning,
        "acceptance": {
            "rsr_at_least_35_of_36": rsr >= 35,
            "mean_reduction_vs_causal_at_least_10_percent": (
                mean_causal is not None and mean_causal >= 10.0
            ),
            "mean_reduction_vs_nonlearning_at_least_3_percent": (
                mean_nonlearning is not None and mean_nonlearning >= 3.0
            ),
        },
    }


def evaluate_validation(agent: ConstrainedSAC) -> dict[str, Any]:
    authorization = json.loads(CANONICAL.read_text(encoding="utf-8"))["authorization"]
    if not authorization["validation_authorized"]:
        raise PermissionError("HYBRID_RL_VALIDATION_V1 is frozen and unopened")
    rows = _split_rows("validation")
    outcomes = [evaluate_three_arm_row(row, agent) for row in rows]
    return {
        "split": "HYBRID_RL_VALIDATION_V1",
        "comparison_arms": ("causal", "time_aware", "hybrid_rl"),
        "summary": paired_three_arm_summary(outcomes),
        "outcomes": outcomes,
    }
