"""Bounded Calibration and exactly-once Capability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

import gymnasium as gym

import gym_longicontrol  # noqa: F401 - registers StochasticTrack-v1
from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.causal_mpc_v2.controller import PriceMPCController
from benchmarks.causal_mpc_v2.planner import _mode
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .controller import OptionArm, TemporalOptionController
from .execution import (
    CONTROLLER_ORDER,
    CapabilityLease,
    PairLedger,
    atomic_json,
    initialize_pair_ledger,
)
from .integrity import (
    CANONICAL,
    HERE,
    SPLIT_LEDGER,
    inherited_hashes,
    method_hashes,
    require_not_paper,
    sha256_file,
    verify_frozen_method,
)
from .observation_adapter import ACTION_IDS
from .split import canonical_sha256, materialize

CALIBRATION_JSON = HERE / "calibration_results.json"
CAPABILITY_JSON = HERE / "capability_results.json"
RESULTS_JSON = HERE / "results.json"
RECORDS = HERE / "capability_records"
CANONICAL_TASK = TaskSpecification(140.0, 0.0)
DECISION_INTERVAL_NATIVE_STEPS = 10
ADAPTIVE_HIGH_RESERVE_S = 24.0
ADAPTIVE_LOW_RESERVE_S = 2.0


def _environment() -> gym.Env:
    return gym.make("StochasticTrack-v1", max_episode_steps=1_800)


def _option_snapshot(controller: TemporalOptionController) -> dict[str, Any] | None:
    return (
        None if controller.active_option is None else controller.active_option.as_dict()
    )


def _finish_transition(
    transition: dict[str, Any],
    controller: TemporalOptionController,
    *,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    transition["option_state_after"] = _option_snapshot(controller)
    transition["termination_mask"] = float(terminated)
    transition["bootstrap_mask"] = float(not (terminated or truncated))
    return transition


def evaluate_option_seed(*, numeric_seed: int, arm: OptionArm | str) -> dict[str, Any]:
    """Evaluate one causal option controller without route/seed access at runtime."""

    require_not_paper(numeric_seed)
    environment = _environment()
    controller = TemporalOptionController(arm)
    selector = AdaptivePrice(
        high_reserve_s=ADAPTIVE_HIGH_RESERVE_S,
        low_reserve_s=ADAPTIVE_LOW_RESERVE_S,
    )
    decisions: list[dict[str, Any]] = []
    native_actions: list[float] = []
    transition_records: list[dict[str, Any]] = []
    planned_phases: Counter[str] = Counter()
    planned_outcomes: Counter[str] = Counter()
    option_actions: dict[int, list[float]] = defaultdict(list)
    solve_times: list[float] = []
    current_transition: dict[str, Any] | None = None
    wall_start = time.perf_counter()
    process_start = time.process_time()
    try:
        observation, info = environment.reset(seed=int(numeric_seed))
        controller.reset()
        selector.reset()
        held_price: float | None = None
        step = 0
        ended = False
        while not ended:
            boundary = step % DECISION_INTERVAL_NATIVE_STEPS == 0
            pre_boundary_option = _option_snapshot(controller)
            if boundary:
                if current_transition is not None:
                    transition_records.append(
                        _finish_transition(
                            current_transition,
                            controller,
                            terminated=False,
                            truncated=False,
                        )
                    )
                base = controller.prepare_strategy_step(observation)
                option_observation = controller.rl_observation(base)
                held_price = float(selector.action(base))
                current_transition = {
                    "native_step_start": step,
                    "proposed_high_level_action": None,
                    "actual_executed_duration_native_steps": 0,
                    "option_state_before": pre_boundary_option,
                    "option_state_after": None,
                    "native_reward_sum": 0.0,
                    "native_signed_energy_kwh": 0.0,
                    "native_speed_excess_integral_m": 0.0,
                    "termination_mask": 0.0,
                    "bootstrap_mask": 1.0,
                    "intervention_events": [],
                    "valid_action_mask": option_observation.valid_action_mask.tolist(),
                    "state_dimension": int(option_observation.vector.size),
                }
            assert held_price is not None and current_transition is not None
            before_energy = float(info["total_energy_kwh"])
            solve_start = time.perf_counter()
            action = controller.action(observation, held_price)
            control_time = time.perf_counter() - solve_start
            decision = controller.last_decision
            if decision is None:
                raise RuntimeError("temporal controller did not record a decision")
            if boundary:
                current_transition["proposed_high_level_action"] = (
                    decision.high_level_action
                )
                if decision.replanned:
                    solve_times.append(control_time)
                    if controller.last_selection is not None:
                        selected = controller.last_selection.selected
                        planned_phases.update(selected.phases)
                        planned_outcomes[selected.outcome] += 1
            observation, reward, terminated, truncated, info = environment.step(
                [action]
            )
            ended = bool(terminated or truncated)
            controller.observe_executed_action(action, episode_ended=ended)
            row = decision.as_dict()
            decisions.append(row)
            native_actions.append(float(action))
            if decision.option_instance_id is not None:
                option_actions[decision.option_instance_id].append(float(action))
            current_transition["actual_executed_duration_native_steps"] += 1
            current_transition["native_reward_sum"] += float(reward)
            current_transition["native_signed_energy_kwh"] += (
                float(info["total_energy_kwh"]) - before_energy
            )
            current_transition["native_speed_excess_integral_m"] += (
                max(0.0, float(info["speed_excess_m_s"])) * 0.1
            )
            if not decision.accepted:
                current_transition["intervention_events"].append(
                    {
                        "native_step": step,
                        "reason": decision.intervention_reason,
                    }
                )
            step += 1
        transition_records.append(
            _finish_transition(
                current_transition,
                controller,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
        )
        wall_time = time.perf_counter() - wall_start
        process_time = time.process_time() - process_start
        metrics = EpisodeMetrics(**info["episode_metrics"])
        feasible = is_feasible(metrics, CANONICAL_TASK)
        histories = [option.as_dict() for option in controller.option_history]
        durations = [
            float(option["end_time_s"] - option["start_time_s"])
            for option in histories
            if option["end_time_s"] is not None
        ]
        distances = [
            float(option["end_position_m"] - option["start_position_m"])
            for option in histories
            if option["end_position_m"] is not None
        ]
        action_hashes = {
            hashlib.sha256(
                json.dumps(actions, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for actions in option_actions.values()
        }
        deadline_falsifications = [
            {
                "native_step": row["native_step"],
                "completion_bound_s": None,
                "observed_completion_s": metrics.travel_time_s,
            }
            for row in decisions
            if row["accepted"]
            and row["intervention_reason"] == "DEADLINE_RECOVERABILITY"
        ]
        return {
            **asdict(metrics),
            "numeric_seed": int(numeric_seed),
            "controller_id": OptionArm(arm).value,
            "feasible": feasible,
            "deadline_compliant": metrics.travel_time_s <= 140.0,
            "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
            "signed_energy_kwh": metrics.energy_kwh,
            "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
            "final_progress_m": float(info["position_m"]),
            "native_action_sha256": canonical_sha256(native_actions),
            "native_step_count": step,
            "planner_solve_count": len(solve_times),
            "planner_solve_time_s": sum(solve_times),
            "episode_wall_time_s": wall_time,
            "active_process_time_s": process_time,
            "host_sleep_time_s": 0.0,
            "wall_minus_process_time_s": max(0.0, wall_time - process_time),
            "physics_prediction_calls": sum(
                int(row["physics_prediction_calls"]) for row in decisions
            ),
            "option_start_count": sum(
                bool(
                    row["high_level_action"]
                    and row["high_level_action"].startswith("START")
                )
                for row in decisions
            ),
            "option_continue_count": sum(
                row["high_level_action"] == "CONTINUE" for row in decisions
            ),
            "option_switch_count": sum(
                option["termination_reason"] == "STRATEGIC_SWITCH"
                for option in histories
            ),
            "option_interruption_count": sum(
                option["termination_reason"] == "SAFETY_INTERRUPTION"
                for option in histories
            ),
            "option_termination_reasons": dict(
                sorted(
                    Counter(
                        option["termination_reason"] for option in histories
                    ).items()
                )
            ),
            "option_duration_s": durations,
            "option_distance_m": distances,
            "planned_phase_counts": dict(sorted(planned_phases.items())),
            "planned_option_outcomes": dict(sorted(planned_outcomes.items())),
            "executed_phase_counts": dict(
                sorted(Counter(row["phase"] for row in decisions).items())
            ),
            "physical_action_mode_counts": dict(
                sorted(Counter(row["action_mode"] for row in decisions).items())
            ),
            "profile_occupancy": dict(
                sorted(
                    Counter(
                        row["profile_id"]
                        for row in decisions
                        if row["profile_id"] is not None
                    ).items()
                )
            ),
            "distinct_executed_option_trajectories": len(action_hashes),
            "safety_override_reasons": dict(
                sorted(
                    Counter(
                        row["intervention_reason"]
                        for row in decisions
                        if not row["accepted"]
                    ).items()
                )
            ),
            "reset_before_planned_phase_threshold_count": sum(
                option["termination_reason"] == "STRATEGIC_SWITCH"
                and option["progress"] < 0.3
                for option in histories
            ),
            "deadline_bound_falsifications": deadline_falsifications,
            "transition_records": transition_records,
            "option_history": histories,
            "high_level_action_count": len(ACTION_IDS),
            "rl_gradient_updates": 0,
        }
    finally:
        environment.close()


def evaluate_legacy_seed(numeric_seed: int) -> dict[str, Any]:
    require_not_paper(numeric_seed)
    environment = _environment()
    controller = PriceMPCController()
    selector = AdaptivePrice(
        high_reserve_s=ADAPTIVE_HIGH_RESERVE_S,
        low_reserve_s=ADAPTIVE_LOW_RESERVE_S,
    )
    wall_start = time.perf_counter()
    process_start = time.process_time()
    decisions: list[dict[str, Any]] = []
    actions: list[float] = []
    solve_times: list[float] = []
    profiles: list[str] = []
    try:
        observation, info = environment.reset(seed=int(numeric_seed))
        controller.reset()
        selector.reset()
        held_price: float | None = None
        step = 0
        ended = False
        while not ended:
            if step % DECISION_INTERVAL_NATIVE_STEPS == 0:
                held_price = float(
                    selector.action(controller.prepare_strategy_step(observation))
                )
            assert held_price is not None
            started = time.perf_counter()
            action = controller.action(observation, held_price)
            control_time = time.perf_counter() - started
            decision = controller.last_decision
            if decision is None:
                raise RuntimeError("legacy controller did not record a decision")
            if decision.replanned:
                solve_times.append(control_time)
                if decision.planner_profile_id is not None:
                    profiles.append(decision.planner_profile_id)
            observation, _, terminated, truncated, info = environment.step([action])
            controller.observe_executed_action(action)
            decisions.append(decision.as_dict())
            actions.append(float(action))
            step += 1
            ended = bool(terminated or truncated)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        feasible = is_feasible(metrics, CANONICAL_TASK)
    finally:
        environment.close()
    active_process_time = time.process_time() - process_start
    wall_time = time.perf_counter() - wall_start
    result = {
        **asdict(metrics),
        "numeric_seed": int(numeric_seed),
        "feasible": feasible,
        "deadline_compliant": metrics.travel_time_s <= 140.0,
        "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
        "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
        "planner_solve_count": len(solve_times),
        "planner_solve_time_s": sum(solve_times),
        "episode_wall_time_s": wall_time,
        "physics_prediction_calls": sum(
            int(row["physics_prediction_calls"]) for row in decisions
        ),
        "native_action_sha256": canonical_sha256(actions),
        "deadline_bound_falsifications": [],
    }
    result.update(
        {
            "controller_id": "FROZEN_MPC_V2",
            "signed_energy_kwh": result["energy_kwh"],
            "final_progress_m": float(info["position_m"]),
            "native_step_count": len(actions),
            "active_process_time_s": active_process_time,
            "host_sleep_time_s": 0.0,
            "wall_minus_process_time_s": max(
                0.0,
                wall_time - active_process_time,
            ),
            "option_start_count": 0,
            "option_continue_count": 0,
            "option_switch_count": 0,
            "option_interruption_count": 0,
            "option_termination_reasons": {},
            "option_duration_s": [],
            "option_distance_m": [],
            "planned_phase_counts": {},
            "planned_option_outcomes": {},
            "executed_phase_counts": {},
            "physical_action_mode_counts": dict(
                sorted(Counter(_mode(action) for action in actions).items())
            ),
            "profile_occupancy": dict(sorted(Counter(profiles).items())),
            "distinct_executed_option_trajectories": 1,
            "safety_override_reasons": dict(
                sorted(
                    Counter(
                        row["intervention_reason"]
                        for row in decisions
                        if row["overridden"]
                    ).items()
                )
            ),
            "reset_before_planned_phase_threshold_count": 0,
            "transition_records": [],
            "option_history": [],
            "high_level_action_count": 0,
            "rl_gradient_updates": 0,
        }
    )
    return result


def evaluate_pair(controller_id: str, seed: int) -> dict[str, Any]:
    if controller_id == "FROZEN_MPC_V2":
        return evaluate_legacy_seed(seed)
    return evaluate_option_seed(numeric_seed=seed, arm=controller_id)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [row for row in rows if row["feasible"]]
    energies = [float(row["energy_kwh"]) for row in feasible]
    durations = [value for row in rows for value in row["option_duration_s"]]
    distances = [value for row in rows for value in row["option_distance_m"]]
    return {
        "episode_count": len(rows),
        "rsr": len(feasible),
        "completion_count": sum(bool(row["completed"]) for row in rows),
        "deadline_compliance_count": sum(
            bool(row["deadline_compliant"]) for row in rows
        ),
        "speed_compliance_count": sum(bool(row["speed_compliant"]) for row in rows),
        "mean_signed_energy_kwh": statistics.fmean(
            float(row["energy_kwh"]) for row in rows
        ),
        "mean_feasible_energy_kwh": statistics.fmean(energies) if energies else None,
        "mean_travel_time_s": statistics.fmean(
            float(row["travel_time_s"]) for row in rows
        ),
        "mean_final_progress_m": statistics.fmean(
            float(row["final_progress_m"]) for row in rows
        ),
        "option_start_count": sum(int(row["option_start_count"]) for row in rows),
        "option_continue_count": sum(int(row["option_continue_count"]) for row in rows),
        "option_switch_count": sum(int(row["option_switch_count"]) for row in rows),
        "option_interruption_count": sum(
            int(row["option_interruption_count"]) for row in rows
        ),
        "mean_option_duration_s": statistics.fmean(durations) if durations else None,
        "median_option_duration_s": statistics.median(durations) if durations else None,
        "mean_option_distance_m": statistics.fmean(distances) if distances else None,
        "planner_calls": sum(int(row["planner_solve_count"]) for row in rows),
        "physics_prediction_calls": sum(
            int(row["physics_prediction_calls"]) for row in rows
        ),
        "planner_time_s": sum(float(row["planner_solve_time_s"]) for row in rows),
        "episode_wall_time_s": sum(float(row["episode_wall_time_s"]) for row in rows),
        "active_process_time_s": sum(
            float(row["active_process_time_s"]) for row in rows
        ),
        "host_sleep_time_s": sum(float(row["host_sleep_time_s"]) for row in rows),
        "deadline_bound_falsification_count": sum(
            len(row["deadline_bound_falsifications"]) for row in rows
        ),
        "executed_phase_counts": dict(
            sorted(
                sum(
                    (Counter(row["executed_phase_counts"]) for row in rows), Counter()
                ).items()
            )
        ),
        "physical_action_mode_counts": dict(
            sorted(
                sum(
                    (Counter(row["physical_action_mode_counts"]) for row in rows),
                    Counter(),
                ).items()
            )
        ),
        "safety_override_reasons": dict(
            sorted(
                sum(
                    (Counter(row["safety_override_reasons"]) for row in rows), Counter()
                ).items()
            )
        ),
        "distinct_native_action_trajectories": len(
            {row["native_action_sha256"] for row in rows}
        ),
    }


def _paired(
    baseline: list[dict[str, Any]], treatment: list[dict[str, Any]], *, name: str
) -> dict[str, Any]:
    left = {int(row["numeric_seed"]): row for row in baseline}
    right = {int(row["numeric_seed"]): row for row in treatment}
    task_rows = []
    for seed in sorted(left.keys() & right.keys()):
        before, after = left[seed], right[seed]
        row = {
            "numeric_seed": seed,
            "baseline_feasible": before["feasible"],
            "treatment_feasible": after["feasible"],
            "travel_time_delta_s": float(after["travel_time_s"])
            - float(before["travel_time_s"]),
        }
        if before["feasible"] and after["feasible"]:
            row["saving_percent"] = (
                100.0
                * (float(before["energy_kwh"]) - float(after["energy_kwh"]))
                / float(before["energy_kwh"])
            )
        else:
            row["saving_percent"] = None
        task_rows.append(row)
    savings = [
        row["saving_percent"] for row in task_rows if row["saving_percent"] is not None
    ]
    time_deltas = [row["travel_time_delta_s"] for row in task_rows]
    return {
        "comparison": name,
        "jointly_feasible_count": len(savings),
        "arithmetic_mean_saving_percent": statistics.fmean(savings)
        if savings
        else None,
        "median_saving_percent": statistics.median(savings) if savings else None,
        "saving_percent_range": [min(savings), max(savings)] if savings else None,
        "mean_travel_time_delta_s": statistics.fmean(time_deltas),
        "task_level": task_rows,
    }


def compile_results(
    records: list[dict[str, Any]], ledger: dict[str, Any]
) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for controller_id in CONTROLLER_ORDER:
        rows = sorted(
            (row for row in records if row["controller_id"] == controller_id),
            key=lambda row: int(row["numeric_seed"]),
        )
        arms[controller_id] = {"summary": _summary(rows), "episodes": rows}
    comparisons = {
        "primary_continue_vs_reset": _paired(
            arms["RESET_FEEDBACK"]["episodes"],
            arms["CONTINUE_FEEDBACK"]["episodes"],
            name="CONTINUE_FEEDBACK relative to RESET_FEEDBACK",
        ),
        "feedback_vs_legacy": _paired(
            arms["FROZEN_MPC_V2"]["episodes"],
            arms["RESET_FEEDBACK"]["episodes"],
            name="RESET_FEEDBACK relative to FROZEN_MPC_V2",
        ),
        "commit_vs_continue": _paired(
            arms["CONTINUE_FEEDBACK"]["episodes"],
            arms["COMMIT_FEEDBACK"]["episodes"],
            name="COMMIT_FEEDBACK relative to CONTINUE_FEEDBACK",
        ),
        "continue_vs_legacy": _paired(
            arms["FROZEN_MPC_V2"]["episodes"],
            arms["CONTINUE_FEEDBACK"]["episodes"],
            name="CONTINUE_FEEDBACK relative to FROZEN_MPC_V2",
        ),
    }
    primary = comparisons["primary_continue_vs_reset"]
    reset_rsr = arms["RESET_FEEDBACK"]["summary"]["rsr"]
    continue_rsr = arms["CONTINUE_FEEDBACK"]["summary"]["rsr"]
    saving = primary["arithmetic_mean_saving_percent"]
    semantic_pass = all(
        arms[name]["summary"]["deadline_bound_falsification_count"] == 0
        for name in CONTROLLER_ORDER
    )
    if not semantic_pass:
        classification = "SEMANTIC_OR_INTEGRITY_STOP"
    elif continue_rsr < reset_rsr:
        classification = "OPTIONS_VALID_FEASIBILITY_REGRESSION"
    elif saving is not None and saving >= 1.0:
        classification = "OPTIONS_VALID_PROMISING"
    else:
        classification = "OPTIONS_VALID_EFFECT_UNCLEAR"
    task_classifications = {}
    selected = {
        int(row["numeric_seed"]): row
        for row in ledger["capability_panel"]["accepted_tasks"]
    }
    for seed in sorted(selected):
        rows = [row for row in records if int(row["numeric_seed"]) == seed]
        if any(row["feasible"] for row in rows):
            task_classifications[str(seed)] = "FEASIBLE_WITNESS"
        elif (
            selected[seed]["feasibility_classification_at_selection"]
            == "PROVEN_INFEASIBLE"
        ):
            task_classifications[str(seed)] = "PROVEN_INFEASIBLE"
        else:
            task_classifications[str(seed)] = "UNRESOLVED"
    return {
        "schema_version": 1,
        "study": "OPTIONS_CAPABILITY_V1",
        "controller_order": list(CONTROLLER_ORDER),
        "panel_ordered_sha256": ledger["capability_panel"]["ordered_sha256"],
        "arms": arms,
        "comparisons": comparisons,
        "task_feasibility_classification": task_classifications,
        "scientific_classification": classification,
        "interpretation_scope": "non-learning execution architecture only",
        "rl_training_episodes": 0,
        "rl_gradient_updates": 0,
        "historical_validation_episodes": 0,
        "paper_episodes": 0,
    }


def prepare_study() -> dict[str, Any]:
    ledger = materialize()
    seeds = [
        int(row["numeric_seed"]) for row in ledger["capability_panel"]["accepted_tasks"]
    ]
    pair_ledger = initialize_pair_ledger(seeds)
    canonical = {
        "schema_version": 1,
        "source_commit": "c7c6175b22e373f476aa60c10a755cb7072d9d5f",
        "authorization": {
            "capability_authorized": True,
            "rl_training_authorized": False,
            "historical_validation_authorized": False,
            "paper_authorized": False,
        },
        "execution": {
            "calibration_complete": False,
            "capability_opened": False,
            "capability_complete": False,
        },
        "frozen_arms": list(CONTROLLER_ORDER),
        "adaptive_price": {
            "high_reserve_s": ADAPTIVE_HIGH_RESERVE_S,
            "low_reserve_s": ADAPTIVE_LOW_RESERVE_S,
            "source": "inherited Causal MPC V2 selected setting",
        },
        "physical_assumptions": {
            "route_length_m": 1000.0,
            "deadline_s": 140.0,
            "native_dt_s": 0.1,
            "outer_episode_steps": 1800,
            "option_distance_m": 100.0,
            "maximum_option_native_steps": 400,
            "strategy_interval_native_steps": 10,
            "speed_tolerance_m_s": 0.0,
            "shield_status": "empirically supported; not a global theorem",
            "unknown_tail": "inherited heuristic; unseen route never modeled as known",
        },
        "resource_gates": {
            "active_compute_hours_maximum": 8.0,
            "retained_artifacts_gb_maximum": 4.0,
            "calibration_episode_maximum": 48,
            "fixture_native_step_maximum": 30000,
        },
        "frozen_method_sha256": method_hashes(),
        "inherited_source_sha256": inherited_hashes(),
        "pair_identities_sha256": pair_ledger["identities_sha256"],
        "paper_state": "SEALED_UNINSPECTED",
    }
    atomic_json(CANONICAL, canonical)
    return canonical


def run_calibration() -> dict[str, Any]:
    verify_frozen_method()
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if canonical["execution"]["calibration_complete"]:
        raise PermissionError("OPTIONS_CALIBRATION_V1 is complete")
    if canonical["execution"]["capability_opened"]:
        raise PermissionError("calibration cannot run after Capability opens")
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    seeds = [
        int(row["numeric_seed"])
        for row in ledger["calibration_panel"]["accepted_tasks"]
    ]
    records = [
        evaluate_pair(controller, seed)
        for seed in seeds
        for controller in CONTROLLER_ORDER
    ]
    total_wall = sum(float(row["episode_wall_time_s"]) for row in records)
    serialized = json.dumps(records, allow_nan=False).encode("utf-8")
    projected_hours = total_wall * 72.0 / 48.0 / 3600.0
    projected_bytes = len(serialized) * 72 // 48
    result = {
        "schema_version": 1,
        "panel_id": "OPTIONS_CALIBRATION_V1",
        "episode_count": len(records),
        "records": records,
        "resource_measurement": {
            "wall_time_s": total_wall,
            "active_process_time_s": sum(
                float(row["active_process_time_s"]) for row in records
            ),
            "host_sleep_time_s": 0.0,
            "projected_capability_active_compute_hours": projected_hours,
            "projected_retained_artifacts_bytes": projected_bytes,
            "active_compute_gate_pass": projected_hours <= 8.0,
            "artifact_gate_pass": projected_bytes <= 4_000_000_000,
        },
        "selection_or_retuning_performed": False,
        "rl_gradient_updates": 0,
        "paper_episodes": 0,
        "historical_validation_episodes": 0,
    }
    if len(records) > 48:
        raise RuntimeError("Calibration episode budget exceeded")
    atomic_json(CALIBRATION_JSON, result)
    canonical["execution"]["calibration_complete"] = True
    atomic_json(CANONICAL, canonical)
    return result


def _preflight_capability() -> tuple[dict[str, Any], dict[str, Any]]:
    verify_frozen_method()
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if not canonical["execution"]["calibration_complete"]:
        raise PermissionError("Calibration must complete before Capability")
    calibration = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    resource = calibration["resource_measurement"]
    if not resource["active_compute_gate_pass"] or not resource["artifact_gate_pass"]:
        raise RuntimeError(
            "resource projection gate failed; Capability remains unopened"
        )
    ledger = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    # Report-schema smoke test uses synthetic records only.
    synthetic = []
    template = calibration["records"][0]
    for controller in CONTROLLER_ORDER:
        row = dict(template)
        row["controller_id"] = controller
        row["numeric_seed"] = -1
        synthetic.append(row)
    compile_results(
        synthetic,
        {
            **ledger,
            "capability_panel": {**ledger["capability_panel"], "accepted_tasks": []},
        },
    )
    return canonical, ledger


def run_capability() -> dict[str, Any]:
    canonical, split_ledger = _preflight_capability()
    if canonical["execution"]["capability_complete"]:
        raise PermissionError("OPTIONS_CAPABILITY_V1 is complete")
    pair_ledger = PairLedger()
    with CapabilityLease() as lease:
        payload = pair_ledger.open()
        canonical["execution"]["capability_opened"] = True
        atomic_json(CANONICAL, canonical)
        for pair_id, pair in payload["pairs"].items():
            if pair["state"] == "COMPLETE":
                continue
            lease.heartbeat(pair_id)
            pair_ledger.start_pair(pair_id)
            record = evaluate_pair(pair["controller_id"], int(pair["numeric_seed"]))
            path = RECORDS / f"{pair['controller_id']}__{pair['numeric_seed']}.json"
            atomic_json(path, record)
            pair_ledger.complete_pair(pair_id, path, sha256_file(path))
        complete_ledger = pair_ledger.finish()
    records = []
    for pair in complete_ledger["pairs"].values():
        path = HERE / pair["record_path"]
        if sha256_file(path) != pair["record_sha256"]:
            raise RuntimeError("completed capability record hash changed")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    result = compile_results(records, split_ledger)
    atomic_json(CAPABILITY_JSON, result)
    atomic_json(RESULTS_JSON, result)
    split_ledger["capability_panel"]["state"] = "COMPLETE"
    split_ledger["capability_panel"]["performance_episode_count"] = 72
    atomic_json(SPLIT_LEDGER, split_ledger)
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    canonical["execution"]["capability_complete"] = True
    atomic_json(CANONICAL, canonical)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "calibration", "capability"))
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare_study()
    elif args.stage == "calibration":
        result = run_calibration()
    else:
        result = run_capability()
    print(
        json.dumps(
            {
                "stage": args.stage,
                "complete": True,
                "schema_version": result["schema_version"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
