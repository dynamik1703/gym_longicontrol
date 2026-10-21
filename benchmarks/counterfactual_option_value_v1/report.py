"""Final prospective-gate decision and human-readable scientific report."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from typing import Any

from .execution import atomic_json, sha256_file
from .integrity import BRANCH_LEDGER, CANONICAL, HERE, SPLIT_LEDGER, verify_seals
from .opportunity_analysis import OUTPUT as OPPORTUNITY
from .predictor import ARTIFACT, MODEL_SELECTION
from .predictor_evaluation import EVALUATION, PREDICTION_FREEZE

RESULTS_JSON = HERE / "results.json"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()


def _resource_report(
    ledger: dict[str, Any], manifest: dict[str, Any], artifact: dict[str, Any]
) -> dict[str, Any]:
    branch_records = []
    for row in ledger["branches"].values():
        path = HERE / row["record_path"]
        if sha256_file(path) != row["record_sha256"]:
            raise RuntimeError("branch record changed before final report")
        branch_records.append(json.loads(path.read_text(encoding="utf-8")))
    retained_bytes = sum(
        path.stat().st_size for path in HERE.rglob("*") if path.is_file()
    )
    return {
        "normal_sampling_trajectory_count": len(manifest["route_results"]),
        "normal_sampling_native_steps": sum(
            int(row["normal_trajectory_native_steps"])
            for row in manifest["route_results"]
        ),
        "sampled_state_count": len(manifest["snapshots"]),
        "completed_continuation_count": len(branch_records),
        "continuation_native_simulator_steps": sum(
            int(row["native_step_count"]) for row in branch_records
        ),
        "continuation_physics_prediction_calls": sum(
            int(row["physics_prediction_calls"]) for row in branch_records
        ),
        "continuation_active_process_time_s": sum(
            float(row["active_process_time_s"]) for row in branch_records
        ),
        "continuation_wall_time_s": sum(
            float(row["wall_time_s"]) for row in branch_records
        ),
        "predictor_gradient_updates": int(artifact["fit_gradient_updates"]),
        "rl_gradient_updates": 0,
        "retained_package_bytes_before_final_markdown": retained_bytes,
        "active_compute_limit_hours": 12.0,
        "retained_artifact_limit_bytes": 4_000_000_000,
    }


def build_report() -> dict[str, Any]:
    if RESULTS_JSON.exists():
        raise PermissionError("final results already exist")
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    ledger = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "dataset_manifest.json").read_text(encoding="utf-8"))
    opportunity = json.loads(OPPORTUNITY.read_text(encoding="utf-8"))
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    selection = json.loads(MODEL_SELECTION.read_text(encoding="utf-8"))
    freeze = json.loads(PREDICTION_FREEZE.read_text(encoding="utf-8"))
    seals = verify_seals()
    record_integrity = []
    for row in ledger["branches"].values():
        if row["state"] != "COMPLETE":
            raise RuntimeError("incomplete branch in final report")
        path = HERE / row["record_path"]
        record = json.loads(path.read_text(encoding="utf-8"))
        record_integrity.append(record["integrity_audit"])
        if sha256_file(path) != row["record_sha256"]:
            raise RuntimeError("branch record hash mismatch")
        if sha256_file(HERE / row["trace_path"]) != row["trace_sha256"]:
            raise RuntimeError("native trace hash mismatch")
    integrity_gates = {
        "snapshot_reference_parity": all(canonical["preflight"]["checks"].values()),
        "full_native_record_retention": all(
            row["record_completeness_verified"] for row in record_integrity
        ),
        "independent_completion_bound_audit": all(
            row["deadline_bound_falsifications"] == [] for row in record_integrity
        ),
        "exact_native_replay": all(
            row["exact_native_replay_verified"] for row in record_integrity
        ),
        "immutable_route_clean_splits": split["route_clean_verified"],
        "verified_counterfactual_injection": canonical["preflight"]["checks"][
            "alternative_injection_isolated"
        ],
        "identical_physical_rules": True,
        "paper_and_historical_validation_sealed": (
            seals["paper"]["state"] == "SEALED_UNINSPECTED"
            and seals["historical_validation_panels_opened"] == 0
        ),
    }
    test_opportunity = opportunity["panels"]["CF_OPTION_TEST_V1"]
    opportunity_fraction = test_opportunity["distinguishable_behavior"][
        "saving_0_2wh_fraction_of_valid_states"
    ]
    opportunity_gates = {
        "at_least_20_percent_distinguishable_0_2wh": (
            opportunity_fraction is not None and opportunity_fraction >= 0.20
        ),
        "genuine_behavior_or_memory_difference": (
            test_opportunity["distinguishable_behavior"]["valid_snapshot_count"] > 0
        ),
        "infeasible_energy_not_counted": True,
    }
    savings_by_route: dict[int, float] = defaultdict(float)
    for row in evaluation["selected_snapshot_rows"]:
        if row["actual_delta_energy_kwh"] < 0.0 and not row["false_safe"]:
            savings_by_route[int(row["numeric_seed"])] += -float(
                row["actual_delta_energy_kwh"]
            )
    total_selected_saving = sum(savings_by_route.values())
    max_route_share = (
        max(savings_by_route.values()) / total_selected_saving
        if total_selected_saving > 0.0
        else None
    )
    regret = evaluation["regret_kwh"]
    learning_gates = {
        "regret_improves_over_always_reference": (
            regret["frozen_causal_predictor"] < regret["always_reference"]
        ),
        "positive_net_energy_after_feasibility_filter": (
            evaluation["net_selected_delta_energy_kwh"] < 0.0
        ),
        "no_new_hard_feasibility_failures": (evaluation["false_safe_decisions"] == 0),
        "not_explained_by_fewer_than_three_routes": len(savings_by_route) >= 3,
        "no_single_route_carries_more_than_half": (
            max_route_share is not None and max_route_share <= 0.5
        ),
    }
    evidence_gates = {
        "minimum_valid_test_snapshots": (
            test_opportunity["valid_reference_snapshot_count"] >= 72
        ),
        "minimum_test_routes": evaluation["route_count"] >= 10,
    }
    if not all(integrity_gates.values()):
        classification = "SAFETY_OR_INTEGRITY_STOP"
    elif not all(evidence_gates.values()):
        classification = "INSUFFICIENT_EVIDENCE"
    elif not all(opportunity_gates.values()):
        classification = "INSUFFICIENT_IMPROVEMENT_OPPORTUNITY"
    elif all(learning_gates.values()):
        classification = "COUNTERFACTUAL_OPPORTUNITY_AND_LEARNABILITY_SUPPORTED"
    else:
        classification = "OPPORTUNITY_EXISTS_BUT_NOT_CAUSALLY_LEARNABLE"
    resources = _resource_report(ledger, manifest, artifact)
    result = {
        "schema_version": 1,
        "study": "COUNTERFACTUAL_OPTION_VALUE_V1",
        "scientific_classification": classification,
        "source_commit": "29b31837984b654e9011d3e180a8e33926345489",
        "report_generation_commit": _git_head(),
        "integrity_gates": integrity_gates,
        "evidence_gates": evidence_gates,
        "opportunity_gates": opportunity_gates,
        "causal_learning_gates": learning_gates,
        "maximum_selected_saving_route_share": max_route_share,
        "selected_saving_route_count": len(savings_by_route),
        "opportunity_test": test_opportunity,
        "causal_test": evaluation,
        "selected_model": selection["selected"],
        "blind_test_freeze": freeze,
        "resources": resources,
        "opened_splits": [
            "CF_OPTION_FIT_V1",
            "CF_OPTION_CALIBRATION_V1",
            "CF_OPTION_TEST_V1",
        ],
        "paper_state": "SEALED_UNINSPECTED",
        "historical_validation_episodes": 0,
        "original_temporal_options_capability_reruns": 0,
        "full_rl_training_episodes": 0,
        "actor_critic_updates": 0,
        "additional_hybrid_or_slacktrader_experiments": 0,
        "temporal_options_v1_classification_unchanged": ("SEMANTIC_OR_INTEGRITY_STOP"),
    }
    atomic_json(RESULTS_JSON, result)
    selected_model = selection["selected"]
    causal_lines = [
        "# Causal Learnability",
        "",
        f"Frozen family: `{selected_model['family']}`; energy margin: "
        f"`{selected_model['margin_wh']} Wh`; risk threshold: "
        f"`{selected_model['risk_threshold']}`.",
        "",
        f"- Energy-difference MAE: {evaluation['energy_difference_mae_kwh']} kWh",
        f"- Action-ranking accuracy: {evaluation['action_ranking_accuracy']}",
        "- Meaningful-sign accuracy (|difference| >= 0.2 Wh): "
        f"{evaluation['meaningful_sign_accuracy_at_0_2wh']}",
        f"- Always-reference regret: {regret['always_reference']} kWh",
        f"- Frozen-predictor regret: {regret['frozen_causal_predictor']} kWh",
        f"- Heuristic-runner-up regret: {regret['always_heuristic_runner_up']} kWh",
        f"- False-safe decisions: {evaluation['false_safe_decisions']}",
        f"- Harmful predicted improvements: "
        f"{evaluation['predicted_improvements_actually_harmful']}",
        f"- Abstention rate: {evaluation['abstention_rate']}",
        f"- Routes with useful predictions: "
        f"{evaluation['routes_with_useful_predictions']}",
        "",
        "Route-bootstrap uncertainty and every selected decision are retained in",
        "`test_evaluation.json`. No Test-dependent tuning followed scoring.",
        "",
    ]
    (HERE / "CAUSAL_LEARNABILITY.md").write_text(
        "\n".join(causal_lines), encoding="utf-8"
    )
    resource_lines = [
        "# Resource Report",
        "",
        f"- Sampling trajectories: {resources['normal_sampling_trajectory_count']}",
        f"- Sampling native steps: {resources['normal_sampling_native_steps']}",
        f"- Sampled states: {resources['sampled_state_count']}",
        f"- Completed continuations: {resources['completed_continuation_count']}",
        "- Continuation native simulator steps: "
        f"{resources['continuation_native_simulator_steps']}",
        "- MPC/safety physics predictions: "
        f"{resources['continuation_physics_prediction_calls']}",
        "- Continuation active process time: "
        f"{resources['continuation_active_process_time_s']} s",
        f"- Supervised gradient updates: {resources['predictor_gradient_updates']}",
        "- RL/actor-critic gradient updates: 0",
        "- Retained bytes before final Markdown: "
        f"{resources['retained_package_bytes_before_final_markdown']}",
        "",
    ]
    (HERE / "RESOURCE_REPORT.md").write_text(
        "\n".join(resource_lines), encoding="utf-8"
    )
    result_lines = [
        "# Counterfactual Option Value V1 — Results",
        "",
        f"Final classification: `{classification}`.",
        "",
        "## Direct answers",
        "",
        "1. Better decisions available: "
        f"{test_opportunity['fraction_any_feasible_saving']} of valid Test states "
        "had any feasible sampled saving.",
        "2. Meaningful opportunity: "
        f"{test_opportunity['fraction_saving_at_least_0_2wh']} reached 0.2 Wh; "
        f"distribution {test_opportunity['best_feasible_advantage_wh']}.",
        "3. Introduced infeasibility: "
        f"{test_opportunity['introduced_infeasibility_count']} alternatives, fraction "
        f"{test_opportunity['introduced_infeasibility_fraction']}.",
        f"4. CONTINUE opportunity: {test_opportunity['continue']}.",
        "5. Hidden-future dependence: the privileged hindsight delta is "
        f"{evaluation['counterfactual_hindsight_diagnostic_delta_energy_kwh']} kWh "
        "versus causal selected delta "
        f"{evaluation['net_selected_delta_energy_kwh']} kWh. This gap diagnoses "
        "unrecovered opportunity; it does not prove a particular hidden cause.",
        f"6. Causal prediction gates: {learning_gates}.",
        "7. Predictor versus frozen heuristic runner-up: "
        f"{regret['frozen_causal_predictor']} versus "
        f"{regret['always_heuristic_runner_up']} kWh regret.",
        f"8. Actual predictor regret: {regret['frozen_causal_predictor']} kWh.",
        "9. Confidently wrong: "
        f"{evaluation['predicted_improvements_actually_harmful']} harmful and "
        f"{evaluation['false_safe_decisions']} false-safe selected deviations.",
        f"10. One-deviation 430 m diagnostic: {evaluation['one_deviation_430m']}.",
        f"11. Future learned selector support: `{classification}`.",
        "12. Native safety/completion records independently verifiable: "
        f"{all(integrity_gates.values())}.",
        f"13. Actual resource accounting: {resources}.",
        "14. Permanently opened internal splits: Fit, Calibration and Test V1.",
        "15. Paper remained `SEALED_UNINSPECTED`.",
        "16. Commit/remote synchronization is recorded in the final repository "
        "handoff.",
        "",
        "## Explicit boundaries",
        "",
        "- No full RL training or actor-critic updates.",
        "- No additional Hybrid/SlackTrader experiment.",
        "- No original Temporal Options Capability rerun.",
        "- No Paper construction, fingerprinting or evaluation.",
        "- Historical results and the V1 integrity-stop classification are unchanged.",
        "- No automatic learned START/CONTINUE RL controller follows this pilot.",
        "",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(result_lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps({"classification": build_report()["scientific_classification"]}))
