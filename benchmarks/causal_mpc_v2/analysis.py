"""Scientific gates, resource projection and final artifact rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import CALIBRATION_JSON, CAPABILITY_JSON
from .integrity import CANONICAL, HERE, method_hashes, verify_paper_seal

RESULTS_JSON = HERE / "results.json"
PREPARATION_STATUS = HERE / "preparation_status.json"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resource_projection(capability: dict[str, Any]) -> dict[str, Any]:
    mpc = [
        capability["arms"][name]["summary"]
        for name in ("FIXED_PRICE_V2", "ADAPTIVE_PRICE_V2")
    ]
    solve_count = sum(int(row["planner_solve_count"]) for row in mpc)
    solve_time = sum(float(row["planner_solve_time_s"]) for row in mpc)
    wall_time = sum(float(row["episode_wall_time_s"]) for row in mpc)
    seconds_per_solve = solve_time / solve_count
    native_training_steps = 3 * 300_000
    training_solves = native_training_steps // 10
    diagnostic_episodes = 3 * 6 * 18
    mean_episode_solves = solve_count / (2 * 18)
    diagnostic_solves = diagnostic_episodes * mean_episode_solves
    projected_solves = training_solves + diagnostic_solves
    projected_planner_hours = projected_solves * seconds_per_solve / 3600.0
    overhead_ratio = wall_time / solve_time if solve_time else 1.0
    projected_total_hours = projected_planner_hours * overhead_ratio
    # Conservative retained-byte budget: full float64 transition tuples,
    # diagnostics and checkpoints. It intentionally exceeds current JSON rates.
    transition_bytes = native_training_steps * 512
    diagnostic_bytes = diagnostic_episodes * 2_000_000
    checkpoint_bytes = 3 * 7 * 50_000_000
    projected_bytes = transition_bytes + diagnostic_bytes + checkpoint_bytes
    return {
        "measured_capability_mpc_episodes": 36,
        "measured_planner_solve_count": solve_count,
        "measured_planner_solve_time_s": solve_time,
        "measured_mpc_wall_time_s": wall_time,
        "measured_seconds_per_solve": seconds_per_solve,
        "measured_solves_per_second": 1.0 / seconds_per_solve,
        "projection_assumptions": {
            "native_training_steps": native_training_steps,
            "training_plans_at_1_hz": training_solves,
            "diagnostic_episodes": diagnostic_episodes,
            "diagnostic_plans": diagnostic_solves,
            "observed_wall_to_planner_ratio": overhead_ratio,
        },
        "projected_three_seed_runtime_hours": projected_total_hours,
        "projected_retained_artifacts_bytes": projected_bytes,
        "projected_retained_artifacts_gb": projected_bytes / 1.0e9,
        "runtime_projection_is_measurement_based": True,
        "artifact_projection_is_conservative_accounting_estimate": True,
    }


def build_results() -> dict[str, Any]:
    verify_paper_seal()
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    capability = json.loads(CAPABILITY_JSON.read_text(encoding="utf-8"))
    if method_hashes() != canonical["frozen_method_sha256"]:
        raise RuntimeError("method hashes changed after Calibration")
    gates = canonical["gates"]
    summaries = {
        name: value["summary"] for name, value in capability["arms"].items()
    }
    comparisons = capability["paired_feasible_energy"]
    adaptive_vs_time = comparisons["adaptive_vs_time_aware"][
        "arithmetic_mean_saving_percent"
    ]
    adaptive_vs_fixed = comparisons["adaptive_vs_fixed"][
        "arithmetic_mean_saving_percent"
    ]
    adaptive = summaries["ADAPTIVE_PRICE_V2"]
    time_aware = summaries["TIME_AWARE"]
    resource = _resource_projection(capability)
    hard_integrity = all(
        (
            adaptive["speed_compliance_count"] == adaptive["episode_count"],
            adaptive["deadline_bound_falsification_count"] == 0,
            adaptive["nonfinite_value_count"] == 0,
            capability["paper_episodes"] == 0,
            capability["validation_episodes"] == 0,
            capability["rl_training_episodes"] == 0,
            capability["rl_gradient_updates"] == 0,
        )
    )
    gate_a = adaptive["rsr"] >= time_aware["rsr"]
    gate_b = (
        adaptive_vs_time is not None
        and adaptive_vs_time
        >= gates["adaptive_vs_time_aware_mean_paired_saving_minimum_percent"]
    )
    gate_c = (
        adaptive_vs_fixed is not None
        and adaptive_vs_fixed
        >= gates["adaptive_vs_fixed_mean_paired_saving_minimum_percent"]
    )
    gate_d = (
        adaptive["native_action_influence_fraction"]
        >= gates["calibration_executed_influence_minimum_fraction"]
        and adaptive["counterfactual_different_first_action_fraction"]
        >= gates[
            "calibration_counterfactual_distinct_first_action_minimum_fraction"
        ]
    )
    gate_e = (
        resource["projected_three_seed_runtime_hours"]
        <= gates["projected_three_seed_runtime_maximum_hours"]
        and resource["projected_retained_artifacts_gb"]
        <= gates["projected_retained_artifacts_maximum_gb"]
    )
    passed = hard_integrity and all((gate_a, gate_b, gate_c, gate_d, gate_e))
    if not hard_integrity:
        decision = "SAFETY_OR_INTEGRITY_STOP"
    elif passed:
        decision = "STRONG_MPC_FOUND_SLACKTRADER_V2_CANDIDATE"
    elif (
        adaptive_vs_time is not None
        and adaptive_vs_time > 0.0
        and gate_a
    ):
        decision = "MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR"
    else:
        decision = "MPC_V2_NOT_COMPETITIVE"
    result = {
        "schema_version": 1,
        "source_commit": canonical["source_commit"],
        "calibration_panel": {
            "id": calibration["panel_id"],
            "ordered_sha256": calibration["panel_ordered_sha256"],
            "episodes": calibration["budget"]["episodes"],
            "selected": calibration["selected"],
            "influence": calibration["influence_gate_calibration"],
        },
        "capability": capability,
        "resource": resource,
        "gates": {
            "hard_integrity": hard_integrity,
            "A_hard_feasibility": gate_a,
            "B_strong_mpc_energy": gate_b,
            "C_dynamic_price_advantage": gate_c,
            "D_executed_control_authority": gate_d,
            "E_resources": gate_e,
            "all_mandatory": passed,
        },
        "decision": decision,
        "mpc_v2_calibration_complete": True,
        "mpc_v2_capability_complete": True,
        "strong_causal_mpc_found": passed,
        "dynamic_time_price_useful": gate_c and gate_d,
        "slacktrader_v2_recommended": passed,
        "slacktrader_training_authorized": False,
        "slacktrader_training_enabled": False,
        "paper_state": "SEALED_UNINSPECTED",
        "paper_episodes": 0,
        "validation_episodes": 0,
        "rl_training_episodes": 0,
        "rl_gradient_updates": 0,
        "frozen_method_sha256": canonical["frozen_method_sha256"],
    }
    return result


def _fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_calibration(results: dict[str, Any]) -> str:
    rows = [
        "# Calibration Results",
        "",
        f"Panel: `{results['panel_id']}` (`{results['panel_ordered_sha256']}`).",
        f"Budget consumed exactly: {results['budget']['episodes']} episodes.",
        "",
        "## Fixed-price candidates",
        "",
        "| Action | Lambda (kWh/s) | RSR | Mean energy (kWh) | Planner time (s) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate in results["fixed_candidates"]:
        summary = candidate["summary"]
        rows.append(
            f"| {candidate['normalized_action']:.2f} | "
            f"{candidate['price_kwh_per_s']:.8f} | {summary['rsr']}/12 | "
            f"{_fmt(summary['mean_feasible_energy_kwh'])} | "
            f"{summary['planner_solve_time_s']:.3f} |"
        )
    rows.extend(
        [
            "",
            "## Adaptive candidates",
            "",
            (
                "| High/low reserve (s) | RSR | Mean energy (kWh) | "
                "Influence | Counterfactual first-action difference |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in results["adaptive_candidates"]:
        summary = candidate["summary"]
        rows.append(
            f"| {candidate['high_reserve_s']:.0f}/{candidate['low_reserve_s']:.0f} | "
            f"{summary['rsr']}/12 | {_fmt(summary['mean_feasible_energy_kwh'])} | "
            f"{100 * summary['native_action_influence_fraction']:.3f}% | "
            f"{100 * summary['counterfactual_different_first_action_fraction']:.3f}% |"
        )
    selected = results["selected"]
    rows.extend(
        [
            "",
            "## Frozen selection",
            "",
            f"Fixed action `{selected['fixed_normalized_action']}` "
            f"(lambda `{selected['fixed_price_kwh_per_s']}` kWh/s); adaptive "
            f"reserves `{selected['adaptive_high_reserve_s']}/"
            f"{selected['adaptive_low_reserve_s']}` s. No Capability outcome "
            "was observed before this freeze.",
            "",
        ]
    )
    return "\n".join(rows)


def render_capability(result: dict[str, Any], final: dict[str, Any]) -> str:
    rows = [
        "# Capability Results",
        "",
        f"Panel: `{result['panel_id']}` (`{result['panel_ordered_sha256']}`).",
        "Exactly 72 controller-task pairs were executed (18 per controller).",
        "",
        (
            "| Controller | RSR | Completion | Deadline | Speed | Mean/median "
            "feasible energy (kWh) | Mean time (s) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in result["controller_order"]:
        summary = result["arms"][name]["summary"]
        rows.append(
            f"| {name} | {summary['rsr']}/18 | "
            f"{summary['completion_count']}/18 | "
            f"{summary['deadline_compliance_count']}/18 | "
            f"{summary['speed_compliance_count']}/18 | "
            f"{_fmt(summary['mean_feasible_energy_kwh'])}/"
            f"{_fmt(summary['median_feasible_energy_kwh'])} | "
            f"{summary['mean_travel_time_s']:.3f} |"
        )
    comparisons = result["paired_feasible_energy"]
    adaptive_time = comparisons["adaptive_vs_time_aware"][
        "arithmetic_mean_saving_percent"
    ]
    adaptive_fixed = comparisons["adaptive_vs_fixed"][
        "arithmetic_mean_saving_percent"
    ]
    fixed_time = comparisons["fixed_vs_time_aware"][
        "arithmetic_mean_saving_percent"
    ]
    rows.extend(
        [
            "",
            "## Paired energy",
            "",
            (
                "Adaptive versus TIME_AWARE: "
                f"{_fmt(adaptive_time, 3)}%."
            ),
            (
                "Adaptive versus FixedPrice V2: "
                f"{_fmt(adaptive_fixed, 3)}%."
            ),
            (
                "FixedPrice V2 versus TIME_AWARE: "
                f"{_fmt(fixed_time, 3)}%."
            ),
            "",
            "The privileged reference was preregistered as omitted, so no new",
            "privileged-reference gap is claimed.",
            "",
            f"Scientific classification: `{final['decision']}`.",
            "",
        ]
    )
    return "\n".join(rows)


def render_resource(resource: dict[str, Any]) -> str:
    return f"""# Resource Report

Capability measured {resource['measured_planner_solve_count']} planner solves in
{resource['measured_planner_solve_time_s']:.3f} s
({resource['measured_seconds_per_solve']:.6f} s/solve,
{resource['measured_solves_per_second']:.3f} solves/s). The two MPC arms used
{resource['measured_mpc_wall_time_s']:.3f} s wall time.

At the measured rate, 900,000 native training steps at 1 Hz planning plus six
18-task Diagnostic passes for three seeds project to
{resource['projected_three_seed_runtime_hours']:.3f} h. Conservative retained
transition/diagnostic/checkpoint accounting projects
{resource['projected_retained_artifacts_gb']:.3f} GB. Runtime is measurement-
based; storage is an explicit conservative accounting estimate, not a measured
future run.
"""


def render_results(result: dict[str, Any]) -> str:
    gates = result["gates"]
    return f"""# Scientific Result

Decision: `{result['decision']}`.

| Gate | Passed |
| --- | --- |
| A — hard feasibility | {gates['A_hard_feasibility']} |
| B — at least 3% vs TIME_AWARE | {gates['B_strong_mpc_energy']} |
| C — at least 1% Adaptive vs Fixed | {gates['C_dynamic_price_advantage']} |
| D — executed price authority | {gates['D_executed_control_authority']} |
| E — resources | {gates['E_resources']} |
| Safety/integrity | {gates['hard_integrity']} |

The result is preserved without threshold changes or post-Capability tuning.
No RL training, Validation or Paper evaluation occurred. SlackTrader V1 remains
`RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT`; Paper remains `SEALED_UNINSPECTED`.

```text
mpc_v2_calibration_complete = true
mpc_v2_capability_complete = true

strong_causal_mpc_found = {str(result['strong_causal_mpc_found']).lower()}
dynamic_time_price_useful = {str(result['dynamic_time_price_useful']).lower()}
slacktrader_v2_recommended = {str(result['slacktrader_v2_recommended']).lower()}

slacktrader_training_authorized = false
slacktrader_training_enabled = false

paper_state = SEALED_UNINSPECTED
```
"""


def finalize() -> dict[str, Any]:
    result = build_results()
    calibration = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8"))
    capability = json.loads(CAPABILITY_JSON.read_text(encoding="utf-8"))
    _write_json(RESULTS_JSON, result)
    status = {
        key: result[key]
        for key in (
            "mpc_v2_calibration_complete",
            "mpc_v2_capability_complete",
            "strong_causal_mpc_found",
            "dynamic_time_price_useful",
            "slacktrader_v2_recommended",
            "slacktrader_training_authorized",
            "slacktrader_training_enabled",
            "paper_state",
        )
    }
    status.update(
        {
            "schema_version": 1,
            "state": "SCIENTIFIC_DECISION_COMPLETE",
            "decision": result["decision"],
        }
    )
    _write_json(PREPARATION_STATUS, status)
    (HERE / "CALIBRATION_RESULTS.md").write_text(
        render_calibration(calibration), encoding="utf-8"
    )
    (HERE / "CAPABILITY_RESULTS.md").write_text(
        render_capability(capability, result), encoding="utf-8"
    )
    (HERE / "RESOURCE_REPORT.md").write_text(
        render_resource(result["resource"]), encoding="utf-8"
    )
    (HERE / "RESULTS.md").write_text(
        render_results(result), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    final = finalize()
    print(json.dumps({"decision": final["decision"]}, sort_keys=True))
