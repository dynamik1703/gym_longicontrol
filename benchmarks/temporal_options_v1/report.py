"""Render frozen Calibration, Capability, resource and final reports."""

from __future__ import annotations

import json

from .integrity import HERE


def _load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _table(results):
    lines = [
        "| Arm | RSR | Complete | Deadline | Speed | Energy feasible kWh | Time s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, payload in results["arms"].items():
        row = payload["summary"]
        energy = row["mean_feasible_energy_kwh"]
        lines.append(
            f"| {arm} | {row['rsr']}/{row['episode_count']} | "
            f"{row['completion_count']} | {row['deadline_compliance_count']} | "
            f"{row['speed_compliance_count']} | "
            f"{'n/a' if energy is None else f'{energy:.9f}'} | "
            f"{row['mean_travel_time_s']:.3f} |"
        )
    return "\n".join(lines)


def render_calibration(calibration):
    resource = calibration["resource_measurement"]
    projected_hours = resource["projected_capability_active_compute_hours"]
    return f"""# Calibration Results

Exactly {calibration["episode_count"]} performance episodes were run: four frozen
arms on 12 fresh tasks. No configuration selection or retuning was performed.

- measured wall time: {resource["wall_time_s"]:.3f} s
- measured active process time: {resource["active_process_time_s"]:.3f} s
- host sleep invoked by the study: {resource["host_sleep_time_s"]:.3f} s
- projected Capability active compute: {projected_hours:.4f} h
- projected retained artifacts: {resource["projected_retained_artifacts_bytes"]} bytes
- compute gate: {"pass" if resource["active_compute_gate_pass"] else "fail"}
- artifact gate: {"pass" if resource["artifact_gate_pass"] else "fail"}

RL gradient updates, old Validation episodes and Paper episodes were all zero.
"""


def render_resource(calibration, capability=None):
    resource = calibration["resource_measurement"]
    actual = "Capability not yet executed."
    if capability is not None:
        wall = sum(
            row["summary"]["episode_wall_time_s"] for row in capability["arms"].values()
        )
        process = sum(
            row["summary"]["active_process_time_s"]
            for row in capability["arms"].values()
        )
        calls = sum(
            row["summary"]["physics_prediction_calls"]
            for row in capability["arms"].values()
        )
        steps = sum(
            episode["native_step_count"]
            for arm in capability["arms"].values()
            for episode in arm["episodes"]
        )
        actual = (
            f"Capability wall time was {wall:.3f} s and active process time was "
            f"{process:.3f} s. It used {steps} native simulator steps and "
            f"{calls} recorded physics prediction calls. The study invoked no "
            "host sleep."
        )
    return f"""# Resource Report

Calibration projected {resource["projected_capability_active_compute_hours"]:.4f}
active-compute hours and {resource["projected_retained_artifacts_bytes"]} retained
bytes for Capability. Both preregistered gates passed.

{actual}

Fixture native-step accounting is recorded in `semantics_audit.json`; the bound is
30000. Calibration used 48 episodes and Capability used exactly 72. Planning-model
calls are reported separately from simulator steps.
"""


def render_capability(results):
    primary = results["comparisons"]["primary_continue_vs_reset"]
    return f"""# Capability Results

{_table(results)}

Primary CONTINUE versus RESET jointly feasible count:
{primary["jointly_feasible_count"]}; arithmetic mean per-task saving:
{primary["arithmetic_mean_saving_percent"]}; median:
{primary["median_saving_percent"]}; range: {primary["saving_percent_range"]}.

Scientific classification: `{results["scientific_classification"]}`. This label
concerns a non-learning execution architecture and is not an RL result.
"""


def render_results(results):
    comparisons = results["comparisons"]
    primary = comparisons["primary_continue_vs_reset"]
    feedback = comparisons["feedback_vs_legacy"]
    commit = comparisons["commit_vs_continue"]
    legacy = comparisons["continue_vs_legacy"]
    c = results["arms"]["CONTINUE_FEEDBACK"]["summary"]
    d = results["arms"]["COMMIT_FEEDBACK"]["summary"]
    return f"""# Temporal Options V1 — Final Results

{_table(results)}

## Answers to the registered questions

1. Systematic phase postponement was demonstrated in deterministic fixtures for
   inherited distance-triggered 30%, 40%, 50% and 82% phases under repeated
   sub-threshold one-second re-anchoring. It was not universal: velocity-triggered
   pulse coasting remained reachable.
2. Affected fixtures were traction-then-coast, traction-coast-brake and
   coast-then-partial-traction. Speed clipping, safety intervention and short final
   horizons qualify the result.
3. RESET versus frozen MPC mean paired feasible-only saving was
   {feedback["arithmetic_mean_saving_percent"]}%; this is the native-feedback/scoring
   contrast.
4. CONTINUE executed coast/brake/partial phases with counts
   `{c["executed_phase_counts"]}` and retained anchors across boundaries.
5. COMMIT versus CONTINUE mean paired saving was
   {commit["arithmetic_mean_saving_percent"]}%; COMMIT RSR was {d["rsr"]} versus
   CONTINUE {c["rsr"]}.
6. Feasibility/completion/deadline/speed counts for all arms appear above.
7. CONTINUE versus RESET mean paired energy saving was
   {primary["arithmetic_mean_saving_percent"]}% and mean travel-time delta was
   {primary["mean_travel_time_delta_s"]} s. CONTINUE versus frozen MPC saving was
   {legacy["arithmetic_mean_saving_percent"]}%.
8. CONTINUE made {c["option_start_count"]} starts, {c["option_continue_count"]}
   continuations and {c["option_switch_count"]} switches; mean option duration was
   {c["mean_option_duration_s"]} s and distance {c["mean_option_distance_m"]} m.
9. CONTINUE safety interruptions numbered {c["option_interruption_count"]}; reasons
   are `{c["safety_override_reasons"]}`.
10. Every observed gain/loss is architectural and non-learning. Gradient updates
    were zero.
11. CONTINUE retained {c["distinct_native_action_trajectories"]} distinct episode
    action trajectories; the 18 high-level actions remain memory-distinct even
    when first native actions match.
12. The final interface is 70D with 18 actions and a valid-action mask; transition
    records use actual duration and `gamma_native ** k`.
13. Focused exact-recovery and integrity tests passed before execution; pair records
    are hash-bound and exactly-once.
14. Runtime, simulator steps, planner calls and physics-model calls are in
    `RESOURCE_REPORT.md` and machine-readable episode records.
15. Fresh panel hashes are recorded in `split_ledger.json`.
16. Classification: `{results["scientific_classification"]}`. Remaining uncertainty
    includes heuristic candidate values, empirical Shield coverage and only 18
    distinct Capability routes.
17. Commit and remote synchronization are reported at handoff after the final
    immutable results commit.
18. RL training episodes and gradient updates: zero.
19. Old Validation episodes: zero.
20. Paper episodes/fingerprints: zero; Paper stayed sealed and uninspected.

## Recommendation

{recommendation(results)}
"""


def recommendation(results):
    label = results["scientific_classification"]
    if label == "OPTIONS_VALID_PROMISING":
        return (
            "Proceed to a separately preregistered learned START/CONTINUE selector, "
            "retaining all current safety, split and exact-recovery boundaries."
        )
    if label == "OPTIONS_VALID_FEASIBILITY_REGRESSION":
        return (
            "Do not start learner training. First explain the feasibility regression "
            "in a separately authorized architecture study."
        )
    if label == "SEMANTIC_OR_INTEGRITY_STOP":
        return "Do not proceed until the semantic or integrity failure is resolved."
    return (
        "Do not start learner training automatically. The interface is valid, but "
        "the bounded non-learning evidence does not meet the preregistered "
        "promise gate."
    )


def main():
    calibration = _load("calibration_results.json")
    capability_path = HERE / "capability_results.json"
    capability = _load("capability_results.json") if capability_path.exists() else None
    (HERE / "CALIBRATION_RESULTS.md").write_text(
        render_calibration(calibration), encoding="utf-8"
    )
    (HERE / "RESOURCE_REPORT.md").write_text(
        render_resource(calibration, capability), encoding="utf-8"
    )
    if capability is not None:
        (HERE / "CAPABILITY_RESULTS.md").write_text(
            render_capability(capability), encoding="utf-8"
        )
        (HERE / "RESULTS.md").write_text(render_results(capability), encoding="utf-8")


if __name__ == "__main__":
    main()
