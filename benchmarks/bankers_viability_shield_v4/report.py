"""Preregistered descriptive summaries after immutable raw outcomes are sealed."""

import json
import time
from collections import Counter
from itertools import combinations

import numpy as np

from .controller_adapters import BANKERS, METHODS
from .integrity import HERE, verify_sources
from .serialization import atomic_json, file_hash, read_evidence, require


def distribution(values):
    return (
        dict(
            n=len(values),
            minimum=min(values),
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
            maximum=max(values),
        )
        if values
        else None
    )


def report():
    start = time.process_time()
    seal = json.loads((HERE / "development_seal.json").read_text())
    require(
        file_hash(HERE / "pair_ledger.json") == seal["ledger_sha256"],
        "raw ledger not sealed",
    )
    ledger = json.loads((HERE / "pair_ledger.json").read_text())
    require(len(ledger["pairs"]) == 72, "matrix")
    manifest = json.loads((HERE / "execution_manifest.json").read_text())
    verify_sources(manifest["identity"])
    methods = {m: {} for m in METHODS}
    native_counts = {m: 0 for m in METHODS}
    divergences = {}
    authority = {}
    diagnostics = []
    mask_summary = {}
    for method in BANKERS:
        authority[method] = dict(
            states=0,
            meaningful=0,
            positive=0,
            negative=0,
            braking_only=0,
            backup_available=0,
            executed_changed=0,
            counts=[],
            distinct=[],
            spans=[],
            maximum_accelerations=[],
            selected_deviations=[],
            conservative_braking=0,
            selected_margins=[],
            binding_reasons=Counter(),
        )
        mask_summary[method] = dict(
            states=0,
            proposal_safety_rejected=0,
            proposal_off_grid=0,
            executed_changed=0,
            native_action_difference=[],
            mask_s=[],
            final_s=[],
        )
    for attempt, pair in ledger["pairs"].items():
        require(pair["state"] == "COMPLETED", "incomplete result")
        result = pair["result"]
        method = result["method"]
        seed = result["task"]["numeric_seed"]
        methods[method][str(seed)] = result
        require(
            file_hash(HERE / "sealed_results" / f"{attempt}.json")
            == pair["result_file_sha256"],
            "sealed pair result",
        )
        meta = result["evidence"]
        rows = read_evidence(
            HERE / "evidence" / meta["file"], meta, manifest["identity"], attempt
        )
        native_counts[method] += len(rows)
        if method == "BANKERS_MASKED_CAUSAL60":
            first = next(
                (r for r in rows if r["executed_action"] != r["proposal_action"]), None
            )
            if first:
                admitted = [c for c in first["certificates"] if c["accepted"]]
                divergences[str(seed)] = dict(
                    step=first["step"],
                    native_pre_time_s=first["causal_pre_state"]["elapsed_time_s"],
                    causal_pre_state=first["causal_pre_state"],
                    proposal=first["proposal_action"],
                    executed=first["executed_action"],
                    continuous_proposal_accepted=first["causal60_proposal_certificate"][
                        "accepted"
                    ],
                    cause="ACTION_GRID_DISCRETIZATION"
                    if first["causal60_proposal_certificate"]["accepted"]
                    else "BANKER_SAFETY_REJECTION",
                    proposal_rejection_reason=first["causal60_proposal_certificate"][
                        "rejection_reason"
                    ],
                    admitted_actions=[c["proposed_action"] for c in admitted],
                    higher_admitted_actions=[
                        c["proposed_action"]
                        for c in admitted
                        if c["proposed_action"] > first["executed_action"]
                    ],
                    alternative_policy_proven=False,
                    masked_later_speed_samples=[
                        dict(
                            step=r["step"],
                            time_s=r["native_time_s"],
                            position_m=r["actual_next_state"]["position_m"],
                            velocity_m_s=r["actual_next_state"]["velocity_m_s"],
                        )
                        for r in rows
                        if r["step"] >= first["step"]
                        and (r["step"] % 100 == 0 or r is rows[-1])
                    ],
                )
        if method not in BANKERS:
            continue
        a = authority[method]
        d = mask_summary[method]
        for row in rows:
            au = row["authority"]
            a["states"] += 1
            a["meaningful"] += int(
                au["distinct_velocities"] >= 2 and au["velocity_span_m_s"] >= 0.05
            )
            a["positive"] += int(au["positive_acceleration_choices"] > 0)
            a["negative"] += int(au["negative_acceleration_choices"] > 0)
            a["braking_only"] += int(
                au["positive_acceleration_choices"] == 0
                and au["negative_acceleration_choices"] > 0
            )
            a["backup_available"] += int(row["safe_action_mask"][0])
            a["executed_changed"] += int(
                row["executed_action"] != row["proposal_action"]
            )
            a["counts"].append(au["count"])
            a["distinct"].append(au["distinct_velocities"])
            a["spans"].append(au["velocity_span_m_s"])
            a["maximum_accelerations"].append(
                max(
                    (
                        c["predicted_next_state"]["velocity_m_s"]
                        - row["causal_pre_state"]["velocity_m_s"]
                    )
                    / 0.1
                    for c in row["certificates"]
                    if c["accepted"]
                )
            )
            a["selected_deviations"].append(
                abs(row["executed_action"] - row["proposal_action"])
            )
            a["conservative_braking"] += int(
                row["executed_action"] < 0
                and row["executed_action"] < row["proposal_action"]
            )
            a["selected_margins"].append(
                row["selected_certificate"]["minimum_velocity_slack_m_s"]
            )
            a["binding_reasons"].update(
                c["rejection_reason"] for c in row["certificates"] if not c["accepted"]
            )
            d["states"] += 1
            d["executed_changed"] += int(
                row["executed_action"] != row["proposal_action"]
            )
            d["proposal_off_grid"] += int(
                row["proposal_action"] not in [i / 10 for i in range(-10, 11)]
            )
            if row["causal60_proposal_certificate"]:
                d["proposal_safety_rejected"] += int(
                    not row["causal60_proposal_certificate"]["accepted"]
                )
            d["native_action_difference"].append(
                abs(row["executed_action"] - row["proposal_action"])
            )
            d["mask_s"].append(row["timing"]["mask_s"])
            d["final_s"].append(row["timing"]["final_s"])
            if row["diagnostic_sampled"]:
                counts = Counter()
                for new, old in zip(row["certificates"], row["inherited_comparison"]):
                    if new["accepted"] == old["accepted"]:
                        counts["agreement"] += 1
                        continue
                    counts["disagreement"] += 1
                    if not new["right_endpoint_compliant"] or not old["right_endpoint"]:
                        counts["endpoint_unsafe"] += 1
                    if not new["backup_feasible"] or not old["known_braking"]:
                        counts["backup_not_certified"] += 1
                    if new["unknown_preview_recoverable"] != old["unknown_preview"]:
                        counts["different_preview_assumptions"] += 1
                    if (
                        old["right_endpoint"]
                        and old["known_braking"]
                        and old["unknown_preview"]
                        and not old["deadline_viable"]
                        and not old["fallback_identical_exception"]
                    ):
                        counts["deadline_only_inherited_rejection"] += 1
                    p = new["predicted_next_state"]
                    q = old["predicted_next_state"]
                    if any(p[k] != q[k] for k in ("position_m", "velocity_m_s")):
                        counts["numerical_discrepancy"] += 1
                diagnostics.append(
                    dict(
                        attempt=attempt,
                        step=row["step"],
                        input_sha256=row["certificates"][0]["input_sha256"],
                        counts=dict(counts),
                    )
                )
    summaries = {}
    for method, entries in methods.items():
        require(len(entries) == 18, "method denominator")
        feasible = [seed for seed, r in entries.items() if r["feasible"]]
        summaries[method] = dict(
            raw_rsr=len(feasible),
            denominator=18,
            completion=sum(r["metrics"]["completed"] for r in entries.values()),
            deadline=sum(
                r["metrics"]["travel_time_s"] <= 140 for r in entries.values()
            ),
            speed=sum(
                r["metrics"]["max_speed_violation_m_s"] == 0 for r in entries.values()
            ),
            feasible_task_ids=feasible,
            failure_ids=[s for s in entries if s not in feasible],
            feasible_only_energy_kwh={
                s: entries[s]["metrics"]["energy_kwh"] for s in feasible
            },
            tasks={
                s: dict(
                    metrics=r["metrics"],
                    feasible=r["feasible"],
                    actual_progress_m=r["actual_progress_m"],
                )
                for s, r in entries.items()
            },
        )
    reference = methods["CAUSAL60"]
    masked = methods["BANKERS_MASKED_CAUSAL60"]
    additional = [
        s for s in reference if reference[s]["feasible"] and not masked[s]["feasible"]
    ]
    for seed in additional:
        meta = reference[seed]["evidence"]
        attempt = meta["attempt"]
        causal_rows = read_evidence(
            HERE / "evidence" / meta["file"], meta, manifest["identity"], attempt
        )
        masked_meta = masked[seed]["evidence"]
        masked_rows = read_evidence(
            HERE / "evidence" / masked_meta["file"],
            masked_meta,
            manifest["identity"],
            masked_meta["attempt"],
        )
        detail = divergences.get(seed)
        if detail:
            detail["subsequent_speed_and_position_differences"] = [
                dict(
                    step=i,
                    native_time_s=masked_rows[i]["native_time_s"],
                    causal_velocity_m_s=causal_rows[i]["actual_next_state"][
                        "velocity_m_s"
                    ],
                    masked_velocity_m_s=masked_rows[i]["actual_next_state"][
                        "velocity_m_s"
                    ],
                    masked_minus_causal_position_m=masked_rows[i]["actual_next_state"][
                        "position_m"
                    ]
                    - causal_rows[i]["actual_next_state"]["position_m"],
                )
                for i in range(detail["step"], min(len(causal_rows), len(masked_rows)))
                if i % 100 == 0
            ]
            detail["route_progress_times_s"] = {
                str(x): {
                    method: next(
                        (
                            r["native_time_s"]
                            for r in rows
                            if r["actual_next_state"]["position_m"] >= x
                        ),
                        None,
                    )
                    for method, rows in (
                        ("CAUSAL60", causal_rows),
                        ("BANKERS_MASKED_CAUSAL60", masked_rows),
                    )
                }
                for x in (100, 250, 500, 750, 1000)
            }
    matched = dict(
        additional_masked_failures=additional,
        additional_masked_deadline_failures=[
            s for s in additional if masked[s]["metrics"]["travel_time_s"] > 140
        ],
        coverage_preserved=not additional,
        task_travel_time_changes_s={
            s: masked[s]["metrics"]["travel_time_s"]
            - reference[s]["metrics"]["travel_time_s"]
            for s in reference
        },
        additional_failure_details={
            s: dict(
                causal60=reference[s]["metrics"],
                masked=masked[s]["metrics"],
                first_divergence=divergences.get(s),
                causal_progress_m=reference[s]["actual_progress_m"],
                masked_progress_m=masked[s]["actual_progress_m"],
            )
            for s in additional
        },
    )
    energy = {}
    for left, right in combinations(METHODS, 2):
        joint = [
            s
            for s in methods[left]
            if methods[left][s]["feasible"] and methods[right][s]["feasible"]
        ]
        energy[left + "__" + right] = dict(
            jointly_feasible=joint,
            n=len(joint),
            signed_energy_kwh={
                s: {
                    left: methods[left][s]["metrics"]["energy_kwh"],
                    right: methods[right][s]["metrics"]["energy_kwh"],
                }
                for s in joint
            },
            right_minus_left_kwh={
                s: methods[right][s]["metrics"]["energy_kwh"]
                - methods[left][s]["metrics"]["energy_kwh"]
                for s in joint
            },
        )
    authority_pass = True
    for m, a in authority.items():
        n = a["states"]
        a["meaningful_fraction"] = a["meaningful"] / n
        a["positive_fraction"] = a["positive"] / n
        a["gate_pass"] = (
            a["meaningful_fraction"] >= 0.25 and a["positive_fraction"] >= 0.05
        )
        authority_pass &= a["gate_pass"]
        a["action_count_histogram"] = {
            str(k): v for k, v in Counter(a["counts"]).items()
        }
        a["distinct_velocity_histogram"] = {
            str(k): v for k, v in Counter(a["distinct"]).items()
        }
        for k in (
            "counts",
            "distinct",
            "spans",
            "selected_margins",
            "maximum_accelerations",
            "selected_deviations",
        ):
            a[k] = distribution(a[k])
        a["binding_reasons"] = dict(a["binding_reasons"])
    for d in mask_summary.values():
        for k in ("native_action_difference", "mask_s", "final_s"):
            d[k] = distribution(d[k])
    speed_pass = all(
        methods[m][s]["metrics"]["max_speed_violation_m_s"] == 0
        for m in BANKERS
        for s in methods[m]
    )
    classification = (
        "BANKERS_V4_COUNTEREXAMPLE_FOUND"
        if not speed_pass
        else "BANKERS_V4_PROGRESS_INSUFFICIENT"
        if additional
        else "BANKERS_V4_ACTION_AUTHORITY_LOW"
        if not authority_pass
        else "BANKERS_V4_READY_FOR_MASKED_RL_DESIGN"
    )
    result = dict(
        study="BANKERS_VIABILITY_SHIELD_V4",
        classification=classification,
        source_identity=manifest["identity"],
        completed_pairs=72,
        split_sha256=manifest["ordered_panel_sha256"],
        controllers=summaries,
        matched_causal60=matched,
        joint_feasible_energy=energy,
        authority=authority,
        mask_diagnostics=mask_summary,
        inherited_diagnostic_sampled_states=len(diagnostics),
        inherited_diagnostic_eligible_states=sum(
            a["states"] for a in authority.values()
        ),
        speed_gate_pass=speed_pass,
        authority_gate_pass=authority_pass,
        progress_gate_pass=not additional,
        independent_replay_pairs=72,
        independent_native_steps=sum(native_counts.values()),
        independent_native_steps_per_method=native_counts,
        independent_grid_candidates=21 * sum(native_counts[m] for m in BANKERS),
        speed_certificate_failures=0,
        first_divergences=divergences,
        rl_training_transitions=0,
        rl_gradient_updates=0,
        paper="SEALED_UNINSPECTED",
        historical_validation="UNTOUCHED",
        report_cpu_s=time.process_time() - start,
    )
    atomic_json(HERE / "predicate_diagnostics.json", diagnostics, immutable=True)
    atomic_json(HERE / "results.json", result, immutable=True)
    print("SCIENTIFIC RESULT", classification, flush=True)
    print(
        {
            m: (s["raw_rsr"], s["completion"], s["deadline"], s["speed"])
            for m, s in summaries.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    report()
