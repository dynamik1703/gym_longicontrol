"""Read-only aggregation and reporting for completed Validation V2 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .constants import (
    AUTHORIZATION,
    CONTROLLER_ORDER,
    EXPECTED_ACTOR_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    PAIR_LEDGER,
    PAIRWISE_ANALYSIS,
    POLICY_SEEDS,
    SPLIT_LEDGER,
    TRANSACTION_LEASE,
    V2_ID,
    V2_ORDERED_SHA256,
    V2_TRANSACTION_ID,
    VALIDATION_INTEGRITY_MD,
    VALIDATION_RESULTS,
    VALIDATION_RESULTS_MD,
)
from .integrity import (
    load_frozen_policies,
    verify_frozen_scientific_sources,
    verify_incident_evidence,
)
from .ledger import aggregate_completed, atomic_json, canonical_sha256, load_json

BOOTSTRAP_REPLICATES = 20_000
DIAGNOSTIC_REFERENCE = {
    "H11": {"rsr": 18, "episodes": 18, "mean_energy_kwh": 0.177847},
    "H29": {"rsr": 18, "episodes": 18, "mean_energy_kwh": 0.177834},
    "H47": {"rsr": 18, "episodes": 18, "mean_energy_kwh": 0.177694},
}
INFLUENCE_THRESHOLDS = {
    "maximum_override_fraction": 0.85,
    "minimum_influence_fraction": 0.10,
    "minimum_influenced_episode_count": 4,
}


def _atomic_text(path: Path, content: str) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "median": None,
            "maximum": None,
        }
    floats = [float(value) for value in values]
    return {
        "count": len(floats),
        "minimum": min(floats),
        "mean": fmean(floats),
        "median": median(floats),
        "maximum": max(floats),
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_mean_interval(values: Sequence[float], *, label: str) -> dict[str, Any]:
    if not values:
        return {
            "confidence_level": 0.95,
            "lower": None,
            "upper": None,
            "replicates": BOOTSTRAP_REPLICATES,
            "sampling_unit": "track",
        }
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    generator = random.Random(seed)
    count = len(values)
    means = sorted(
        fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return {
        "confidence_level": 0.95,
        "lower": _percentile(means, 0.025),
        "upper": _percentile(means, 0.975),
        "replicates": BOOTSTRAP_REPLICATES,
        "deterministic_rng_seed": seed,
        "sampling_unit": "track",
        "interpretation": "descriptive_percentile_interval",
    }


def _controller_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = [row["result"]["outcome"] for row in rows]
    feasible = [row for row in outcomes if row["feasible"]]
    targets = [float(value) for row in outcomes for value in row["target_speed_m_s"]]
    proposals = sum(int(row["proposal_count"]) for row in outcomes)
    overrides = sum(int(row["override_count"]) for row in outcomes)
    influenced = sum(int(row["native_action_influence_count"]) for row in outcomes)
    reasons = Counter()
    for row in outcomes:
        reasons.update(row["intervention_reasons"])
    return {
        "episode_count": len(outcomes),
        "rsr": sum(bool(row["feasible"]) for row in outcomes),
        "completion_count": sum(bool(row["completed"]) for row in outcomes),
        "deadline_compliant_count": sum(
            bool(row["deadline_compliant"]) for row in outcomes
        ),
        "speed_compliant_count": sum(bool(row["speed_compliant"]) for row in outcomes),
        "travel_time_s": _distribution([row["travel_time_s"] for row in outcomes]),
        "feasible_signed_net_energy_kwh": _distribution(
            [row["signed_net_energy_kwh"] for row in feasible]
        ),
        "failure_modes": dict(
            sorted(Counter(row["exclusive_failure_mode"] for row in outcomes).items())
        ),
        "target_speed_m_s": _distribution(targets),
        "shield": {
            "proposal_count": proposals,
            "override_count": overrides,
            "override_fraction": overrides / proposals if proposals else 0.0,
            "native_action_influence_count": influenced,
            "native_action_influence_fraction": (
                influenced / proposals if proposals else 0.0
            ),
            "influenced_episode_count": sum(
                int(row["native_action_influence_count"]) > 0 for row in outcomes
            ),
            "intervention_reasons": dict(sorted(reasons.items())),
        },
        "deadline_bound_falsifications": [
            {
                "numeric_seed": row["numeric_seed"],
                **falsification,
            }
            for row in outcomes
            for falsification in row["deadline_bound_falsifications"]
        ],
    }


def _paired_comparison(
    episodes: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    baseline: str,
    treatment: str,
    required_feasible: Sequence[str],
    label: str,
) -> dict[str, Any]:
    rows = []
    common = sorted(set(episodes[baseline]) & set(episodes[treatment]))
    for numeric_seed in common:
        if not all(
            episodes[name][numeric_seed]["feasible"] for name in required_feasible
        ):
            continue
        baseline_energy = float(episodes[baseline][numeric_seed]["energy_kwh"])
        treatment_energy = float(episodes[treatment][numeric_seed]["energy_kwh"])
        if baseline_energy == 0.0:
            raise ZeroDivisionError("paired percentage baseline energy is zero")
        difference = baseline_energy - treatment_energy
        rows.append(
            {
                "numeric_seed": numeric_seed,
                "baseline_energy_kwh": baseline_energy,
                "treatment_energy_kwh": treatment_energy,
                "energy_difference_kwh": difference,
                "absolute_energy_difference_kwh": abs(difference),
                "saving_percent": 100.0 * difference / baseline_energy,
            }
        )
    percentages = [row["saving_percent"] for row in rows]
    differences = [row["energy_difference_kwh"] for row in rows]
    absolute_differences = [row["absolute_energy_difference_kwh"] for row in rows]
    return {
        "baseline": baseline,
        "treatment": treatment,
        "required_feasible_controllers": list(required_feasible),
        "paired_track_count": len(rows),
        "per_track": rows,
        "saving_percent": {
            **_distribution(percentages),
            "bootstrap_95_percent_interval_for_mean": _bootstrap_mean_interval(
                percentages, label=f"{label}:percent"
            ),
        },
        "energy_difference_kwh": {
            **_distribution(differences),
            "absolute_magnitude": _distribution(absolute_differences),
            "bootstrap_95_percent_interval_for_mean": _bootstrap_mean_interval(
                differences, label=f"{label}:kwh"
            ),
        },
    }


def _influence_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    shield = summary["shield"]
    checks = {
        "override_fraction_at_most_0_85": shield["override_fraction"]
        <= INFLUENCE_THRESHOLDS["maximum_override_fraction"],
        "influence_fraction_at_least_0_10": shield["native_action_influence_fraction"]
        >= INFLUENCE_THRESHOLDS["minimum_influence_fraction"],
        "at_least_four_influenced_episodes": shield["influenced_episode_count"]
        >= INFLUENCE_THRESHOLDS["minimum_influenced_episode_count"],
    }
    return {
        "thresholds": dict(INFLUENCE_THRESHOLDS),
        "checks": checks,
        "passed": all(checks.values()),
    }


def analyze_completed() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ledger = load_json(PAIR_LEDGER)
    recovery = aggregate_completed(PAIR_LEDGER)
    authorization = load_json(AUTHORIZATION)
    lease = load_json(TRANSACTION_LEASE)
    split = load_json(SPLIT_LEDGER)
    if authorization.get("v2_state") != "COMPLETED_SEALED":
        raise RuntimeError("completed V2 authorization is not sealed")
    if authorization.get("v2_validation_authorized") or authorization.get(
        "v2_validation_enabled"
    ):
        raise RuntimeError("V2 evaluation authorization remains enabled")
    if lease.get("state") != "COMPLETED_SEALED":
        raise RuntimeError("V2 transaction lease is not permanently sealed")
    rows_by_controller = {
        controller: [
            pair
            for pair in ledger["pairs"]
            if pair["identity"]["controller_id"] == controller
        ]
        for controller in CONTROLLER_ORDER
    }
    summaries = {
        controller: _controller_summary(rows)
        for controller, rows in rows_by_controller.items()
    }
    episodes = {
        controller: {
            int(pair["identity"]["numeric_seed"]): pair["result"]["outcome"]
            for pair in rows
        }
        for controller, rows in rows_by_controller.items()
    }
    pairwise: dict[str, Any] = {
        "schema_version": 1,
        "study": V2_ID,
        "transaction_id": V2_TRANSACTION_ID,
        "split_sha256": V2_ORDERED_SHA256,
        "method": {
            "statistic": "arithmetic mean of per-track saving percentages",
            "energy_subset": (
                "registered tracks satisfying the stated joint-feasibility filter"
            ),
            "sampling_unit": "track; learned policy seeds are replications",
            "bootstrap": {
                "kind": "descriptive percentile bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "confidence_level": 0.95,
                "deterministic_seed": (
                    "SHA-256 derived separately for each comparison and statistic"
                ),
            },
        },
        "time_aware_vs_causal60": _paired_comparison(
            episodes,
            baseline="CAUSAL60",
            treatment="TIME_AWARE",
            required_feasible=("CAUSAL60", "TIME_AWARE"),
            label="TIME_AWARE_vs_CAUSAL60",
        ),
        "hybrid": {},
    }
    for controller in POLICY_SEEDS:
        pairwise["hybrid"][controller] = {
            "vs_causal60": _paired_comparison(
                episodes,
                baseline="CAUSAL60",
                treatment=controller,
                required_feasible=("CAUSAL60", controller),
                label=f"{controller}_vs_CAUSAL60",
            ),
            "vs_time_aware": _paired_comparison(
                episodes,
                baseline="TIME_AWARE",
                treatment=controller,
                required_feasible=("CAUSAL60", "TIME_AWARE", controller),
                label=f"{controller}_vs_TIME_AWARE",
            ),
        }
    influence = {
        controller: _influence_gate(summaries[controller])
        for controller in POLICY_SEEDS
    }
    feasibility_pass = all(summaries[name]["rsr"] >= 35 for name in POLICY_SEEDS)
    influence_pass = all(influence[name]["passed"] for name in POLICY_SEEDS)
    causal_pass = all(
        pairwise["hybrid"][name]["vs_causal60"]["saving_percent"]["mean"] is not None
        and pairwise["hybrid"][name]["vs_causal60"]["saving_percent"]["mean"] >= 10.0
        for name in POLICY_SEEDS
    )
    time_aware_pass = all(
        pairwise["hybrid"][name]["vs_time_aware"]["saving_percent"]["mean"] is not None
        and pairwise["hybrid"][name]["vs_time_aware"]["saving_percent"]["mean"] >= 3.0
        for name in POLICY_SEEDS
    )
    if not feasibility_pass:
        classification = "FEASIBILITY_LOSS"
    elif not influence_pass:
        classification = "SHIELD_DOMINATES_POLICY"
    elif not causal_pass:
        classification = "FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN"
    elif not time_aware_pass:
        classification = "ENERGY_GAIN_EXPLAINED_BY_NONLEARNING_CONTROL"
    else:
        classification = "INCREMENTAL_HYBRID_ENERGY_GAIN"
    policies, _ = load_frozen_policies()
    incident = verify_incident_evidence()
    sources = verify_frozen_scientific_sources()
    result_hashes = [pair["result_sha256"] for pair in ledger["pairs"]]
    integrity_checks = {
        "exactly_180_unique_completed_pairs": len(set(result_hashes)) == 180
        and ledger["performance_episode_count"] == 180,
        "no_pending_started_or_interrupted_pairs": Counter(
            pair["state"] for pair in ledger["pairs"]
        )
        == {"COMPLETED": 180},
        "ledger_completed_sealed": ledger["state"] == "COMPLETED_SEALED",
        "lease_completed_sealed": lease["state"] == "COMPLETED_SEALED"
        and lease["completed_pair_count"] == 180,
        "authorization_disabled_after_completion": not authorization[
            "v2_validation_authorized"
        ]
        and not authorization["v2_validation_enabled"],
        "five_frozen_controller_identities": tuple(ledger["controller_order"])
        == CONTROLLER_ORDER,
        "split_unchanged": ledger["split_sha256"] == V2_ORDERED_SHA256
        and split["ordered_sha256"] == V2_ORDERED_SHA256,
        "transaction_id_unchanged": ledger["run_transaction_id"] == V2_TRANSACTION_ID
        and lease["transaction_id"] == V2_TRANSACTION_ID,
        "checkpoint_hashes_unchanged": all(
            policies[name]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256[name]
            for name in POLICY_SEEDS
        ),
        "actor_hashes_unchanged": all(
            policies[name]["actor_parameter_sha256"] == EXPECTED_ACTOR_SHA256[name]
            for name in POLICY_SEEDS
        ),
        "training_counters_unchanged": all(
            policies[name]["native_training_transitions"] == 300_000
            and policies[name]["gradient_updates"] == 30_000
            for name in POLICY_SEEDS
        ),
        "scientific_sources_unchanged": sources["passed"],
        "v1_remains_incident_aborted": incident["passed"]
        and authorization["v1_state"] == "INCIDENT_ABORTED",
        "paper_sealed_uninspected": authorization["paper_state"] == "SEALED_UNINSPECTED"
        and not authorization["paper_authorized"]
        and authorization["paper_episodes"] == 0
        and authorization["paper_content_fingerprints"] == 0,
        "result_hashes_valid": all(
            canonical_sha256(pair["result"]) == pair["result_sha256"]
            for pair in ledger["pairs"]
        ),
        "zero_deadline_bound_falsifications": all(
            not summaries[name]["deadline_bound_falsifications"]
            for name in CONTROLLER_ORDER
        ),
    }
    integrity = {
        "schema_version": 1,
        "study": V2_ID,
        "post_incident_amendment": True,
        "transaction_id": V2_TRANSACTION_ID,
        "split_sha256": V2_ORDERED_SHA256,
        "result_set_sha256": recovery["result_set_sha256"],
        "aggregation_evaluator_calls": recovery["evaluation_calls"],
        "checks": integrity_checks,
        "passed": all(integrity_checks.values()),
        "lease": lease,
        "authorization": authorization,
        "policies": policies,
        "scientific_sources": sources,
        "v1_incident": incident,
    }
    if not integrity["passed"]:
        failed = sorted(name for name, passed in integrity_checks.items() if not passed)
        raise RuntimeError(f"post-execution V2 integrity failed: {failed}")
    diagnostic_comparison = {
        name: {
            "diagnostic": DIAGNOSTIC_REFERENCE[name],
            "validation": {
                "rsr": summaries[name]["rsr"],
                "episodes": summaries[name]["episode_count"],
                "completion_count": summaries[name]["completion_count"],
                "deadline_compliant_count": summaries[name]["deadline_compliant_count"],
                "speed_compliant_count": summaries[name]["speed_compliant_count"],
                "mean_feasible_energy_kwh": summaries[name][
                    "feasible_signed_net_energy_kwh"
                ]["mean"],
                "mean_travel_time_s": summaries[name]["travel_time_s"]["mean"],
                "target_speed_m_s": summaries[name]["target_speed_m_s"],
                "shield": summaries[name]["shield"],
            },
        }
        for name in POLICY_SEEDS
    }
    results = {
        "schema_version": 1,
        "study": V2_ID,
        "state": "COMPLETED_SEALED",
        "post_incident_amendment": True,
        "transaction_id": V2_TRANSACTION_ID,
        "split_sha256": V2_ORDERED_SHA256,
        "planned_pairs": 180,
        "completed_pairs": 180,
        "pending_pairs": 0,
        "started_pairs": 0,
        "interrupted_unrecoverable_pairs": 0,
        "controller_summaries": summaries,
        "hybrid_pooled_descriptive_rsr": {
            "successes": sum(summaries[name]["rsr"] for name in POLICY_SEEDS),
            "episodes": 108,
            "caveat": (
                "36 track instances under three learned-seed replications, "
                "not 108 independent tracks"
            ),
        },
        "influence_gate": influence,
        "acceptance": {
            "feasibility_minimum_per_seed": "35/36",
            "mean_paired_saving_vs_causal_minimum_percent": 10.0,
            "mean_paired_saving_vs_time_aware_minimum_percent": 3.0,
            "feasibility_passed_all_hybrid_seeds": feasibility_pass,
            "influence_gate_passed_all_hybrid_seeds": influence_pass,
            "causal_energy_target_passed_all_hybrid_seeds": causal_pass,
            "incremental_time_aware_target_passed_all_hybrid_seeds": time_aware_pass,
            "classification": classification,
            "rl_incremental_benefit_over_nonlearning": classification
            == "INCREMENTAL_HYBRID_ENERGY_GAIN",
        },
        "diagnostic_to_validation": diagnostic_comparison,
        "pairwise_analysis_sha256": canonical_sha256(pairwise),
        "integrity_sha256": canonical_sha256(integrity),
    }
    return results, pairwise, integrity


def _fmt(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _results_markdown(results: Mapping[str, Any], pairwise: Mapping[str, Any]) -> str:
    summaries = results["controller_summaries"]
    lines = [
        "# Hybrid RL Validation V2 Results",
        "",
        (
            "> **Amended post-incident Validation.** V1 remains permanently "
            "`INCIDENT_ABORTED`: one CAUSAL60 episode and one TIME_AWARE episode "
            "completed without persisted results, and one Hybrid environment was "
            "reset without an action. V1 was not replayed or relabeled."
        ),
        "",
        f"Transaction: `{V2_TRANSACTION_ID}`",
        f"Ordered split SHA-256: `{V2_ORDERED_SHA256}`",
        (
            "Ledger: `COMPLETED_SEALED` — 180 completed, 0 pending, "
            "0 started, 0 interrupted."
        ),
        "",
        "## Feasibility and physical outcomes",
        "",
        (
            "| Controller | RSR | Completion | Deadline | Speed | "
            "Mean travel time (s) | Mean feasible energy (kWh) |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CONTROLLER_ORDER:
        row = summaries[name]
        lines.append(
            f"| {name} | {row['rsr']}/36 | {row['completion_count']}/36 | "
            f"{row['deadline_compliant_count']}/36 | "
            f"{row['speed_compliant_count']}/36 | "
            f"{_fmt(row['travel_time_s']['mean'])} | "
            f"{_fmt(row['feasible_signed_net_energy_kwh']['mean'], 9)} |"
        )
    lines.extend(["", "## Paired feasible-only energy", ""])
    control = pairwise["time_aware_vs_causal60"]
    lines.append(
        f"TIME_AWARE versus CAUSAL60: n={control['paired_track_count']}, mean saving "
        f"{_fmt(control['saving_percent']['mean'])}%, median "
        f"{_fmt(control['saving_percent']['median'])}%, range "
        f"[{_fmt(control['saving_percent']['minimum'])}, "
        f"{_fmt(control['saving_percent']['maximum'])}]%."
    )
    lines.extend(
        [
            "",
            "| Hybrid | vs CAUSAL60 n / mean saving | vs TIME_AWARE n / mean saving |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in POLICY_SEEDS:
        causal = pairwise["hybrid"][name]["vs_causal60"]
        time_aware = pairwise["hybrid"][name]["vs_time_aware"]
        lines.append(
            f"| {name} | {causal['paired_track_count']} / "
            f"{_fmt(causal['saving_percent']['mean'])}% | "
            f"{time_aware['paired_track_count']} / "
            f"{_fmt(time_aware['saving_percent']['mean'])}% |"
        )
    lines.extend(
        [
            "",
            (
                "Full medians, ranges, absolute-unit energy differences, per-track "
                "values, and deterministic descriptive 95% bootstrap intervals are "
                "in `validation_pairwise_analysis.json`."
            ),
            "",
            "## Strategy, Shield, and decision",
            "",
        ]
    )
    for name in POLICY_SEEDS:
        row = summaries[name]
        gate = results["influence_gate"][name]
        lines.append(
            f"- {name}: target speed mean {_fmt(row['target_speed_m_s']['mean'])} m/s; "
            f"override fraction {_fmt(row['shield']['override_fraction'])}; "
            "action-influence fraction "
            f"{_fmt(row['shield']['native_action_influence_fraction'])}; "
            "influence gate "
            f"{'PASS' if gate['passed'] else 'FAIL'}."
        )
    lines.extend(
        [
            "",
            f"Frozen classification: **`{results['acceptance']['classification']}`**.",
            "",
            (
                "No training, optimizer update, checkpoint selection, reward change, "
                "Shield change, split change, post-hoc tuning, V1 replay, or Paper "
                "interaction occurred. Diagnostic-to-Validation comparisons and "
                "complete intervention/failure details are retained in the "
                "machine-readable artifacts."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _integrity_markdown(integrity: Mapping[str, Any]) -> str:
    lines = [
        "# Hybrid RL Validation V2 Integrity",
        "",
        (
            "V2 is an explicitly amended post-incident panel. V1 remains permanently "
            "`INCIDENT_ABORTED`; Paper remains `SEALED_UNINSPECTED`."
        ),
        "",
        f"Transaction: `{integrity['transaction_id']}`",
        f"Split: `{integrity['split_sha256']}`",
        f"Result set: `{integrity['result_set_sha256']}`",
        "",
        "## Final checks",
        "",
    ]
    for name, passed in integrity["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    lines.extend(
        [
            "",
            (
                "The transaction used one durable, non-reclaimable exclusive lease. "
                "Every pair was registered before evaluator entry, durably marked "
                "`STARTED` before reset, completed once, read back, and sealed with "
                "its result SHA-256. The completed ledger and lease permanently "
                "block another execution."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_final_artifacts() -> dict[str, Any]:
    results, pairwise, integrity = analyze_completed()
    atomic_json(PAIRWISE_ANALYSIS, pairwise)
    atomic_json(VALIDATION_RESULTS, results)
    _atomic_text(VALIDATION_RESULTS_MD, _results_markdown(results, pairwise))
    _atomic_text(VALIDATION_INTEGRITY_MD, _integrity_markdown(integrity))
    return {
        "validation_results_sha256": hashlib.sha256(
            VALIDATION_RESULTS.read_bytes()
        ).hexdigest(),
        "pairwise_analysis_sha256": hashlib.sha256(
            PAIRWISE_ANALYSIS.read_bytes()
        ).hexdigest(),
        "integrity_passed": integrity["passed"],
    }


def verify_final_artifacts() -> dict[str, Any]:
    results, pairwise, integrity = analyze_completed()
    checks = {
        "validation_results_exact": load_json(VALIDATION_RESULTS) == results,
        "pairwise_analysis_exact": load_json(PAIRWISE_ANALYSIS) == pairwise,
        "results_markdown_exact": VALIDATION_RESULTS_MD.read_text(encoding="utf-8")
        == _results_markdown(results, pairwise),
        "integrity_markdown_exact": VALIDATION_INTEGRITY_MD.read_text(encoding="utf-8")
        == _integrity_markdown(integrity),
    }
    if not all(checks.values()):
        raise RuntimeError(f"final artifact verification failed: {checks}")
    return {"checks": checks, "integrity": integrity, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    payload = (
        verify_final_artifacts() if arguments.verify_only else write_final_artifacts()
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
