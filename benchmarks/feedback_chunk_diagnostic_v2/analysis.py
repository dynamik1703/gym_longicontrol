"""Preregistered route-level analysis for the completed Diagnostic V2 panel."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any

from .constants import RESULTS
from .utils import atomic_json, canonical_sha256, load_json

METHOD_CONTROLLERS = {
    "CHUNK3": (
        "CHUNK3_S11_300K",
        "CHUNK3_S29_300K",
        "CHUNK3_S47_300K",
    ),
    "RESIDUAL1S": (
        "RESIDUAL1S_S11_300K",
        "RESIDUAL1S_S29_300K",
        "RESIDUAL1S_S47_300K",
    ),
    "HYBRID": ("HYBRID_H11", "HYBRID_H29", "HYBRID_H47"),
    "ADAPTIVE_PRICE_MPC_V2": ("ADAPTIVE_PRICE_MPC_V2",),
    "FIXED_PRICE_MPC_V2": ("FIXED_PRICE_MPC_V2",),
    "TIME_AWARE": ("TIME_AWARE",),
    "CAUSAL60": ("CAUSAL60",),
}


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _bootstrap_mean(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"resamples": 10_000, "lower_95": None, "upper_95": None}
    generator = random.Random(11_400_000)
    estimates = []
    count = len(values)
    for _ in range(10_000):
        estimates.append(
            statistics.fmean(generator.choice(values) for _ in range(count))
        )
    estimates.sort()
    return {
        "resamples": 10_000,
        "lower_95": estimates[249],
        "upper_95": estimates[9_749],
    }


def _index(results: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    indexed: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in results["outcomes"]:
        identity = row["identity"]
        indexed[identity["controller_id"]][int(identity["numeric_seed"])] = row[
            "outcome"
        ]
    return dict(indexed)


def _controller_summary(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    ordered = [rows[seed] for seed in sorted(rows)]
    feasible_energy = [float(row["energy_kwh"]) for row in ordered if row["feasible"]]
    return {
        "route_count": len(ordered),
        "feasible_count": sum(bool(row["feasible"]) for row in ordered),
        "raw_rsr": statistics.fmean(float(row["feasible"]) for row in ordered),
        "completed_count": sum(bool(row["completed"]) for row in ordered),
        "deadline_compliant_count": sum(
            bool(row["deadline_compliant"]) for row in ordered
        ),
        "speed_compliant_count": sum(bool(row["speed_compliant"]) for row in ordered),
        "mean_feasible_energy_kwh": _mean(feasible_energy),
        "median_feasible_energy_kwh": (
            statistics.median(feasible_energy) if feasible_energy else None
        ),
        "failed_numeric_seeds": [
            seed for seed in sorted(rows) if not rows[seed]["feasible"]
        ],
        "shield": {
            "proposal_count": sum(int(row["proposal_count"]) for row in ordered),
            "override_count": sum(int(row["override_count"]) for row in ordered),
            "native_action_influence_count": sum(
                int(row["native_action_influence_count"]) for row in ordered
            ),
        },
        "chunk": {
            "chunk_count": sum(int(row["chunk_count"]) for row in ordered),
            "full_chunk_count": sum(int(row["full_chunk_count"]) for row in ordered),
            "cancelled_chunk_count": sum(
                int(row["cancelled_chunk_count"]) for row in ordered
            ),
        },
    }


def _method_routes(
    indexed: dict[str, dict[int, dict[str, Any]]], method: str
) -> dict[int, dict[str, Any]]:
    controllers = METHOD_CONTROLLERS[method]
    seeds = sorted(indexed[controllers[0]])
    rows = {}
    for seed in seeds:
        outcomes = [indexed[controller][seed] for controller in controllers]
        feasible = [bool(row["feasible"]) for row in outcomes]
        energies = [float(row["energy_kwh"]) for row in outcomes]
        rows[seed] = {
            "all_seed_outcomes_feasible": all(feasible),
            "feasible_seed_count": sum(feasible),
            "seed_feasibility": feasible,
            "method_mean_energy_kwh": statistics.fmean(energies),
            "method_mean_feasible_energy_kwh": (
                statistics.fmean(energies) if all(feasible) else None
            ),
            "training_seed_energy_population_sd_kwh": (
                statistics.pstdev(energies) if len(energies) > 1 else 0.0
            ),
        }
    return rows


def _comparison(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
) -> dict[str, Any]:
    seeds = sorted(set(left) & set(right))
    jointly_feasible = [
        seed
        for seed in seeds
        if left[seed]["all_seed_outcomes_feasible"]
        and right[seed]["all_seed_outcomes_feasible"]
    ]
    savings = []
    differences = []
    for seed in jointly_feasible:
        left_energy = float(left[seed]["method_mean_feasible_energy_kwh"])
        right_energy = float(right[seed]["method_mean_feasible_energy_kwh"])
        differences.append(right_energy - left_energy)
        savings.append(100.0 * (right_energy - left_energy) / right_energy)
    additional_failures = [
        seed
        for seed in seeds
        if right[seed]["all_seed_outcomes_feasible"]
        and not left[seed]["all_seed_outcomes_feasible"]
    ]
    return {
        "left": left_name,
        "right": right_name,
        "route_is_independent_unit": True,
        "jointly_feasible_route_count": len(jointly_feasible),
        "jointly_feasible_numeric_seeds": jointly_feasible,
        "left_additional_failure_numeric_seeds": additional_failures,
        "no_additional_failures": not additional_failures,
        "mean_per_task_percentage_saving": _mean(savings),
        "median_per_task_percentage_saving": (
            statistics.median(savings) if savings else None
        ),
        "mean_absolute_energy_difference_kwh": _mean(differences),
        "median_absolute_energy_difference_kwh": (
            statistics.median(differences) if differences else None
        ),
        "route_bootstrap_95": _bootstrap_mean(savings),
        "per_task_percentage_savings": dict(zip(jointly_feasible, savings)),
    }


def analyze() -> dict[str, Any]:
    results = load_json(RESULTS)
    if results.get("state") != "COMPLETED_INDEPENDENTLY_VERIFIED":
        raise RuntimeError("analysis requires sealed verified results")
    indexed = _index(results)
    if any(len(rows) != 36 for rows in indexed.values()) or len(indexed) != 13:
        raise RuntimeError("analysis input is not the frozen 13 x 36 panel")
    raw = {
        controller: _controller_summary(rows)
        for controller, rows in sorted(indexed.items())
    }
    method_routes = {
        method: _method_routes(indexed, method) for method in METHOD_CONTROLLERS
    }
    comparisons = {
        "chunk_vs_residual": _comparison(
            method_routes["CHUNK3"],
            method_routes["RESIDUAL1S"],
            left_name="CHUNK3",
            right_name="RESIDUAL1S",
        ),
        "chunk_vs_adaptive": _comparison(
            method_routes["CHUNK3"],
            method_routes["ADAPTIVE_PRICE_MPC_V2"],
            left_name="CHUNK3",
            right_name="ADAPTIVE_PRICE_MPC_V2",
        ),
        "chunk_vs_fixed": _comparison(
            method_routes["CHUNK3"],
            method_routes["FIXED_PRICE_MPC_V2"],
            left_name="CHUNK3",
            right_name="FIXED_PRICE_MPC_V2",
        ),
        "chunk_vs_hybrid": _comparison(
            method_routes["CHUNK3"],
            method_routes["HYBRID"],
            left_name="CHUNK3",
            right_name="HYBRID",
        ),
        "residual_vs_adaptive": _comparison(
            method_routes["RESIDUAL1S"],
            method_routes["ADAPTIVE_PRICE_MPC_V2"],
            left_name="RESIDUAL1S",
            right_name="ADAPTIVE_PRICE_MPC_V2",
        ),
        "hybrid_vs_time_aware": _comparison(
            method_routes["HYBRID"],
            method_routes["TIME_AWARE"],
            left_name="HYBRID",
            right_name="TIME_AWARE",
        ),
    }
    seed_specific_regressions = []
    for numeric_seed in sorted(method_routes["CHUNK3"]):
        for controller in METHOD_CONTROLLERS["CHUNK3"]:
            training_seed = controller.split("_S", 1)[1].split("_", 1)[0]
            residual = f"RESIDUAL1S_S{training_seed}_300K"
            if not indexed[controller][numeric_seed]["feasible"]:
                feasible_comparators = [
                    name
                    for name in (
                        residual,
                        "ADAPTIVE_PRICE_MPC_V2",
                        "FIXED_PRICE_MPC_V2",
                    )
                    if indexed[name][numeric_seed]["feasible"]
                ]
                if feasible_comparators:
                    seed_specific_regressions.append(
                        {
                            "numeric_seed": numeric_seed,
                            "chunk_controller": controller,
                            "feasible_comparators": feasible_comparators,
                        }
                    )
    residual_target = comparisons["chunk_vs_residual"]
    adaptive_target = comparisons["chunk_vs_adaptive"]
    residual_success = bool(
        residual_target["no_additional_failures"]
        and residual_target["mean_per_task_percentage_saving"] is not None
        and residual_target["mean_per_task_percentage_saving"] >= 1.0
    )
    adaptive_success = bool(
        adaptive_target["no_additional_failures"]
        and adaptive_target["mean_per_task_percentage_saving"] is not None
        and adaptive_target["mean_per_task_percentage_saving"] >= 3.0
    )
    if seed_specific_regressions:
        classification = "CHUNK_V2_FEASIBILITY_REGRESSION"
    elif residual_success and adaptive_success:
        classification = "CHUNK_V2_PROMISING_FOR_VALIDATION"
    else:
        classification = "CHUNK_V2_NO_INCREMENTAL_GAIN"
    analysis = {
        "schema_version": 1,
        "raw_feasibility_reported_first": True,
        "controller_results": raw,
        "method_route_results": method_routes,
        "comparisons": comparisons,
        "training_seed_variability": {
            method: {
                "mean_per_route_population_sd_kwh": statistics.fmean(
                    row["training_seed_energy_population_sd_kwh"]
                    for row in routes.values()
                )
            }
            for method, routes in method_routes.items()
            if len(METHOD_CONTROLLERS[method]) == 3
        },
        "seed_specific_feasibility_regressions": seed_specific_regressions,
        "frozen_targets": {
            "chunk_vs_residual_no_extra_failures_and_1_percent": residual_success,
            "chunk_vs_adaptive_no_extra_failures_and_3_percent": adaptive_success,
        },
        "classification": classification,
        "limitations": [
            "Diagnostic panel, not an independently authorized final Validation",
            "36 routes; route bootstrap intervals are descriptive",
            "three fixed training seeds per learned method",
            "method-level route means keep three seeds grouped; 108 outcomes are "
            "not treated as independent routes",
        ],
    }
    analysis["analysis_sha256"] = canonical_sha256(analysis)
    atomic_json(RESULTS.parent / "paired_comparisons.json", analysis)
    return analysis


if __name__ == "__main__":
    import json

    print(json.dumps(analyze(), sort_keys=True))
