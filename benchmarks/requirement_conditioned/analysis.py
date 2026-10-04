"""Analyze the preregistered requirement-conditioned study."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import RequirementEpisodeEvaluation, RequirementRunResult, load_result

DEFAULT_FROZEN_V2_RESULTS = Path("benchmarks/constrained_rl_v2/results.json")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Ranks require a non-empty finite vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return Spearman's rho with average ranks, or ``None`` if undefined."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.ptp(left_ranks) == 0 or np.ptp(right_ranks) == 0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _episode_summary(
    episodes: Iterable[RequirementEpisodeEvaluation],
) -> dict[str, Any]:
    values = tuple(episodes)
    if not values:
        raise ValueError("At least one episode is required")
    feasible_energy = [item.energy_kwh for item in values if item.feasible]
    return {
        "episode_count": len(values),
        "feasible_count": sum(item.feasible for item in values),
        "requirement_satisfaction_rate": sum(item.feasible for item in values)
        / len(values),
        "completion_rate": sum(item.completed for item in values) / len(values),
        "deadline_compliance_rate": sum(item.deadline_met for item in values)
        / len(values),
        "speed_compliance_rate": sum(item.speed_compliant for item in values)
        / len(values),
        "mean_travel_time_s": fmean(item.travel_time_s for item in values),
        "median_travel_time_s": float(median(item.travel_time_s for item in values)),
        "mean_feasible_energy_kwh": (
            fmean(feasible_energy) if feasible_energy else None
        ),
        "median_feasible_energy_kwh": (
            float(median(feasible_energy)) if feasible_energy else None
        ),
        "mean_max_speed_violation_m_s": fmean(
            item.max_speed_violation_m_s for item in values
        ),
        "mean_integrated_speed_violation_m": fmean(
            item.integrated_speed_violation_m for item in values
        ),
    }


def _profile_difference(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("Action profiles must share a shape")
    return float(np.mean(np.abs(left_array - right_array)))


def analyze_controllability(
    episodes: Sequence[RequirementEpisodeEvaluation], acceptance
) -> dict[str, Any]:
    """Compute the preregistered within-policy, within-track diagnostics."""

    grouped: dict[tuple[int, int], list[RequirementEpisodeEvaluation]] = defaultdict(
        list
    )
    for item in episodes:
        # The caller has one result per training seed, so it attaches that seed below.
        training_seed = getattr(item, "_training_seed", None)
        if training_seed is None:
            raise ValueError("Controllability episodes must include training seed")
        grouped[(training_seed, item.track_seed)].append(item)

    travel_checks: list[bool] = []
    energy_checks: list[bool] = []
    travel_deltas: list[float] = []
    energy_deltas: list[float] = []
    tight_loose_time_deltas: list[float] = []
    tight_loose_energy_deltas: list[float] = []
    interpolation_checks: list[bool] = []
    interpolation_energy_checks: list[bool] = []
    group_rows = []
    for (training_seed, track_seed), raw_values in sorted(grouped.items()):
        values = sorted(raw_values, key=lambda item: item.requirement_margin_s)
        margins = [item.requirement_margin_s for item in values]
        if margins != [20.0, 30.0, 40.0, 50.0, 60.0]:
            raise ValueError(
                f"Incomplete requirement group: {(training_seed, track_seed)}"
            )
        group_travel_checks = []
        group_energy_checks = []
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                delta_time = right.travel_time_s - left.travel_time_s
                check_time = (
                    right.travel_time_s
                    >= left.travel_time_s
                    - acceptance.travel_time_monotonicity_tolerance_s
                )
                travel_checks.append(check_time)
                group_travel_checks.append(check_time)
                travel_deltas.append(delta_time)
                if left.feasible and right.feasible:
                    delta_energy = right.energy_kwh - left.energy_kwh
                    check_energy = (
                        right.energy_kwh
                        <= left.energy_kwh
                        + acceptance.energy_monotonicity_tolerance_kwh
                    )
                    energy_checks.append(check_energy)
                    group_energy_checks.append(check_energy)
                    energy_deltas.append(delta_energy)
        time_range = max(item.travel_time_s for item in values) - min(
            item.travel_time_s for item in values
        )
        feasible = [item for item in values if item.feasible]
        energy_range = (
            max(item.energy_kwh for item in feasible)
            - min(item.energy_kwh for item in feasible)
            if len(feasible) >= 2
            else None
        )
        action_difference = _profile_difference(
            values[0].action_profile, values[-1].action_profile
        )
        sensitive_reasons = []
        if time_range >= acceptance.requirement_sensitive_travel_time_range_s:
            sensitive_reasons.append("travel_time_range")
        if (
            energy_range is not None
            and energy_range >= acceptance.requirement_sensitive_energy_range_kwh
        ):
            sensitive_reasons.append("feasible_energy_range")
        if action_difference >= acceptance.requirement_sensitive_mean_action_difference:
            sensitive_reasons.append("action_profile_difference")
        time_by_margin = {
            item.requirement_margin_s: item.travel_time_s for item in values
        }
        episode_by_margin = {item.requirement_margin_s: item for item in values}
        local_interpolation = []
        local_interpolation_energy = []
        for margin, lower, upper in ((30.0, 20.0, 40.0), (50.0, 40.0, 60.0)):
            tolerance = acceptance.travel_time_monotonicity_tolerance_s
            directional = (
                time_by_margin[margin] >= time_by_margin[lower] - tolerance
                and time_by_margin[upper] >= time_by_margin[margin] - tolerance
            )
            interpolation_checks.append(directional)
            local_interpolation.append(directional)
            triplet = [episode_by_margin[value] for value in (lower, margin, upper)]
            if all(item.feasible for item in triplet):
                energy_tolerance = acceptance.energy_monotonicity_tolerance_kwh
                energy_directional = (
                    triplet[1].energy_kwh <= triplet[0].energy_kwh + energy_tolerance
                    and triplet[2].energy_kwh
                    <= triplet[1].energy_kwh + energy_tolerance
                )
                interpolation_energy_checks.append(energy_directional)
                local_interpolation_energy.append(energy_directional)
        tight_loose_time_deltas.append(
            values[-1].travel_time_s - values[0].travel_time_s
        )
        if values[0].feasible and values[-1].feasible:
            tight_loose_energy_deltas.append(
                values[-1].energy_kwh - values[0].energy_kwh
            )
        energy_rho_values = [item for item in values if item.feasible]
        group_rows.append(
            {
                "training_seed": training_seed,
                "track_seed": track_seed,
                "travel_time_range_s": time_range,
                "feasible_energy_range_kwh": energy_range,
                "tight_to_loose_action_profile_mean_absolute_difference": (
                    action_difference
                ),
                "requirement_sensitive": bool(sensitive_reasons),
                "sensitivity_reasons": sensitive_reasons,
                "travel_pairwise_monotonicity_rate": sum(group_travel_checks)
                / len(group_travel_checks),
                "energy_pair_count": len(group_energy_checks),
                "energy_pairwise_monotonicity_rate": (
                    sum(group_energy_checks) / len(group_energy_checks)
                    if group_energy_checks
                    else None
                ),
                "deadline_travel_time_spearman": spearman_correlation(
                    margins, [item.travel_time_s for item in values]
                ),
                "deadline_feasible_energy_spearman": (
                    spearman_correlation(
                        [item.requirement_margin_s for item in energy_rho_values],
                        [item.energy_kwh for item in energy_rho_values],
                    )
                    if len(energy_rho_values) >= 2
                    else None
                ),
                "interpolation_travel_directional_rate": sum(local_interpolation)
                / len(local_interpolation),
                "interpolation_feasible_energy_comparison_count": len(
                    local_interpolation_energy
                ),
            }
        )

    def defined_mean(key):
        values = [row[key] for row in group_rows if row[key] is not None]
        return fmean(values) if values else None

    sensitive = sum(row["requirement_sensitive"] for row in group_rows)
    reason_counts = Counter(
        reason for row in group_rows for reason in row["sensitivity_reasons"]
    )
    return {
        "group_count": len(group_rows),
        "ordered_pair_count": len(travel_checks),
        "travel_time_pairwise_monotonicity_rate": sum(travel_checks)
        / len(travel_checks),
        "median_ordered_pair_travel_time_delta_s": float(median(travel_deltas)),
        "median_tight_to_loose_travel_time_delta_s": float(
            median(tight_loose_time_deltas)
        ),
        "mean_deadline_travel_time_spearman": defined_mean(
            "deadline_travel_time_spearman"
        ),
        "feasible_energy_pair_count": len(energy_checks),
        "feasible_energy_pairwise_monotonicity_rate": (
            sum(energy_checks) / len(energy_checks) if energy_checks else None
        ),
        "median_feasible_ordered_pair_energy_delta_kwh": (
            float(median(energy_deltas)) if energy_deltas else None
        ),
        "median_feasible_tight_to_loose_energy_delta_kwh": (
            float(median(tight_loose_energy_deltas))
            if tight_loose_energy_deltas
            else None
        ),
        "mean_deadline_feasible_energy_spearman": defined_mean(
            "deadline_feasible_energy_spearman"
        ),
        "requirement_sensitive_group_count": sensitive,
        "requirement_sensitive_group_rate": sensitive / len(group_rows),
        "sensitivity_reason_counts": dict(sorted(reason_counts.items())),
        "interpolation_travel_comparison_count": len(interpolation_checks),
        "interpolation_travel_directional_rate": sum(interpolation_checks)
        / len(interpolation_checks),
        "interpolation_feasible_energy_comparison_count": len(
            interpolation_energy_checks
        ),
        "interpolation_feasible_energy_directional_rate": (
            sum(interpolation_energy_checks) / len(interpolation_energy_checks)
            if interpolation_energy_checks
            else None
        ),
        "groups": group_rows,
    }


class _SeededEpisode:
    """Transparent proxy used only to add the policy seed to a frozen episode."""

    def __init__(self, episode: RequirementEpisodeEvaluation, training_seed: int):
        self._episode = episode
        self._training_seed = training_seed

    def __getattr__(self, name):
        return getattr(self._episode, name)


def _load_primary_runs(results_directory: Path, configuration):
    runs = []
    target = configuration.simulator_step_budget
    for seed in configuration.training_seeds:
        path = (
            results_directory
            / f"training-seed-{seed}"
            / f"target-{target:06d}"
            / "validation-result.json"
        )
        runs.append(load_result(path))
    return tuple(runs)


def _validate_run(run: RequirementRunResult, configuration, *, split: str, target: int):
    if run.configuration_sha256 != configuration_sha256(configuration):
        raise ValueError("Result does not match the frozen configuration")
    if run.evaluation_split_id != split or run.simulator_step_target != target:
        raise ValueError("Unexpected evaluation split or checkpoint")
    reserved = set(configuration.track_splits.paper_final_test_reserved)
    if any(item.track_seed in reserved for item in run.episodes):
        raise ValueError("Reserved paper-final tracks occur in results")


def _training_diagnostics(results_directory: Path, configuration) -> dict[str, Any]:
    pooled: dict[float, list[dict[str, Any]]] = defaultdict(list)
    by_seed = {}
    for seed in configuration.training_seeds:
        rows = _read_json(
            results_directory / f"training-seed-{seed}" / "training-diagnostics.json"
        )
        if not rows:
            raise ValueError("Training diagnostics are empty")
        exposure = Counter(float(row["requirement_margin_s"]) for row in rows)
        for row in rows:
            pooled[float(row["requirement_margin_s"])].append(row)
        required = (
            "lagrange_speed",
            "lagrange_deadline",
            "speed_integral_m",
            "deadline_deficit_integral_s",
            "loss/q0",
            "loss/q1",
            "loss/q2",
            "loss/actor_total",
            "alpha",
        )
        nonfinite = any(
            not np.isfinite(float(row[key])) for row in rows for key in required
        )
        by_seed[str(seed)] = {
            "completed_training_episode_count": len(rows),
            "requirement_exposure_counts": {
                f"{margin:g}": exposure[margin] for margin in sorted(exposure)
            },
            "maximum_exposure_count_difference": max(exposure.values())
            - min(exposure.values()),
            "final_multipliers": {
                "speed": rows[-1]["lagrange_speed"],
                "deadline": rows[-1]["lagrange_deadline"],
            },
            "maximum_multipliers": {
                "speed": max(row["lagrange_speed"] for row in rows),
                "deadline": max(row["lagrange_deadline"] for row in rows),
            },
            "has_nonfinite_required_diagnostic": bool(nonfinite),
            "final_simulator_steps": rows[-1]["simulator_steps"],
        }
        if by_seed[str(seed)]["maximum_exposure_count_difference"] > 1:
            raise ValueError(
                f"Training requirements are not balanced for seed {seed}: {exposure}"
            )

    def margin_summary(rows):
        return {
            "episode_count": len(rows),
            "completion_rate": sum(row["completed"] for row in rows) / len(rows),
            "mean_speed_cost_return_m": fmean(row["speed_integral_m"] for row in rows),
            "median_speed_cost_return_m": float(
                median(row["speed_integral_m"] for row in rows)
            ),
            "mean_deadline_cost_return_s": fmean(
                row["deadline_deficit_integral_s"] for row in rows
            ),
            "median_deadline_cost_return_s": float(
                median(row["deadline_deficit_integral_s"] for row in rows)
            ),
            "mean_energy_kwh": fmean(row["energy_kwh"] for row in rows),
        }

    return {
        "by_training_seed": by_seed,
        "pooled_by_training_requirement_margin": {
            f"{margin:g}": margin_summary(rows)
            for margin, rows in sorted(pooled.items())
        },
    }


def _criterion(value: float | None, threshold: float, *, maximum=False):
    passed = value is not None and (
        value <= threshold if maximum else value >= threshold
    )
    return {"value": value, "threshold": threshold, "passed": passed}


def _decision_gate(criteria: dict[str, dict[str, Any]], by_margin) -> tuple[str, str]:
    if all(item["passed"] for item in criteria.values()):
        return (
            "F",
            "All satisfaction, controllability, and interpolation criteria pass.",
        )
    rsr_keys = ("overall_rsr", "minimum_requirement_rsr", "minimum_seed_rsr")
    rsr_pass = all(criteria[key]["passed"] for key in rsr_keys)
    time_sensitivity_keys = (
        "travel_time_pairwise_monotonicity",
        "requirement_sensitive_group_rate",
    )
    time_sensitive = all(criteria[key]["passed"] for key in time_sensitivity_keys)
    interpolation_keys = ("interpolation_rsr_gap", "interpolation_directional_rate")
    interpolation_pass = all(criteria[key]["passed"] for key in interpolation_keys)
    if rsr_pass and time_sensitive and interpolation_pass:
        return (
            "A",
            "Core satisfaction, time control, and interpolation pass, but the "
            "strong energy-adaptation criterion does not.",
        )
    if rsr_pass and not time_sensitive:
        return "B", "Satisfaction passes, but behavioral sensitivity does not."
    if rsr_pass and not interpolation_pass:
        return "C", "Seen requirements pass, but interpolation does not."
    tight = by_margin["20"]["requirement_satisfaction_rate"]
    looser = max(
        by_margin[key]["requirement_satisfaction_rate"]
        for key in ("30", "40", "50", "60")
    )
    if tight < 0.5 and looser >= 0.5:
        return "D", "The tight requirement fails while at least one looser level works."
    return (
        "E",
        "Pooled satisfaction is below target with broad multi-requirement degradation.",
    )


def analyze(
    results_directory: str | Path,
    configuration,
    *,
    feasibility_path: str | Path,
    frozen_v2_results_path: str | Path = DEFAULT_FROZEN_V2_RESULTS,
) -> dict[str, Any]:
    root = Path(results_directory)
    final_target = configuration.simulator_step_budget
    primary_runs = _load_primary_runs(root, configuration)
    for run in primary_runs:
        _validate_run(
            run,
            configuration,
            split="validation-requirements-v1",
            target=final_target,
        )
    if {run.training_seed for run in primary_runs} != set(configuration.training_seeds):
        raise ValueError("Final validation training seeds are incomplete")
    primary_episodes = tuple(item for run in primary_runs for item in run.episodes)
    if len(primary_episodes) != 135:
        raise ValueError("Expected 135 final primary validation episodes")
    expected_tracks = set(configuration.track_splits.validation)
    for run in primary_runs:
        if {item.track_seed for item in run.episodes} != expected_tracks:
            raise ValueError("Validation tracks are incomplete")
        combinations = {
            (item.track_seed, item.requirement_margin_s) for item in run.episodes
        }
        expected_combinations = {
            (track, margin)
            for track in configuration.track_splits.validation
            for margin in configuration.requirements.evaluation_margins_s
        }
        if combinations != expected_combinations or len(run.episodes) != 45:
            raise ValueError("Validation track/requirement matrix is incomplete")

    learning_curves = []
    checkpoint_results = []
    for seed in configuration.training_seeds:
        for target in configuration.simulator_step_checkpoints:
            run = load_result(
                root
                / f"training-seed-{seed}"
                / f"target-{target:06d}"
                / "validation-result.json"
            )
            _validate_run(
                run,
                configuration,
                split="validation-requirements-v1",
                target=target,
            )
            checkpoint_results.append(run)
            development_run = load_result(
                root
                / f"training-seed-{seed}"
                / f"target-{target:06d}"
                / "development-result.json"
            )
            _validate_run(
                development_run,
                configuration,
                split="development-requirements-v1",
                target=target,
            )
            if {item.track_seed for item in development_run.episodes} != set(
                configuration.track_splits.development_calibration
            ):
                raise ValueError("Development tracks are incomplete")
            checkpoint_results.append(development_run)
            learning_curves.append(
                {
                    "training_seed": seed,
                    "simulator_step_target": target,
                    "simulator_steps": run.simulator_steps,
                    "overall_rsr": run.overall_summary["requirement_satisfaction_rate"],
                    "completion_rate": run.overall_summary["completion_rate"],
                    "deadline_compliance_rate": run.overall_summary[
                        "deadline_compliance_rate"
                    ],
                    "speed_compliance_rate": run.overall_summary[
                        "speed_compliance_rate"
                    ],
                    "rsr_by_margin": {
                        key: value["requirement_satisfaction_rate"]
                        for key, value in run.summaries_by_margin.items()
                    },
                }
            )

    by_margin = {
        f"{margin:g}": _episode_summary(
            item for item in primary_episodes if item.requirement_margin_s == margin
        )
        for margin in configuration.requirements.evaluation_margins_s
    }
    learning_stability = []
    for seed in configuration.training_seeds:
        rows = sorted(
            (row for row in learning_curves if row["training_seed"] == seed),
            key=lambda row: row["simulator_step_target"],
        )
        peak = max(rows, key=lambda row: row["overall_rsr"])
        learning_stability.append(
            {
                "training_seed": seed,
                "peak_rsr": peak["overall_rsr"],
                "peak_simulator_step_target": peak["simulator_step_target"],
                "final_rsr": rows[-1]["overall_rsr"],
                "peak_to_final_rsr_drop": peak["overall_rsr"] - rows[-1]["overall_rsr"],
                "final_minus_250k_rsr": rows[-1]["overall_rsr"]
                - rows[-2]["overall_rsr"],
            }
        )
    seen = tuple(
        item
        for item in primary_episodes
        if item.requirement_margin_s in configuration.requirements.training_margins_s
    )
    interpolation = tuple(
        item
        for item in primary_episodes
        if item.requirement_margin_s
        in configuration.requirements.interpolation_margins_s
    )
    by_seed = {
        str(run.training_seed): {
            **_episode_summary(run.episodes),
            "by_margin": {
                key: _episode_summary(
                    item
                    for item in run.episodes
                    if item.requirement_margin_s == float(key)
                )
                for key in by_margin
            },
            "checkpoint_multipliers": {
                "speed": run.checkpoint_multipliers[0],
                "deadline": run.checkpoint_multipliers[1],
            },
            "simulator_steps": run.simulator_steps,
            "gradient_updates": run.gradient_updates,
        }
        for run in primary_runs
    }
    seeded = tuple(
        _SeededEpisode(item, run.training_seed)
        for run in primary_runs
        for item in run.episodes
    )
    controllability = analyze_controllability(seeded, configuration.acceptance)

    canonical_runs = []
    for seed in configuration.training_seeds:
        run = load_result(
            root
            / f"training-seed-{seed}"
            / f"target-{final_target:06d}"
            / "canonical-140-result.json"
        )
        _validate_run(
            run,
            configuration,
            split="validation-canonical-140-v1",
            target=final_target,
        )
        canonical_runs.append(run)
    canonical_episodes = tuple(item for run in canonical_runs for item in run.episodes)
    canonical_summary = _episode_summary(canonical_episodes)
    frozen_v2 = _read_json(frozen_v2_results_path)["summaries"][
        "validation-3000-3008-v2/simulator-target-300000"
    ]
    frozen_v2_feasible_count = round(frozen_v2["mean_rsr"] * frozen_v2["episode_count"])
    if (frozen_v2_feasible_count, frozen_v2["episode_count"]) != (21, 27):
        raise ValueError("Frozen Constrained V2 comparison no longer equals 21/27")
    canonical = {
        "requirement_conditioned": {
            **canonical_summary,
            "by_training_seed": {
                str(run.training_seed): _episode_summary(run.episodes)
                for run in canonical_runs
            },
        },
        "frozen_constrained_v2": {
            "source": str(frozen_v2_results_path),
            "feasible_count": frozen_v2_feasible_count,
            "episode_count": frozen_v2["episode_count"],
            "requirement_satisfaction_rate": frozen_v2["mean_rsr"],
        },
        "feasible_count_difference": (
            canonical_summary["feasible_count"] - frozen_v2_feasible_count
        ),
        "rsr_difference": canonical_summary["requirement_satisfaction_rate"]
        - frozen_v2["mean_rsr"],
    }

    overall = _episode_summary(primary_episodes)
    seen_summary = _episode_summary(seen)
    interpolation_summary = _episode_summary(interpolation)
    interpolation_gap = (
        seen_summary["requirement_satisfaction_rate"]
        - interpolation_summary["requirement_satisfaction_rate"]
    )
    acceptance = configuration.acceptance
    criteria = {
        "overall_rsr": _criterion(
            overall["requirement_satisfaction_rate"], acceptance.minimum_overall_rsr
        ),
        "minimum_requirement_rsr": _criterion(
            min(item["requirement_satisfaction_rate"] for item in by_margin.values()),
            acceptance.minimum_rsr_per_requirement,
        ),
        "minimum_seed_rsr": _criterion(
            min(item["requirement_satisfaction_rate"] for item in by_seed.values()),
            acceptance.minimum_overall_rsr_per_training_seed,
        ),
        "travel_time_pairwise_monotonicity": _criterion(
            controllability["travel_time_pairwise_monotonicity_rate"],
            acceptance.minimum_travel_time_pairwise_monotonicity,
        ),
        "energy_pairwise_monotonicity": _criterion(
            controllability["feasible_energy_pairwise_monotonicity_rate"],
            acceptance.minimum_energy_pairwise_monotonicity,
        ),
        "requirement_sensitive_group_rate": _criterion(
            controllability["requirement_sensitive_group_rate"],
            acceptance.minimum_requirement_sensitive_track_seed_rate,
        ),
        "interpolation_rsr_gap": _criterion(
            interpolation_gap, acceptance.maximum_interpolation_rsr_gap, maximum=True
        ),
        "interpolation_directional_rate": _criterion(
            controllability["interpolation_travel_directional_rate"],
            acceptance.minimum_interpolation_directional_rate,
        ),
    }
    gate, reason = _decision_gate(criteria, by_margin)
    feasibility = _read_json(feasibility_path)
    if feasibility.get("reserved_final_test_evaluated") is not False:
        raise ValueError("Physics feasibility artifact does not preserve the seal")
    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": configuration_sha256(configuration),
        "result_file_count": len(checkpoint_results) + len(canonical_runs),
        "primary_validation_episode_count": len(primary_episodes),
        "reserved_final_test_evaluated": False,
        "physics_feasibility": feasibility,
        "final_validation": {
            "overall": overall,
            "seen_requirements": seen_summary,
            "interpolation_requirements": interpolation_summary,
            "interpolation_rsr_gap": interpolation_gap,
            "by_margin": by_margin,
            "by_training_seed": by_seed,
        },
        "controllability": controllability,
        "learning_curves": learning_curves,
        "learning_stability": learning_stability,
        "training_diagnostics": _training_diagnostics(root, configuration),
        "canonical_140": canonical,
        "acceptance": criteria,
        "decision_gate": gate,
        "decision_reason": reason,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--feasibility",
        type=Path,
        default=Path("benchmarks/requirement_conditioned/feasibility.json"),
    )
    parser.add_argument(
        "--frozen-v2-results", type=Path, default=DEFAULT_FROZEN_V2_RESULTS
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = analyze(
        args.results_directory,
        load_configuration(args.config),
        feasibility_path=args.feasibility,
        frozen_v2_results_path=args.frozen_v2_results,
    )
    print(_write_json(args.output, payload))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
