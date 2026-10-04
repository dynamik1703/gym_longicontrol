"""Track- and seed-aware analysis for Scalar Reward Study V2."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .analysis import failure_mode, paired_energy
from .references import load_reference_results
from .v2_config import (
    DEFAULT_V2_CONFIG_PATH,
    load_v2_configuration,
    v2_configuration_sha256,
)
from .v2_evaluation import V2BenchmarkRunResult, load_v2_result


def discover_v2_results(path: str | Path) -> tuple[V2BenchmarkRunResult, ...]:
    root = Path(path)
    paths = sorted(root.rglob("*-result.json")) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"No V2 result files found under {root}")
    return tuple(load_v2_result(item) for item in paths)


def _result_summary(runs: Sequence[V2BenchmarkRunResult]) -> dict[str, Any]:
    episodes = tuple(episode for run in runs for episode in run.episodes)
    feasible_energy = [item.energy_kwh for item in episodes if item.feasible]
    seed_summaries = {
        str(run.training_seed): asdict(run.summary)
        for run in sorted(runs, key=lambda item: item.training_seed)
    }
    rsr_values = [run.summary.requirement_satisfaction_rate for run in runs]
    return {
        "reward_parameters": asdict(runs[0].reward_parameters),
        "training_seed_count": len(runs),
        "episode_count": len(episodes),
        "by_training_seed": seed_summaries,
        "mean_rsr": fmean(rsr_values),
        "min_rsr_across_training_seeds": min(rsr_values),
        "max_rsr_across_training_seeds": max(rsr_values),
        "rsr_range_across_training_seeds": max(rsr_values) - min(rsr_values),
        "mean_completion_rate": fmean(
            run.summary.completion_rate for run in runs
        ),
        "mean_speed_compliance_rate": fmean(
            run.summary.speed_compliance_rate for run in runs
        ),
        "mean_travel_time_s": fmean(item.travel_time_s for item in episodes),
        "median_travel_time_s": float(
            median(item.travel_time_s for item in episodes)
        ),
        "feasible_episode_count": len(feasible_energy),
        "mean_feasible_energy_kwh": (
            fmean(feasible_energy) if feasible_energy else None
        ),
        "median_feasible_energy_kwh": (
            float(median(feasible_energy)) if feasible_energy else None
        ),
        "failure_mode_counts": dict(
            sorted(
                Counter(
                    failure_mode(item, runs[0].task) for item in episodes
                ).items()
            )
        ),
        "mean_abs_jerk_m_s3": fmean(
            item.mean_abs_jerk_m_s3 for item in episodes
        ),
        "mean_action_total_variation": fmean(
            item.action_total_variation for item in episodes
        ),
        "mean_acceleration_sign_changes": fmean(
            item.acceleration_sign_change_count for item in episodes
        ),
    }


def _acceptance(configuration, runs, fast_reference):
    criteria = configuration.acceptance
    per_seed = {}
    for run in sorted(runs, key=lambda item: item.training_seed):
        energy_pairs = paired_energy(run.episodes, fast_reference.episodes)
        if energy_pairs:
            ratios = [
                item["left_energy_kwh"] / item["right_energy_kwh"]
                for item in energy_pairs["tracks"]
            ]
            mean_energy_ratio = fmean(ratios)
        else:
            mean_energy_ratio = None
        checks = {
            "rsr": (
                run.summary.requirement_satisfaction_rate
                >= criteria.minimum_rsr_per_training_seed
            ),
            "completion": (
                run.summary.completion_rate
                >= criteria.minimum_completion_rate_per_training_seed
            ),
            "speed_compliance": (
                run.summary.speed_compliance_rate
                >= criteria.minimum_speed_compliance_rate_per_training_seed
            ),
            "energy": (
                mean_energy_ratio is not None
                and mean_energy_ratio
                <= criteria.maximum_mean_energy_ratio_to_fast_reference
            ),
        }
        per_seed[str(run.training_seed)] = {
            "accepted": all(checks.values()),
            "checks": checks,
            "mean_energy_ratio_to_fast_reference": mean_energy_ratio,
            "paired_track_count": (
                energy_pairs["paired_track_count"] if energy_pairs else 0
            ),
        }
    rsr_values = [run.summary.requirement_satisfaction_rate for run in runs]
    seed_range = max(rsr_values) - min(rsr_values)
    seed_stability = (
        seed_range <= criteria.maximum_rsr_range_across_training_seeds
    )
    return {
        "accepted": seed_stability
        and all(item["accepted"] for item in per_seed.values()),
        "seed_stability_accepted": seed_stability,
        "rsr_range_across_training_seeds": seed_range,
        "by_training_seed": per_seed,
    }


def analyze_v2_results(
    runs: Sequence[V2BenchmarkRunResult],
    configuration,
    references,
    *,
    v1_summary: dict[str, Any] | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    expected_hash = v2_configuration_sha256(configuration)
    if {run.configuration_sha256 for run in runs} != {expected_hash}:
        raise ValueError("V2 results do not match the analysis configuration")
    expected_ids = {
        item.configuration_id for item in configuration.reward_candidates
    }
    expected_seeds = set(configuration.training_seeds)
    grouped = defaultdict(list)
    for run in runs:
        key = (
            run.evaluation_split_id,
            run.training_steps,
            run.reward_parameters.configuration_id,
        )
        grouped[key].append(run)

    validation_id = "validation-3000-3008-v1"
    exploratory_id = "v1-exploratory-1000-1008-v1"
    if require_complete:
        expected_keys = {
            (validation_id, step, identifier)
            for step in configuration.learning_curve_steps
            for identifier in expected_ids
        } | {
            (exploratory_id, step, identifier)
            for step in configuration.comparison_steps
            for identifier in expected_ids
        }
        if set(grouped) != expected_keys:
            missing = expected_keys - set(grouped)
            extra = set(grouped) - expected_keys
            raise ValueError(
                f"Incomplete V2 result set; missing={missing}, extra={extra}"
            )
        for key, values in grouped.items():
            seeds = {item.training_seed for item in values}
            if seeds != expected_seeds or len(values) != len(expected_seeds):
                raise ValueError(f"Incomplete V2 training seeds for {key}: {seeds}")

    summaries = {
        f"{split}/step-{step}/{identifier}": _result_summary(values)
        for (split, step, identifier), values in sorted(grouped.items())
    }
    fast_reference = next(
        item
        for item in references.policies
        if item.policy_id == "fast-compliant-oracle"
    )
    final_step = configuration.training.total_training_steps
    acceptance = {}
    paired_reference = []
    for identifier in sorted(expected_ids):
        final_runs = grouped.get((exploratory_id, final_step, identifier), [])
        if final_runs:
            acceptance[identifier] = _acceptance(
                configuration, final_runs, fast_reference
            )
            for run in sorted(final_runs, key=lambda item: item.training_seed):
                comparison = paired_energy(run.episodes, fast_reference.episodes)
                if comparison:
                    paired_reference.append(
                        {
                            "configuration_id": identifier,
                            "training_seed": run.training_seed,
                            **comparison,
                        }
                    )

    v1_comparison = None
    if v1_summary is not None:
        best_v1 = max(
            v1_summary["configuration_summaries"].items(),
            key=lambda item: item[1]["mean_rsr"],
        )
        final_summaries = {
            identifier: summaries[
                f"{exploratory_id}/step-{final_step}/{identifier}"
            ]
            for identifier in sorted(expected_ids)
            if f"{exploratory_id}/step-{final_step}/{identifier}" in summaries
        }
        if final_summaries:
            best_v2 = max(
                final_summaries.items(), key=lambda item: item[1]["mean_rsr"]
            )
            v1_comparison = {
                "best_v1_configuration": best_v1[0],
                "best_v1_mean_rsr": best_v1[1]["mean_rsr"],
                "best_v2_configuration": best_v2[0],
                "best_v2_mean_rsr": best_v2[1]["mean_rsr"],
                "absolute_mean_rsr_change": (
                    best_v2[1]["mean_rsr"] - best_v1[1]["mean_rsr"]
                ),
            }

    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": expected_hash,
        "result_file_count": len(runs),
        "reserved_final_test_evaluated": False,
        "acceptance_criteria": asdict(configuration.acceptance),
        "summaries": summaries,
        "acceptance": acceptance,
        "paired_energy_against_fast_reference": paired_reference,
        "v1_to_v2": v1_comparison,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_V2_CONFIG_PATH)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--v1-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    configuration = load_v2_configuration(args.config)
    v1_summary = (
        json.loads(args.v1_summary.read_text(encoding="utf-8"))
        if args.v1_summary
        else None
    )
    result = analyze_v2_results(
        discover_v2_results(args.results),
        configuration,
        load_reference_results(args.references),
        v1_summary=v1_summary,
        require_complete=not args.allow_incomplete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
