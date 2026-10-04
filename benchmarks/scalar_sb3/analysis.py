"""Algorithm-independent analysis of the SB3 scalar baseline results."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from benchmarks.scalar_sac.analysis import failure_mode, paired_energy
from benchmarks.scalar_sac.references import ReferenceResults, load_reference_results
from benchmarks.scalar_sac.v2_evaluation import load_v2_result

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import SB3BenchmarkRunResult, load_result


def discover_results(path: str | Path) -> tuple[SB3BenchmarkRunResult, ...]:
    root = Path(path)
    paths = sorted(root.rglob("*-result.json")) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"No SB3 result files found under {root}")
    return tuple(load_result(item) for item in paths)


def _summary(runs: Sequence[SB3BenchmarkRunResult]) -> dict[str, Any]:
    episodes = tuple(episode for run in runs for episode in run.episodes)
    task = runs[0].task
    feasible_energy = [episode.energy_kwh for episode in episodes if episode.feasible]

    def seed_summary(run):
        travel_times = [episode.travel_time_s for episode in run.episodes]
        return {
            **asdict(run.summary),
            "time_compliance_rate": 1.0 - run.summary.time_violation_rate,
            "failure_mode_counts": dict(
                sorted(
                    Counter(
                        failure_mode(episode, task) for episode in run.episodes
                    ).items()
                )
            ),
            "travel_time_distribution_s": {
                "minimum": min(travel_times),
                "median": float(median(travel_times)),
                "mean": fmean(travel_times),
                "maximum": max(travel_times),
            },
        }

    return {
        "episode_count": len(episodes),
        "by_training_seed": {
            str(run.training_seed): seed_summary(run)
            for run in sorted(runs, key=lambda item: item.training_seed)
        },
        "mean_rsr": fmean(
            run.summary.requirement_satisfaction_rate for run in runs
        ),
        "minimum_rsr": min(
            run.summary.requirement_satisfaction_rate for run in runs
        ),
        "maximum_rsr": max(
            run.summary.requirement_satisfaction_rate for run in runs
        ),
        "mean_completion_rate": fmean(run.summary.completion_rate for run in runs),
        "mean_speed_compliance_rate": fmean(
            run.summary.speed_compliance_rate for run in runs
        ),
        "mean_time_compliance_rate": sum(
            episode.travel_time_s <= task.max_time_s for episode in episodes
        )
        / len(episodes),
        "travel_time_s": {
            "minimum": min(episode.travel_time_s for episode in episodes),
            "median": float(median(episode.travel_time_s for episode in episodes)),
            "mean": fmean(episode.travel_time_s for episode in episodes),
            "maximum": max(episode.travel_time_s for episode in episodes),
        },
        "feasible_episode_count": len(feasible_energy),
        "mean_feasible_energy_kwh": (
            fmean(feasible_energy) if feasible_energy else None
        ),
        "median_feasible_energy_kwh": (
            float(median(feasible_energy)) if feasible_energy else None
        ),
        "failure_mode_counts": dict(
            sorted(Counter(failure_mode(item, task) for item in episodes).items())
        ),
        "mean_training_wall_time_s": fmean(
            run.training_wall_time_s for run in runs
        ),
        "policy_updates_by_training_seed": {
            str(run.training_seed): run.policy_updates
            for run in sorted(runs, key=lambda item: item.training_seed)
        },
        "diagnostics_by_training_seed": {
            str(run.training_seed): run.diagnostics
            for run in sorted(runs, key=lambda item: item.training_seed)
        },
    }


def _acceptance(configuration, runs, fast_reference):
    criteria = configuration.acceptance
    per_seed = {}
    for run in sorted(runs, key=lambda item: item.training_seed):
        comparison = paired_energy(run.episodes, fast_reference.episodes)
        energy_ratio = None
        if comparison:
            energy_ratio = fmean(
                track["left_energy_kwh"] / track["right_energy_kwh"]
                for track in comparison["tracks"]
            )
        checks = {
            "rsr": run.summary.requirement_satisfaction_rate
            >= criteria.minimum_rsr_per_training_seed,
            "completion": run.summary.completion_rate
            >= criteria.minimum_completion_rate_per_training_seed,
            "speed_compliance": run.summary.speed_compliance_rate
            >= criteria.minimum_speed_compliance_rate_per_training_seed,
            "energy": energy_ratio is not None
            and energy_ratio <= criteria.maximum_mean_energy_ratio_to_fast_reference,
        }
        per_seed[str(run.training_seed)] = {
            "accepted": all(checks.values()),
            "checks": checks,
            "mean_energy_ratio_to_fast_reference": energy_ratio,
            "paired_track_count": comparison["paired_track_count"]
            if comparison
            else 0,
        }
    rsr = [run.summary.requirement_satisfaction_rate for run in runs]
    rsr_range = max(rsr) - min(rsr)
    stable = rsr_range <= criteria.maximum_rsr_range_across_training_seeds
    return {
        "accepted": stable and all(item["accepted"] for item in per_seed.values()),
        "seed_stability_accepted": stable,
        "rsr_range_across_training_seeds": rsr_range,
        "by_training_seed": per_seed,
    }


def _custom_sac_curves(path: str | Path | None):
    if path is None:
        return None
    paths = sorted(
        Path(path).glob(
            "v2b-balanced/training-seed-*/step-*/validation-result.json"
        )
    )
    rows = []
    for item in paths:
        run = load_v2_result(item)
        rows.append(
            {
                "algorithm": "custom-sac-v2b",
                "training_seed": run.training_seed,
                "training_steps": run.training_steps,
                "rsr": run.summary.requirement_satisfaction_rate,
                "completion_rate": run.summary.completion_rate,
                "speed_compliance_rate": run.summary.speed_compliance_rate,
            }
        )
    return rows


def analyze(
    runs: Sequence[SB3BenchmarkRunResult],
    configuration,
    references: ReferenceResults,
    *,
    custom_sac_results: str | Path | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    expected_hash = configuration_sha256(configuration)
    if {run.configuration_sha256 for run in runs} != {expected_hash}:
        raise ValueError("SB3 results do not match the analysis configuration")
    if any(
        episode.evaluation_seed
        in set(configuration.track_splits.paper_final_test_reserved)
        for run in runs
        for episode in run.episodes
    ):
        raise ValueError("Reserved final-test seeds occur in SB3 results")
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run.algorithm, run.evaluation_split_id, run.training_steps)].append(
            run
        )
    algorithms = ("sac", "ppo")
    split_ids = ("development-2000-2008-v1", "validation-3000-3008-v1")
    expected_keys = {
        (algorithm, split_id, step)
        for algorithm in algorithms
        for split_id in split_ids
        for step in configuration.learning_curve_steps
    }
    if require_complete:
        if set(grouped) != expected_keys:
            raise ValueError(
                "Incomplete SB3 result set; "
                f"missing={expected_keys - set(grouped)}, "
                f"extra={set(grouped) - expected_keys}"
            )
        expected_seeds = set(configuration.training_seeds)
        for key, values in grouped.items():
            seeds = {run.training_seed for run in values}
            if seeds != expected_seeds or len(values) != len(expected_seeds):
                raise ValueError(f"Incomplete training seeds for {key}: {seeds}")
    summaries = {
        f"{algorithm}/{split_id}/step-{step}": _summary(values)
        for (algorithm, split_id, step), values in sorted(grouped.items())
    }
    fast_reference = next(
        item
        for item in references.policies
        if item.policy_id == "fast-compliant-oracle"
    )
    if (
        references.configuration_sha256 != expected_hash
        or references.evaluation_set_id != "validation-3000-3008-v1"
        or references.task != asdict(configuration.task)
    ):
        raise ValueError("Reference results do not match the SB3 validation protocol")
    final_step = configuration.total_training_steps
    acceptance = {
        algorithm: _acceptance(
            configuration,
            grouped[(algorithm, "validation-3000-3008-v1", final_step)],
            fast_reference,
        )
        for algorithm in algorithms
        if (algorithm, "validation-3000-3008-v1", final_step) in grouped
    }
    paired = []
    for seed in configuration.training_seeds:
        sac = next(
            (
                run
                for run in grouped.get(
                    ("sac", "validation-3000-3008-v1", final_step), []
                )
                if run.training_seed == seed
            ),
            None,
        )
        ppo = next(
            (
                run
                for run in grouped.get(
                    ("ppo", "validation-3000-3008-v1", final_step), []
                )
                if run.training_seed == seed
            ),
            None,
        )
        if sac and ppo:
            comparison = paired_energy(sac.episodes, ppo.episodes)
            if comparison:
                paired.append({"training_seed": seed, **comparison})
    curves = [
        {
            "algorithm": run.algorithm,
            "training_seed": run.training_seed,
            "training_steps": run.training_steps,
            "rsr": run.summary.requirement_satisfaction_rate,
            "completion_rate": run.summary.completion_rate,
            "speed_compliance_rate": run.summary.speed_compliance_rate,
        }
        for run in sorted(
            runs,
            key=lambda item: (item.algorithm, item.training_seed, item.training_steps),
        )
        if run.evaluation_split_id == "validation-3000-3008-v1"
    ]
    custom_curves = _custom_sac_curves(custom_sac_results)
    if custom_curves:
        curves.extend(custom_curves)
    stability = []
    by_policy = defaultdict(list)
    for row in curves:
        by_policy[(row["algorithm"], row["training_seed"])].append(row)
    for (algorithm, seed), rows in sorted(by_policy.items()):
        rows.sort(key=lambda item: item["training_steps"])
        peak = max(rows, key=lambda item: item["rsr"])
        final = rows[-1]
        stability.append(
            {
                "algorithm": algorithm,
                "training_seed": seed,
                "peak_rsr": peak["rsr"],
                "peak_training_steps": peak["training_steps"],
                "final_rsr": final["rsr"],
                "peak_to_final_change": final["rsr"] - peak["rsr"],
            }
        )
    sac_ok = acceptance.get("sac", {}).get("accepted", False)
    ppo_ok = acceptance.get("ppo", {}).get("accepted", False)
    decision = (
        "A"
        if sac_ok and ppo_ok
        else "B"
        if sac_ok
        else "C"
        if ppo_ok
        else "D"
    )
    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": expected_hash,
        "result_file_count": len(runs),
        "reserved_final_test_evaluated": False,
        "acceptance_criteria": asdict(configuration.acceptance),
        "summaries": summaries,
        "acceptance": acceptance,
        "paired_sac_ppo_energy": paired,
        "learning_curves": curves,
        "learning_stability": stability,
        "decision_case": decision,
        "reference_summaries": {
            policy.policy_id: {
                "description": policy.description,
                "privileged_track_access": policy.privileged_track_access,
                "summary": asdict(policy.summary),
                "failure_mode_counts": dict(
                    Counter(
                        failure_mode(episode, configuration.task)
                        for episode in policy.episodes
                    )
                ),
            }
            for policy in references.policies
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--custom-sac-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    payload = analyze(
        discover_results(args.results),
        configuration,
        load_reference_results(args.references),
        custom_sac_results=args.custom_sac_results,
        require_complete=not args.allow_incomplete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
