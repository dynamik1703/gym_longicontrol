"""Analyze constrained SAC-Lagrangian using external physical metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from benchmarks.scalar_sac.analysis import failure_mode, paired_energy
from benchmarks.scalar_sac.references import load_reference_results

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import ConstrainedRunResult, load_result

DEFAULT_SCALAR_SB3 = Path("benchmarks/scalar_sb3/results.json")
DEFAULT_SCALAR_V2 = Path("benchmarks/scalar_sac/results-v2.json")
DEFAULT_CREDIT = Path("benchmarks/credit_assignment/results.json")
DEFAULT_REFERENCES = Path("benchmarks/scalar_sb3/reference-results.json")


def discover_results(path: str | Path) -> tuple[ConstrainedRunResult, ...]:
    root = Path(path)
    paths = sorted(root.rglob("*-result.json")) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"No constrained result files found under {root}")
    return tuple(load_result(item) for item in paths)


def _summary(runs: Sequence[ConstrainedRunResult]) -> dict[str, Any]:
    episodes = tuple(episode for run in runs for episode in run.episodes)
    task = runs[0].task
    feasible_energy = [episode.energy_kwh for episode in episodes if episode.feasible]

    def seed_summary(run):
        values = asdict(run.summary)
        return {
            **values,
            "time_compliance_rate": 1.0 - run.summary.time_violation_rate,
            "failure_mode_counts": dict(
                sorted(
                    Counter(
                        failure_mode(episode, task) for episode in run.episodes
                    ).items()
                )
            ),
            "simulator_steps": run.simulator_steps,
            "gradient_updates": run.gradient_updates,
            "diagnostics": run.diagnostics,
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
        "mean_time_compliance_rate": sum(
            episode.travel_time_s <= task.max_time_s for episode in episodes
        )
        / len(episodes),
        "mean_speed_compliance_rate": fmean(
            run.summary.speed_compliance_rate for run in runs
        ),
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
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return raw


def _diagnostic_summary(results_directory: Path, training_seed: int):
    path = (
        results_directory
        / f"training-seed-{training_seed}"
        / "training-diagnostics.json"
    )
    rows = _read_json_list(path)
    multiplier_keys = ("lagrange_speed", "lagrange_task")
    critic_keys = ("loss/q0", "loss/q1", "loss/q2", "loss/q_total")
    required_keys = (*multiplier_keys, *critic_keys, "loss/actor_total", "alpha")
    nonfinite = any(
        not np.isfinite(float(row[key])) for row in rows for key in required_keys
    )
    maximum_multiplier = max(
        float(row[key]) for row in rows for key in multiplier_keys
    )
    maximum_critic_loss = max(
        abs(float(row[key])) for row in rows for key in critic_keys
    )
    return {
        "episode_count": len(rows),
        "final": {key: rows[-1][key] for key in required_keys},
        "maximum_lagrange_speed": max(row["lagrange_speed"] for row in rows),
        "maximum_lagrange_task": max(row["lagrange_task"] for row in rows),
        "maximum_multiplier": maximum_multiplier,
        "maximum_absolute_critic_loss": maximum_critic_loss,
        "has_nonfinite_required_diagnostic": bool(nonfinite),
    }


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Expected non-empty JSON list: {path}")
    return raw


def analyze(
    runs: Sequence[ConstrainedRunResult],
    configuration,
    *,
    results_directory: str | Path,
    scalar_sb3_path: str | Path = DEFAULT_SCALAR_SB3,
    scalar_v2_path: str | Path = DEFAULT_SCALAR_V2,
    credit_path: str | Path = DEFAULT_CREDIT,
    references_path: str | Path = DEFAULT_REFERENCES,
    require_complete: bool = True,
) -> dict[str, Any]:
    expected_hash = configuration_sha256(configuration)
    if {run.configuration_sha256 for run in runs} != {expected_hash}:
        raise ValueError("Constrained results do not match the frozen configuration")
    reserved = set(configuration.track_splits.paper_final_test_reserved)
    if any(
        episode.evaluation_seed in reserved
        for run in runs
        for episode in run.episodes
    ):
        raise ValueError("Reserved final-test seeds occur in constrained results")
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run.evaluation_split_id, run.simulator_step_target)].append(run)
    split_ids = ("development-2000-2008-v1", "validation-3000-3008-v1")
    expected_keys = {
        (split_id, step)
        for split_id in split_ids
        for step in configuration.simulator_step_checkpoints
    }
    if require_complete:
        if set(grouped) != expected_keys:
            raise ValueError(
                "Incomplete constrained results; "
                f"missing={expected_keys - set(grouped)}, "
                f"extra={set(grouped) - expected_keys}"
            )
        expected_seeds = set(configuration.training_seeds)
        for key, values in grouped.items():
            seeds = {run.training_seed for run in values}
            if seeds != expected_seeds or len(values) != len(expected_seeds):
                raise ValueError(f"Incomplete training seeds for {key}: {seeds}")
    summaries = {
        f"{split_id}/simulator-target-{step}": _summary(values)
        for (split_id, step), values in sorted(grouped.items())
    }
    validation_runs = [
        run
        for run in runs
        if run.evaluation_split_id == "validation-3000-3008-v1"
    ]
    learning_curves = [
        {
            "training_seed": run.training_seed,
            "simulator_step_target": run.simulator_step_target,
            "simulator_steps": run.simulator_steps,
            "rsr": run.summary.requirement_satisfaction_rate,
            "completion_rate": run.summary.completion_rate,
            "time_compliance_rate": 1.0 - run.summary.time_violation_rate,
            "speed_compliance_rate": run.summary.speed_compliance_rate,
        }
        for run in sorted(
            validation_runs,
            key=lambda item: (item.training_seed, item.simulator_step_target),
        )
    ]
    final_target = configuration.simulator_step_budget
    final_runs = sorted(
        grouped[("validation-3000-3008-v1", final_target)],
        key=lambda item: item.training_seed,
    )
    stability = []
    for seed in configuration.training_seeds:
        rows = [row for row in learning_curves if row["training_seed"] == seed]
        peak = max(rows, key=lambda item: item["rsr"])
        final = rows[-1]
        stability.append(
            {
                "training_seed": seed,
                "peak_rsr": peak["rsr"],
                "peak_simulator_step_target": peak["simulator_step_target"],
                "final_rsr": final["rsr"],
                "peak_to_final_rsr_drop": peak["rsr"] - final["rsr"],
            }
        )

    references = load_reference_results(references_path)
    fast_reference = next(
        item
        for item in references.policies
        if item.policy_id == "fast-compliant-oracle"
    )
    conservative_reference = next(
        item
        for item in references.policies
        if item.policy_id == "conservative-oracle"
    )
    energy_by_seed = {}
    for run in final_runs:
        comparison = paired_energy(run.episodes, fast_reference.episodes)
        ratios = (
            [
                row["left_energy_kwh"] / row["right_energy_kwh"]
                for row in comparison["tracks"]
            ]
            if comparison
            else []
        )
        energy_by_seed[str(run.training_seed)] = {
            "paired_feasible_track_count": len(ratios),
            "mean_energy_ratio_to_fast_reference": fmean(ratios) if ratios else None,
            "comparison": comparison,
        }

    criteria = configuration.acceptance
    per_seed = {}
    for run in final_runs:
        ratio = energy_by_seed[str(run.training_seed)][
            "mean_energy_ratio_to_fast_reference"
        ]
        checks = {
            "rsr": run.summary.requirement_satisfaction_rate
            >= criteria.minimum_rsr_per_training_seed,
            "completion": run.summary.completion_rate
            >= criteria.minimum_completion_rate_per_training_seed,
            "speed_compliance": run.summary.speed_compliance_rate
            >= criteria.minimum_speed_compliance_rate_per_training_seed,
            "energy": ratio is not None
            and ratio <= criteria.maximum_mean_energy_ratio_to_fast_reference,
        }
        per_seed[str(run.training_seed)] = {
            "accepted": all(checks.values()),
            "checks": checks,
            "mean_energy_ratio_to_fast_reference": ratio,
        }
    final_rsr = [run.summary.requirement_satisfaction_rate for run in final_runs]
    rsr_range = max(final_rsr) - min(final_rsr)
    stable_range = rsr_range <= criteria.maximum_rsr_range_across_training_seeds
    stable_late = all(
        row["peak_to_final_rsr_drop"]
        <= criteria.maximum_peak_to_final_rsr_drop
        for row in stability
    )
    robust = (
        stable_range
        and stable_late
        and all(item["accepted"] for item in per_seed.values())
    )

    scalar_sb3 = _read_json(scalar_sb3_path)
    scalar_final = scalar_sb3["summaries"][
        "sac/validation-3000-3008-v1/step-300000"
    ]
    scalar_seed_rsr = {
        seed: scalar_final["by_training_seed"][str(seed)][
            "requirement_satisfaction_rate"
        ]
        for seed in configuration.training_seeds
    }
    constrained_seed_rsr = {
        run.training_seed: run.summary.requirement_satisfaction_rate
        for run in final_runs
    }
    mean_final_rsr = fmean(constrained_seed_rsr.values())
    improved_seeds = [
        seed
        for seed in configuration.training_seeds
        if constrained_seed_rsr[seed] > scalar_seed_rsr[seed]
    ]
    material_gain = mean_final_rsr - criteria.frozen_scalar_sac_mean_rsr
    material = (
        material_gain >= criteria.minimum_material_mean_rsr_gain
        and len(improved_seeds) >= criteria.minimum_improved_training_seeds
    )

    final_summary = summaries[
        f"validation-3000-3008-v1/simulator-target-{final_target}"
    ]
    standstill = (
        final_summary["mean_completion_rate"]
        <= criteria.standstill_maximum_completion_rate
        and final_summary["mean_speed_compliance_rate"]
        >= criteria.standstill_minimum_speed_compliance_rate
    )
    diagnostic_summaries = {
        str(seed): _diagnostic_summary(Path(results_directory), seed)
        for seed in configuration.training_seeds
    }
    unstable_seeds = [
        seed
        for seed, summary in diagnostic_summaries.items()
        if summary["has_nonfinite_required_diagnostic"]
        or summary["maximum_multiplier"]
        > criteria.instability_multiplier_threshold
        or summary["maximum_absolute_critic_loss"]
        > criteria.instability_critic_loss_threshold
    ]
    systematically_unstable = (
        len(unstable_seeds) >= criteria.minimum_unstable_training_seeds
    )
    decision = (
        "E"
        if systematically_unstable
        else "A"
        if robust
        else "D"
        if standstill
        else "B"
        if material
        else "C"
    )
    scalar_v2 = _read_json(scalar_v2_path)
    credit = _read_json(credit_path)
    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": expected_hash,
        "result_file_count": len(runs),
        "reserved_final_test_evaluated": False,
        "summaries": summaries,
        "learning_curves": learning_curves,
        "learning_stability": stability,
        "training_diagnostics": diagnostic_summaries,
        "acceptance": {
            "accepted": robust,
            "by_training_seed": per_seed,
            "rsr_range_across_training_seeds": rsr_range,
            "seed_stability_accepted": stable_range,
            "late_stability_accepted": stable_late,
        },
        "material_improvement": {
            "accepted": material,
            "constrained_mean_rsr": mean_final_rsr,
            "frozen_scalar_sac_mean_rsr": criteria.frozen_scalar_sac_mean_rsr,
            "mean_rsr_gain": material_gain,
            "minimum_required_gain": criteria.minimum_material_mean_rsr_gain,
            "constrained_rsr_by_seed": constrained_seed_rsr,
            "frozen_scalar_sac_rsr_by_seed": scalar_seed_rsr,
            "improved_training_seeds": improved_seeds,
        },
        "standstill_flag": standstill,
        "optimizer_instability": {
            "systematic": systematically_unstable,
            "unstable_training_seeds": unstable_seeds,
        },
        "decision_case": decision,
        "feasible_energy": {
            "by_training_seed": energy_by_seed,
            "fast_reference_summary": asdict(fast_reference.summary),
            "conservative_reference_summary": asdict(
                conservative_reference.summary
            ),
        },
        "frozen_comparisons": {
            "sb3_sac_v2b_300k": scalar_final,
            "sb3_sac_v2b_learning_curves": [
                row
                for row in scalar_sb3["learning_curves"]
                if row["algorithm"] == "sac"
            ],
            "custom_scalar_v2b_validation_mean_rsr_300k": scalar_v2[
                "iterations"
            ]["v2b"]["validation_mean_rsr_by_step"]["v2b-balanced"]["300000"],
            "custom_scalar_v2b_historical_exploratory_300k": scalar_v2[
                "iterations"
            ]["v2b"]["exploratory_300k"]["v2b-balanced"],
            "credit_condition_c_300k": credit["final_summaries"][
                "c-horizon-only/validation-3000-3008-v1/"
                "simulator-target-300000"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scalar-sb3", type=Path, default=DEFAULT_SCALAR_SB3)
    parser.add_argument("--scalar-v2", type=Path, default=DEFAULT_SCALAR_V2)
    parser.add_argument("--credit", type=Path, default=DEFAULT_CREDIT)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    payload = analyze(
        discover_results(args.results),
        configuration,
        results_directory=args.results,
        scalar_sb3_path=args.scalar_sb3,
        scalar_v2_path=args.scalar_v2,
        credit_path=args.credit,
        references_path=args.references,
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
