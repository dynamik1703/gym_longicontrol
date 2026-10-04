"""Analyze constrained V2 with frozen external physical requirements."""

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

from benchmarks.constrained_rl.results import load_result as load_v1_result
from benchmarks.scalar_sac.analysis import paired_energy
from benchmarks.scalar_sac.evaluation import EpisodeEvaluation
from benchmarks.scalar_sac.references import load_reference_results
from gym_longicontrol.domain.task import TaskSpecification

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import ConstrainedV2RunResult, load_result

DEFAULT_V1_ANALYSIS = Path("benchmarks/constrained_rl/results.json")
DEFAULT_V1_RUNS = Path("runs/constrained-rl-20260927")
DEFAULT_SCALAR_SB3 = Path("benchmarks/scalar_sb3/results.json")
DEFAULT_REFERENCES = Path("benchmarks/scalar_sb3/reference-results.json")


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_results(path: str | Path) -> tuple[ConstrainedV2RunResult, ...]:
    root = Path(path)
    paths = sorted(root.rglob("*-result.json")) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"No constrained V2 result files found under {root}")
    return tuple(load_result(item) for item in paths)


def behavior_category(
    episode: EpisodeEvaluation, task: TaskSpecification
) -> str:
    """Return the preregistered exclusive physical behavior category."""

    if episode.feasible:
        return "fully_feasible"
    speed_compliant = (
        episode.max_speed_violation_m_s <= task.max_speed_violation_m_s
    )
    if episode.completed and not speed_compliant:
        return "completed_with_speed_violation"
    if episode.completed:
        return "completed_too_slowly"
    if episode.final_position_m <= 1.0:
        return "standstill"
    return "partial_progress_or_crawling"


def _summary(runs: Sequence[ConstrainedV2RunResult]) -> dict[str, Any]:
    episodes = tuple(episode for run in runs for episode in run.episodes)
    task = runs[0].task
    feasible_energy = [episode.energy_kwh for episode in episodes if episode.feasible]

    def seed_summary(run):
        values = asdict(run.summary)
        return {
            **values,
            "time_compliance_rate": 1.0 - run.summary.time_violation_rate,
            "behavior_counts": dict(
                sorted(
                    Counter(
                        behavior_category(episode, task) for episode in run.episodes
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
        "behavior_counts": dict(
            sorted(Counter(behavior_category(item, task) for item in episodes).items())
        ),
    }


def _read_diagnostics(results_directory: Path, training_seed: int):
    path = (
        results_directory
        / f"training-seed-{training_seed}"
        / "training-diagnostics.json"
    )
    rows = _read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty diagnostic list: {path}")
    return rows


def _diagnostic_summary(results_directory: Path, training_seed: int):
    rows = _read_diagnostics(results_directory, training_seed)
    multiplier_keys = ("lagrange_speed", "lagrange_deadline")
    critic_keys = ("loss/q0", "loss/q1", "loss/q2", "loss/q_total")
    required = (*multiplier_keys, *critic_keys, "loss/actor_total", "alpha")
    nonfinite = any(
        not np.isfinite(float(row[key])) for row in rows for key in required
    )
    return {
        "episode_count": len(rows),
        "first": {key: rows[0][key] for key in required},
        "final": {key: rows[-1][key] for key in required},
        "maximum_lagrange_speed": max(row["lagrange_speed"] for row in rows),
        "maximum_lagrange_deadline": max(
            row["lagrange_deadline"] for row in rows
        ),
        "maximum_multiplier": max(
            float(row[key]) for row in rows for key in multiplier_keys
        ),
        "maximum_absolute_critic_loss": max(
            abs(float(row[key])) for row in rows for key in critic_keys
        ),
        "has_nonfinite_required_diagnostic": bool(nonfinite),
        "first_positive_deadline_cost_simulator_step": next(
            (
                row["simulator_steps"]
                for row in rows
                if row["deadline_deficit_integral_s"] > 0
            ),
            None,
        ),
        "first_positive_deadline_multiplier_simulator_step": next(
            (row["simulator_steps"] for row in rows if row["lagrange_deadline"] > 0),
            None,
        ),
        "maximum_deadline_deficit_integral_s": max(
            row["deadline_deficit_integral_s"] for row in rows
        ),
        "final_training_position_m": rows[-1]["final_position_m"],
    }


def _frozen_v1_behavior_counts(path: Path, task: TaskSpecification) -> dict[str, int]:
    result_paths = sorted(
        path.glob("training-seed-*/target-300000/validation-3000-3008-result.json")
    )
    if len(result_paths) != 3:
        raise ValueError("Frozen V1 raw validation results are incomplete")
    runs = [load_v1_result(item) for item in result_paths]
    episodes = [episode for run in runs for episode in run.episodes]
    return dict(
        sorted(Counter(behavior_category(item, task) for item in episodes).items())
    )


def analyze(
    runs: Sequence[ConstrainedV2RunResult],
    configuration,
    *,
    results_directory: str | Path,
    v1_analysis_path: str | Path = DEFAULT_V1_ANALYSIS,
    v1_runs_path: str | Path = DEFAULT_V1_RUNS,
    scalar_sb3_path: str | Path = DEFAULT_SCALAR_SB3,
    references_path: str | Path = DEFAULT_REFERENCES,
    require_complete: bool = True,
) -> dict[str, Any]:
    expected_hash = configuration_sha256(configuration)
    if {run.configuration_sha256 for run in runs} != {expected_hash}:
        raise ValueError("V2 results do not match the frozen configuration")
    reserved = set(configuration.track_splits.paper_final_test_reserved)
    if any(
        episode.evaluation_seed in reserved
        for run in runs
        for episode in run.episodes
    ):
        raise ValueError("Reserved final-test seeds occur in V2 results")
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run.evaluation_split_id, run.simulator_step_target)].append(run)
    split_ids = ("development-2000-2008-v2", "validation-3000-3008-v2")
    expected_keys = {
        (split_id, step)
        for split_id in split_ids
        for step in configuration.simulator_step_checkpoints
    }
    if require_complete:
        if set(grouped) != expected_keys:
            raise ValueError(
                f"Incomplete V2 results: missing={expected_keys - set(grouped)}, "
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
        if run.evaluation_split_id == "validation-3000-3008-v2"
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
            "standstill_rate": sum(
                behavior_category(episode, run.task) == "standstill"
                for episode in run.episodes
            )
            / len(run.episodes),
        }
        for run in sorted(
            validation_runs,
            key=lambda item: (item.training_seed, item.simulator_step_target),
        )
    ]
    final_target = configuration.simulator_step_budget
    final_runs = sorted(
        grouped[("validation-3000-3008-v2", final_target)],
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

    final_summary_key = (
        f"validation-3000-3008-v2/simulator-target-{final_target}"
    )
    final_summary = summaries[final_summary_key]
    final_episode_count = final_summary["episode_count"]
    standstill_count = final_summary["behavior_counts"].get("standstill", 0)
    standstill_rate = standstill_count / final_episode_count
    seed_rsr = {
        run.training_seed: run.summary.requirement_satisfaction_rate
        for run in final_runs
    }
    seed_completion = {
        run.training_seed: run.summary.completion_rate for run in final_runs
    }
    improved_seeds = [
        seed
        for seed in configuration.training_seeds
        if seed_rsr[seed] > 0 and seed_completion[seed] > 0
    ]
    criteria = configuration.acceptance
    material = (
        final_summary["mean_rsr"] >= criteria.minimum_material_pooled_rsr
        and len(improved_seeds) >= criteria.minimum_improved_training_seeds
        and standstill_rate <= criteria.maximum_material_standstill_rate
    )
    late_stability = all(
        row["peak_to_final_rsr_drop"] <= criteria.maximum_peak_to_final_rsr_drop
        for row in stability
    )
    speed_credible = (
        final_summary["mean_speed_compliance_rate"]
        >= criteria.minimum_credible_speed_compliance_rate
    )
    credible = material and speed_credible and late_stability

    references = load_reference_results(references_path)
    fast_reference = next(
        item
        for item in references.policies
        if item.policy_id == "fast-compliant-oracle"
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
    strong_by_seed = {}
    for run in final_runs:
        ratio = energy_by_seed[str(run.training_seed)][
            "mean_energy_ratio_to_fast_reference"
        ]
        checks = {
            "rsr": run.summary.requirement_satisfaction_rate
            >= criteria.strong_minimum_rsr_per_training_seed,
            "completion": run.summary.completion_rate
            >= criteria.strong_minimum_completion_rate_per_training_seed,
            "speed": run.summary.speed_compliance_rate
            >= criteria.strong_minimum_speed_compliance_rate_per_training_seed,
            "energy": ratio is not None
            and ratio <= criteria.strong_maximum_mean_energy_ratio_to_fast_reference,
        }
        strong_by_seed[str(run.training_seed)] = {
            "accepted": all(checks.values()),
            "checks": checks,
            "mean_energy_ratio_to_fast_reference": ratio,
        }
    rsr_range = max(seed_rsr.values()) - min(seed_rsr.values())
    strong = (
        all(item["accepted"] for item in strong_by_seed.values())
        and rsr_range <= criteria.strong_maximum_rsr_range
        and late_stability
    )

    diagnostics = {
        str(seed): _diagnostic_summary(Path(results_directory), seed)
        for seed in configuration.training_seeds
    }
    unstable_seeds = [
        int(seed)
        for seed, summary in diagnostics.items()
        if summary["has_nonfinite_required_diagnostic"]
        or summary["maximum_multiplier"]
        > criteria.instability_multiplier_threshold
        or summary["maximum_absolute_critic_loss"]
        > criteria.instability_critic_loss_threshold
    ]
    unstable = len(unstable_seeds) >= criteria.minimum_unstable_training_seeds
    completion_substantial = (
        final_summary["mean_completion_rate"]
        >= criteria.minimum_material_pooled_completion_rate
        and sum(value > 0 for value in seed_completion.values())
        >= criteria.minimum_improved_training_seeds
        and standstill_rate <= criteria.maximum_material_standstill_rate
    )
    decision = (
        "E"
        if unstable
        else "A"
        if credible
        else "B"
        if completion_substantial and not speed_credible
        else "D"
        if standstill_rate > criteria.standstill_dominance_rate
        else "C"
    )

    v1 = _read_json(v1_analysis_path)
    v1_final_key = "validation-3000-3008-v1/simulator-target-300000"
    v1_final = v1["summaries"][v1_final_key]
    v1_behavior = _frozen_v1_behavior_counts(
        Path(v1_runs_path), configuration.task
    )
    scalar = _read_json(scalar_sb3_path)
    scalar_final = scalar["summaries"]["sac/validation-3000-3008-v1/step-300000"]
    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": expected_hash,
        "result_file_count": len(runs),
        "reserved_final_test_evaluated": False,
        "summaries": summaries,
        "learning_curves": learning_curves,
        "learning_stability": stability,
        "training_diagnostics": diagnostics,
        "acceptance": {
            "material_improvement_over_v1": material,
            "credible_constrained_baseline": credible,
            "strong_baseline": strong,
            "improved_training_seeds": improved_seeds,
            "standstill_rate": standstill_rate,
            "speed_credible": speed_credible,
            "late_stability": late_stability,
            "rsr_range": rsr_range,
            "strong_by_training_seed": strong_by_seed,
        },
        "optimizer_instability": {
            "systematic": unstable,
            "unstable_training_seeds": unstable_seeds,
        },
        "decision_case": decision,
        "feasible_energy": {
            "ranking_permitted": material,
            "by_training_seed": energy_by_seed,
            "fast_reference_summary": asdict(fast_reference.summary),
        },
        "frozen_comparisons": {
            "constrained_v1_300k": v1_final,
            "constrained_v1_learning_curves": v1["learning_curves"],
            "constrained_v1_behavior_counts": v1_behavior,
            "scalar_sb3_sac_v2b_300k": scalar_final,
            "scalar_sb3_sac_v2b_learning_curves": [
                row for row in scalar["learning_curves"] if row["algorithm"] == "sac"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--v1-analysis", type=Path, default=DEFAULT_V1_ANALYSIS)
    parser.add_argument("--v1-runs", type=Path, default=DEFAULT_V1_RUNS)
    parser.add_argument("--scalar-sb3", type=Path, default=DEFAULT_SCALAR_SB3)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    payload = analyze(
        discover_results(args.results),
        configuration,
        results_directory=args.results,
        v1_analysis_path=args.v1_analysis,
        v1_runs_path=args.v1_runs,
        scalar_sb3_path=args.scalar_sb3,
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
