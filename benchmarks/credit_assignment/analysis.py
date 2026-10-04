"""Analyze the preregistered credit-assignment experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, median
from typing import Any

from benchmarks.scalar_sac.analysis import failure_mode, paired_energy

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import CreditAssignmentRunResult, load_result


def discover_results(path: str | Path) -> tuple[CreditAssignmentRunResult, ...]:
    root = Path(path)
    paths = sorted(root.rglob("*-result.json")) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"No credit-assignment results found under {root}")
    return tuple(load_result(item) for item in paths)


def _run_summary(run: CreditAssignmentRunResult) -> dict[str, Any]:
    task = run.task
    return {
        **asdict(run.summary),
        "time_compliance_rate": sum(
            item.travel_time_s <= task.max_time_s for item in run.episodes
        )
        / len(run.episodes),
        "failure_mode_counts": dict(
            sorted(Counter(failure_mode(item, task) for item in run.episodes).items())
        ),
        "simulator_steps": run.simulator_steps,
        "agent_decisions": run.agent_decisions,
        "gradient_updates": run.gradient_updates,
        "diagnostics": run.diagnostics,
    }


def _group_summary(runs: Sequence[CreditAssignmentRunResult]) -> dict[str, Any]:
    episodes = tuple(item for run in runs for item in run.episodes)
    task = runs[0].task
    feasible_energy = [item.energy_kwh for item in episodes if item.feasible]
    seed_values = {
        str(run.training_seed): _run_summary(run)
        for run in sorted(runs, key=lambda item: item.training_seed)
    }
    rsr = [run.summary.requirement_satisfaction_rate for run in runs]
    return {
        "episode_count": len(episodes),
        "by_training_seed": seed_values,
        "mean_rsr": fmean(rsr),
        "minimum_rsr": min(rsr),
        "maximum_rsr": max(rsr),
        "rsr_range": max(rsr) - min(rsr),
        "mean_completion_rate": fmean(
            run.summary.completion_rate for run in runs
        ),
        "mean_time_compliance_rate": sum(
            item.travel_time_s <= task.max_time_s for item in episodes
        )
        / len(episodes),
        "mean_speed_compliance_rate": fmean(
            run.summary.speed_compliance_rate for run in runs
        ),
        "failure_mode_counts": dict(
            sorted(Counter(failure_mode(item, task) for item in episodes).items())
        ),
        "feasible_episode_count": len(feasible_energy),
        "mean_feasible_energy_kwh": (
            fmean(feasible_energy) if feasible_energy else None
        ),
        "median_feasible_energy_kwh": (
            float(median(feasible_energy)) if feasible_energy else None
        ),
        "mean_training_wall_time_s": fmean(
            run.training_wall_time_s for run in runs
        ),
    }


def _effective_discount(configuration) -> list[dict[str, Any]]:
    delays = (10.0, 30.0, 60.0, 140.0)
    rows = []
    for condition in configuration.conditions:
        decision_interval = configuration.simulator_dt_s * condition.action_repeat
        rows.append(
            {
                "condition_id": condition.condition_id,
                "gamma_per_decision": condition.gamma,
                "action_repeat": condition.action_repeat,
                "decision_interval_s": decision_interval,
                "decision_frequency_hz": 1.0 / decision_interval,
                "continuous_time_constant_s": (
                    -decision_interval / math.log(condition.gamma)
                ),
                "weights": {
                    str(int(delay)): condition.gamma ** (delay / decision_interval)
                    for delay in delays
                },
            }
        )
    return rows


def _robust(configuration, runs: Sequence[CreditAssignmentRunResult]) -> bool:
    criteria = configuration.acceptance
    rsr = [run.summary.requirement_satisfaction_rate for run in runs]
    return bool(
        all(
            run.summary.requirement_satisfaction_rate
            >= criteria.minimum_rsr_per_training_seed
            and run.summary.completion_rate
            >= criteria.minimum_completion_rate_per_training_seed
            and run.summary.speed_compliance_rate
            >= criteria.minimum_speed_compliance_rate_per_training_seed
            for run in runs
        )
        and max(rsr) - min(rsr)
        <= criteria.maximum_rsr_range_across_training_seeds
    )


def _decision_gate(configuration, final_by_condition) -> dict[str, Any]:
    baseline = {
        run.training_seed: run.summary.requirement_satisfaction_rate
        for run in final_by_condition["a-baseline"]
    }
    scores = {}
    for index, condition in enumerate(configuration.conditions):
        runs = final_by_condition[condition.condition_id]
        by_seed = {
            run.training_seed: run.summary.requirement_satisfaction_rate
            for run in runs
        }
        deltas = {
            str(seed): by_seed[seed] - baseline[seed]
            for seed in configuration.training_seeds
        }
        values = list(by_seed.values())
        improved_seed_count = sum(value > 0 for value in deltas.values())
        mean_gain = fmean(values) - fmean(baseline.values())
        scores[condition.condition_id] = {
            "mean_rsr": fmean(values),
            "minimum_rsr": min(values),
            "maximum_rsr": max(values),
            "rsr_range": max(values) - min(values),
            "rsr_delta_vs_a_by_training_seed": deltas,
            "mean_rsr_gain_vs_a": mean_gain,
            "improved_training_seed_count": improved_seed_count,
            "material_improvement": (
                condition.condition_id != "a-baseline"
                and mean_gain >= configuration.material_mean_rsr_improvement
                and improved_seed_count >= 2
            ),
            "robust": _robust(configuration, runs),
            "simplicity_rank": index,
        }
    interventions = [
        condition.condition_id for condition in configuration.conditions[1:]
    ]
    best = max(
        interventions,
        key=lambda condition_id: (
            scores[condition_id]["robust"],
            scores[condition_id]["material_improvement"],
            scores[condition_id]["minimum_rsr"],
            scores[condition_id]["mean_rsr"],
            -scores[condition_id]["simplicity_rank"],
        ),
    )
    mapping = {
        "b-discount-only": (
            "A",
            "Higher gamma solves most of the diagnosed problem.",
        ),
        "c-horizon-only": (
            "B",
            "Action repeat solves most of the diagnosed problem.",
        ),
        "d-horizon-discount": (
            "C",
            "The combined horizon and discount intervention is required.",
        ),
    }
    if scores[best]["robust"]:
        decision_case, conclusion = mapping[best]
    elif not any(scores[item]["material_improvement"] for item in interventions):
        decision_case = "E"
        conclusion = "No intervention materially improves robustness."
    elif (
        scores[best]["rsr_range"]
        > configuration.acceptance.maximum_rsr_range_across_training_seeds
    ):
        decision_case = "F"
        conclusion = "Improvement occurs, but seed instability remains dominant."
    else:
        decision_case, conclusion = mapping[best]
    return {
        "case": decision_case,
        "conclusion": conclusion,
        "selected_condition_id": best,
        "scores": scores,
    }


def _paired_track_outcomes(configuration, final_by_condition):
    baseline = {
        run.training_seed: run for run in final_by_condition["a-baseline"]
    }
    comparisons = {}
    for condition in configuration.conditions[1:]:
        rows = []
        for run in final_by_condition[condition.condition_id]:
            reference = baseline[run.training_seed]
            baseline_episodes = {
                item.evaluation_seed: item for item in reference.episodes
            }
            for episode in run.episodes:
                original = baseline_episodes[episode.evaluation_seed]
                rows.append(
                    {
                        "training_seed": run.training_seed,
                        "evaluation_seed": episode.evaluation_seed,
                        "baseline_feasible": original.feasible,
                        "condition_feasible": episode.feasible,
                        "feasibility_change": (
                            "improved"
                            if episode.feasible and not original.feasible
                            else "regressed"
                            if original.feasible and not episode.feasible
                            else "unchanged"
                        ),
                        "baseline_failure_mode": failure_mode(
                            original, configuration.task
                        ),
                        "condition_failure_mode": failure_mode(
                            episode, configuration.task
                        ),
                        "paired_feasible_energy_delta_kwh": (
                            episode.energy_kwh - original.energy_kwh
                            if episode.feasible and original.feasible
                            else None
                        ),
                    }
                )
        comparisons[condition.condition_id] = {
            "counts": dict(Counter(row["feasibility_change"] for row in rows)),
            "tracks": rows,
        }
    return comparisons


def analyze(
    runs: Sequence[CreditAssignmentRunResult],
    configuration,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    expected_hash = configuration_sha256(configuration)
    if {run.configuration_sha256 for run in runs} != {expected_hash}:
        raise ValueError("Results do not match the analysis configuration")
    reserved = set(configuration.track_splits.paper_final_test_reserved)
    if any(
        episode.evaluation_seed in reserved
        for run in runs
        for episode in run.episodes
    ):
        raise ValueError("Reserved final-test seeds occur in results")

    grouped = defaultdict(list)
    for run in runs:
        grouped[
            (
                run.condition_id,
                run.evaluation_split_id,
                run.simulator_step_target,
            )
        ].append(run)
    split_ids = ("development-2000-2008-v1", "validation-3000-3008-v1")
    expected_keys = {
        (condition.condition_id, split_id, step)
        for condition in configuration.conditions
        for split_id in split_ids
        for step in configuration.simulator_step_checkpoints
    }
    if require_complete:
        if set(grouped) != expected_keys:
            raise ValueError(
                "Incomplete result set; "
                f"missing={expected_keys - set(grouped)}, "
                f"extra={set(grouped) - expected_keys}"
            )
        expected_seeds = set(configuration.training_seeds)
        for key, values in grouped.items():
            seeds = {run.training_seed for run in values}
            if seeds != expected_seeds or len(values) != len(expected_seeds):
                raise ValueError(f"Incomplete training seeds for {key}: {seeds}")

    summaries = {
        f"{condition}/{split_id}/simulator-target-{step}": _group_summary(values)
        for (condition, split_id, step), values in sorted(grouped.items())
    }
    curves = [
        {
            "condition_id": run.condition_id,
            "condition_label": run.condition_label,
            "training_seed": run.training_seed,
            "simulator_step_target": run.simulator_step_target,
            "simulator_steps": run.simulator_steps,
            "agent_decisions": run.agent_decisions,
            "gradient_updates": run.gradient_updates,
            "rsr": run.summary.requirement_satisfaction_rate,
            "completion_rate": run.summary.completion_rate,
            "time_compliance_rate": sum(
                item.travel_time_s <= run.task.max_time_s for item in run.episodes
            )
            / len(run.episodes),
            "speed_compliance_rate": run.summary.speed_compliance_rate,
        }
        for run in sorted(
            runs,
            key=lambda item: (
                item.condition_id,
                item.training_seed,
                item.simulator_step_target,
            ),
        )
        if run.evaluation_split_id == "validation-3000-3008-v1"
    ]
    final_target = configuration.simulator_step_budget
    final_by_condition = {
        condition.condition_id: grouped[
            (
                condition.condition_id,
                "validation-3000-3008-v1",
                final_target,
            )
        ]
        for condition in configuration.conditions
        if (
            condition.condition_id,
            "validation-3000-3008-v1",
            final_target,
        )
        in grouped
    }
    decision = (
        _decision_gate(configuration, final_by_condition)
        if len(final_by_condition) == len(configuration.conditions)
        else None
    )
    stability = []
    for condition in configuration.conditions:
        for seed in configuration.training_seeds:
            rows = [
                item
                for item in curves
                if item["condition_id"] == condition.condition_id
                and item["training_seed"] == seed
            ]
            if not rows:
                continue
            peak = max(rows, key=lambda item: item["rsr"])
            final = rows[-1]
            stability.append(
                {
                    "condition_id": condition.condition_id,
                    "training_seed": seed,
                    "peak_rsr": peak["rsr"],
                    "peak_simulator_step_target": peak["simulator_step_target"],
                    "final_rsr": final["rsr"],
                    "peak_to_final_change": final["rsr"] - peak["rsr"],
                }
            )
    energy = {}
    for condition_id, condition_runs in final_by_condition.items():
        per_seed = {}
        for run in condition_runs:
            feasible = [item for item in run.episodes if item.feasible]
            per_seed[str(run.training_seed)] = {
                "eligible": (
                    run.summary.requirement_satisfaction_rate
                    >= configuration.acceptance.minimum_rsr_per_training_seed
                ),
                "feasible_track_count": len(feasible),
                "mean_feasible_energy_kwh": (
                    fmean(item.energy_kwh for item in feasible)
                    if feasible
                    else None
                ),
                "median_feasible_energy_kwh": (
                    float(median(item.energy_kwh for item in feasible))
                    if feasible
                    else None
                ),
                "per_track_kwh": {
                    str(item.evaluation_seed): item.energy_kwh for item in feasible
                },
            }
        energy[condition_id] = per_seed
    pairwise_energy = {}
    condition_ids = [
        item.condition_id
        for item in configuration.conditions
        if item.condition_id in final_by_condition
    ]
    for left_index, left_id in enumerate(condition_ids):
        for right_id in condition_ids[left_index + 1 :]:
            by_right = {
                run.training_seed: run for run in final_by_condition[right_id]
            }
            pairwise_energy[f"{left_id}__vs__{right_id}"] = {
                str(run.training_seed): paired_energy(
                    run.episodes, by_right[run.training_seed].episodes
                )
                for run in final_by_condition[left_id]
            }
    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": expected_hash,
        "result_file_count": len(runs),
        "reserved_final_test_evaluated": False,
        "interaction_accounting": {
            "primary_budget_unit": "underlying simulator transitions",
            "simulator_step_budget_per_condition_and_seed": final_target,
            "agent_decisions_and_gradient_updates_recorded_separately": True,
        },
        "effective_discount": _effective_discount(configuration),
        "physical_time_matched_repeat_5_gamma_from_0_99": 0.99**5,
        "summaries": summaries,
        "learning_curves": curves,
        "learning_stability": stability,
        "decision_gate": decision,
        "paired_track_outcomes_vs_a": (
            _paired_track_outcomes(configuration, final_by_condition)
            if len(final_by_condition) == len(configuration.conditions)
            else None
        ),
        "energy_after_feasibility": energy,
        "paired_feasible_energy": pairwise_energy,
    }


def compact_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep auditable outcomes while omitting repeated intermediate episodes."""

    final_summaries = {
        key: value
        for key, value in payload["summaries"].items()
        if key.endswith("simulator-target-300000")
    }
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "benchmark_name",
            "configuration_sha256",
            "result_file_count",
            "reserved_final_test_evaluated",
            "interaction_accounting",
            "effective_discount",
            "physical_time_matched_repeat_5_gamma_from_0_99",
            "learning_curves",
            "learning_stability",
            "decision_gate",
            "paired_track_outcomes_vs_a",
            "energy_after_feasibility",
            "paired_feasible_energy",
        )
    } | {"final_summaries": final_summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    payload = analyze(
        discover_results(args.results),
        configuration,
        require_complete=not args.allow_incomplete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if args.compact_output:
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        args.compact_output.write_text(
            json.dumps(compact_analysis(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(args.compact_output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
