"""Track-aware analysis for completed scalar SAC benchmark runs."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from gym_longicontrol.domain.task import TaskSpecification

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .evaluation import BenchmarkRunResult, EpisodeEvaluation, load_run_result
from .references import ReferenceResults, load_reference_results


def discover_results(path: str | Path) -> tuple[BenchmarkRunResult, ...]:
    root = Path(path)
    paths = sorted(root.rglob("result.json")) if root.is_dir() else [root]
    if not paths:
        raise ValueError(f"No result.json files found under {root}")
    return tuple(load_run_result(item) for item in paths)


def failure_mode(episode: EpisodeEvaluation, task: TaskSpecification) -> str:
    failures = []
    if not episode.completed:
        failures.append("incomplete")
    if episode.travel_time_s > task.max_time_s:
        failures.append("time")
    if episode.max_speed_violation_m_s > task.max_speed_violation_m_s:
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def paired_energy(
    left: Iterable[EpisodeEvaluation],
    right: Iterable[EpisodeEvaluation],
) -> dict[str, Any] | None:
    left_by_track = {item.evaluation_seed: item for item in left}
    right_by_track = {item.evaluation_seed: item for item in right}
    paired = []
    for track_seed in sorted(left_by_track.keys() & right_by_track.keys()):
        first, second = left_by_track[track_seed], right_by_track[track_seed]
        if first.feasible and second.feasible:
            paired.append(
                {
                    "evaluation_seed": track_seed,
                    "left_energy_kwh": first.energy_kwh,
                    "right_energy_kwh": second.energy_kwh,
                    "delta_energy_kwh": first.energy_kwh - second.energy_kwh,
                }
            )
    if not paired:
        return None
    differences = [item["delta_energy_kwh"] for item in paired]
    return {
        "paired_track_count": len(paired),
        "mean_delta_energy_kwh": fmean(differences),
        "median_delta_energy_kwh": float(median(differences)),
        "min_delta_energy_kwh": min(differences),
        "max_delta_energy_kwh": max(differences),
        "tracks": paired,
    }


def _reward_contributions(run: BenchmarkRunResult) -> dict[str, dict[str, float]]:
    parameters = run.reward_parameters
    values = defaultdict(list)
    for episode in run.episodes:
        values["progress"].append(
            parameters.progress_weight * episode.final_position_m / 1000.0
        )
        values["energy"].append(
            -parameters.energy_weight
            * episode.energy_kwh
            / run.energy_normalization_kwh
        )
        values["time"].append(
            -parameters.time_weight
            * episode.travel_time_s
            / run.task.max_time_s
        )
        values["speed_violation"].append(
            -parameters.speed_violation_weight
            * episode.integrated_speed_violation_m
            / run.speed_violation_normalization_m
        )
    return {
        name: {"mean": fmean(items), "median": float(median(items))}
        for name, items in values.items()
    }


def _configuration_summary(runs: Sequence[BenchmarkRunResult]) -> dict[str, Any]:
    episodes = tuple(item for run in runs for item in run.episodes)
    task = runs[0].task
    feasible_energy = [item.energy_kwh for item in episodes if item.feasible]
    seed_rsr = {
        str(run.training_seed): run.summary.requirement_satisfaction_rate
        for run in sorted(runs, key=lambda item: item.training_seed)
    }
    modes = Counter(failure_mode(item, task) for item in episodes)
    contributions = [_reward_contributions(run) for run in runs]
    contribution_names = contributions[0]
    return {
        "reward_parameters": runs[0].reward_parameters.__dict__,
        "training_seed_count": len(runs),
        "episode_count": len(episodes),
        "rsr_by_training_seed": seed_rsr,
        "mean_rsr": fmean(seed_rsr.values()),
        "min_rsr_across_training_seeds": min(seed_rsr.values()),
        "max_rsr_across_training_seeds": max(seed_rsr.values()),
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
        "min_travel_time_s": min(item.travel_time_s for item in episodes),
        "max_travel_time_s": max(item.travel_time_s for item in episodes),
        "feasible_episode_count": len(feasible_energy),
        "mean_feasible_energy_kwh": (
            fmean(feasible_energy) if feasible_energy else None
        ),
        "median_feasible_energy_kwh": (
            float(median(feasible_energy)) if feasible_energy else None
        ),
        "min_feasible_energy_kwh": (
            min(feasible_energy) if feasible_energy else None
        ),
        "max_feasible_energy_kwh": (
            max(feasible_energy) if feasible_energy else None
        ),
        "failure_mode_counts": dict(sorted(modes.items())),
        "mean_traction_energy_kwh": fmean(
            item.traction_energy_kwh for item in episodes
        ),
        "mean_regenerative_energy_kwh": fmean(
            item.regenerative_energy_kwh for item in episodes
        ),
        "mean_abs_jerk_m_s3": fmean(item.mean_abs_jerk_m_s3 for item in episodes),
        "mean_action_total_variation": fmean(
            item.action_total_variation for item in episodes
        ),
        "mean_acceleration_sign_changes": fmean(
            item.acceleration_sign_change_count for item in episodes
        ),
        "reward_contributions": {
            name: {
                "mean": fmean(item[name]["mean"] for item in contributions),
                "median_of_episode_medians": float(
                    median(item[name]["median"] for item in contributions)
                ),
            }
            for name in contribution_names
        },
    }


def _comparison(label_left, left, label_right, right):
    value = paired_energy(left, right)
    if value is None:
        return None
    return {"left": label_left, "right": label_right, **value}


def analyze_results(
    runs: Sequence[BenchmarkRunResult],
    configuration,
    references: ReferenceResults | None = None,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run.reward_parameters.configuration_id].append(run)
    expected_ids = {item.configuration_id for item in configuration.reward_grid}
    expected_seeds = set(configuration.training_seeds)
    if require_complete:
        if set(grouped) != expected_ids:
            raise ValueError("Result set does not contain all reward configurations")
        for configuration_id, items in grouped.items():
            seeds = {item.training_seed for item in items}
            if seeds != expected_seeds or len(items) != len(expected_seeds):
                raise ValueError(
                    f"Incomplete training seeds for {configuration_id}: {seeds}"
                )
            if any(
                tuple(item.evaluation_seed for item in run.episodes)
                != configuration.evaluation_seeds
                for run in items
            ):
                raise ValueError(f"Wrong evaluation tracks for {configuration_id}")
    hashes = {run.configuration_sha256 for run in runs}
    if len(hashes) != 1:
        raise ValueError("Runs were produced from different configurations")
    expected_hash = configuration_sha256(configuration)
    if hashes != {expected_hash}:
        raise ValueError("Run configuration hash does not match the analysis config")
    if references is not None and (
        references.configuration_sha256 != expected_hash
        or references.evaluation_set_id != configuration.evaluation_set_id
        or references.task != configuration.task.__dict__
    ):
        raise ValueError("Reference results do not match the analysis config")

    summaries = {
        identifier: _configuration_summary(items)
        for identifier, items in sorted(grouped.items())
    }
    by_policy = {
        (run.reward_parameters.configuration_id, run.training_seed): run
        for run in runs
    }
    configuration_pairs = []
    for left_id, right_id in itertools.combinations(sorted(grouped), 2):
        rows = []
        for training_seed in sorted(expected_seeds):
            left = by_policy.get((left_id, training_seed))
            right = by_policy.get((right_id, training_seed))
            if left is None or right is None:
                continue
            comparison = _comparison(
                f"{left_id}/seed-{training_seed}",
                left.episodes,
                f"{right_id}/seed-{training_seed}",
                right.episodes,
            )
            if comparison:
                rows.append(comparison)
        if rows:
            differences = [
                track["delta_energy_kwh"]
                for row in rows
                for track in row["tracks"]
            ]
            configuration_pairs.append(
                {
                    "left_configuration": left_id,
                    "right_configuration": right_id,
                    "paired_observation_count": len(differences),
                    "mean_delta_energy_kwh": fmean(differences),
                    "median_delta_energy_kwh": float(median(differences)),
                    "by_training_seed": rows,
                }
            )

    within_configuration = []
    for identifier, items in sorted(grouped.items()):
        for left, right in itertools.combinations(
            sorted(items, key=lambda item: item.training_seed), 2
        ):
            comparison = _comparison(
                f"seed-{left.training_seed}",
                left.episodes,
                f"seed-{right.training_seed}",
                right.episodes,
            )
            if comparison:
                within_configuration.append(
                    {"configuration_id": identifier, **comparison}
                )

    reference_comparisons = []
    reference_summary = None
    if references is not None:
        reference_summary = {
            item.policy_id: {
                "description": item.description,
                "privileged_track_access": item.privileged_track_access,
                "summary": item.summary.__dict__,
                "failure_mode_counts": dict(
                    Counter(
                        failure_mode(episode, configuration.task)
                        for episode in item.episodes
                    )
                ),
            }
            for item in references.policies
        }
        for run in runs:
            for reference in references.policies:
                comparison = _comparison(
                    f"{run.reward_parameters.configuration_id}/seed-{run.training_seed}",
                    run.episodes,
                    reference.policy_id,
                    reference.episodes,
                )
                if comparison:
                    reference_comparisons.append(comparison)

    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": next(iter(hashes)),
        "run_count": len(runs),
        "task": configuration.task.__dict__,
        "configuration_summaries": summaries,
        "paired_energy": {
            "between_reward_configurations": configuration_pairs,
            "between_training_seeds": within_configuration,
            "against_references": reference_comparisons,
        },
        "reference_summaries": reference_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    references = (
        load_reference_results(args.references) if args.references else None
    )
    analysis = analyze_results(
        discover_results(args.results),
        configuration,
        references,
        require_complete=not args.allow_incomplete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
