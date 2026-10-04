"""Analyze Binary Success Reward V1 without using training return as evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, median
from typing import Any

from benchmarks.scalar_sac.analysis import failure_mode

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import BinaryRunResult, load_result

DEFAULT_SCALAR_RESULTS = Path("benchmarks/scalar_sb3/results.json")
DEFAULT_CONSTRAINED_RESULTS = Path("benchmarks/constrained_rl_v2/results.json")
DEFAULT_REQUIREMENT_RESULTS = Path("benchmarks/requirement_conditioned/results.json")


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def discover_results(root: str | Path) -> tuple[BinaryRunResult, ...]:
    paths = sorted(Path(root).glob("training-seed-*/step-*/*-result.json"))
    if not paths:
        raise ValueError("No binary result files found")
    return tuple(load_result(path) for path in paths)


def _summary(runs: Sequence[BinaryRunResult]) -> dict[str, Any]:
    episodes = tuple(item for run in runs for item in run.episodes)
    task = runs[0].task
    feasible_energy = [item.energy_kwh for item in episodes if item.feasible]
    return {
        "episode_count": len(episodes),
        "feasible_count": sum(item.feasible for item in episodes),
        "requirement_satisfaction_rate": sum(item.feasible for item in episodes)
        / len(episodes),
        "completion_rate": sum(item.completed for item in episodes) / len(episodes),
        "deadline_compliance_rate": sum(
            item.travel_time_s <= task.max_time_s for item in episodes
        )
        / len(episodes),
        "speed_compliance_rate": sum(
            item.max_speed_violation_m_s <= task.max_speed_violation_m_s
            for item in episodes
        )
        / len(episodes),
        "mean_travel_time_s": fmean(item.travel_time_s for item in episodes),
        "median_travel_time_s": float(median(item.travel_time_s for item in episodes)),
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


def _validate_complete(runs, configuration):
    expected_hash = configuration_sha256(configuration)
    if {run.configuration_sha256 for run in runs} != {expected_hash}:
        raise ValueError("Binary result configuration hash mismatch")
    reserved = set(configuration.track_splits.paper_final_test_reserved)
    if any(item.evaluation_seed in reserved for run in runs for item in run.episodes):
        raise ValueError("Reserved final-test seed occurs in binary results")
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run.evaluation_split_id, run.training_steps)].append(run)
    expected = {
        (split, step)
        for split in ("development-2000-2008-v1", "validation-3000-3008-v1")
        for step in configuration.learning_curve_steps
    }
    if set(grouped) != expected:
        raise ValueError(
            f"Incomplete binary checkpoints: missing={expected - set(grouped)}, "
            f"extra={set(grouped) - expected}"
        )
    for key, values in grouped.items():
        seeds = {item.training_seed for item in values}
        if seeds != set(configuration.training_seeds) or len(values) != 3:
            raise ValueError(f"Incomplete binary training seeds for {key}: {seeds}")
    return grouped


def _training_outcomes(root: Path, configuration) -> dict[str, Any]:
    by_seed = {}
    all_rows = []
    for seed in configuration.training_seeds:
        payload = _read(root / f"training-seed-{seed}" / "training-outcomes.json")
        if payload["configuration_sha256"] != configuration_sha256(configuration):
            raise ValueError("Training outcome configuration hash mismatch")
        rows = payload["episodes"]
        successes = [item for item in rows if item["success"]]
        if payload["successful_training_episode_count"] != len(successes):
            raise ValueError("Stored training success count is inconsistent")
        intervals = [
            right["training_step"] - left["training_step"]
            for left, right in zip(successes, successes[1:])
        ]
        bins = {}
        cumulative = {}
        for checkpoint in configuration.learning_curve_steps:
            lower = checkpoint - 50_000
            bins[str(checkpoint)] = sum(
                lower < item["training_step"] <= checkpoint for item in successes
            )
            cumulative[str(checkpoint)] = sum(
                item["training_step"] <= checkpoint for item in successes
            )
        by_seed[str(seed)] = {
            "completed_training_episode_count": len(rows),
            "successful_training_episode_count": len(successes),
            "training_episode_success_rate": (
                len(successes) / len(rows) if rows else 0.0
            ),
            "first_success_training_step": (
                successes[0]["training_step"] if successes else None
            ),
            "last_success_training_step": (
                successes[-1]["training_step"] if successes else None
            ),
            "median_steps_between_successes": (
                float(median(intervals)) if intervals else None
            ),
            "maximum_steps_between_successes": max(intervals) if intervals else None,
            "successes_by_50k_interval": bins,
            "cumulative_successes_by_checkpoint": cumulative,
            "failure_mode_counts": dict(
                sorted(Counter(item["failure_mode"] for item in rows).items())
            ),
        }
        all_rows.extend({**item, "training_seed": seed} for item in rows)
    return {
        "by_training_seed": by_seed,
        "pooled_completed_training_episode_count": len(all_rows),
        "pooled_successful_training_episode_count": sum(
            item["success"] for item in all_rows
        ),
        "pooled_training_episode_success_rate": sum(
            item["success"] for item in all_rows
        )
        / len(all_rows),
    }


def _criterion(value, threshold, *, maximum=False):
    passed = value is not None and (
        value <= threshold if maximum else value >= threshold
    )
    return {"value": value, "threshold": threshold, "passed": passed}


def decision_gate(criteria, final_by_seed, training, feasible_count) -> tuple[str, str]:
    """Apply the preregistered gate order exactly."""

    if all(item["passed"] for item in criteria.values()):
        return "A", "Every predefined credibility criterion passes."
    total_training_success = training["pooled_successful_training_episode_count"]
    if total_training_success == 0 and feasible_count == 0:
        return "D", "Training and final Validation never escape zero success."
    seed_values = [
        final_by_seed[str(seed)]["requirement_satisfaction_rate"]
        for seed in sorted(map(int, final_by_seed))
    ]
    if max(seed_values) >= 5 / 9 and (
        min(seed_values) < 3 / 9 or max(seed_values) - min(seed_values) >= 5 / 9
    ):
        return "C", "Sparse learning is strongly dependent on the training seed."
    if feasible_count > 5 and all(value > 0 for value in seed_values):
        return "B", "Binary reward produces partial but non-robust learning."
    if total_training_success > 0 and feasible_count <= 5:
        return "E", "Training successes occur without useful final generalization."
    return "F", "The outcome is mixed sparse evidence not covered by stronger gates."


def analyze(
    results_directory: str | Path,
    configuration,
    *,
    scalar_results_path: str | Path = DEFAULT_SCALAR_RESULTS,
    constrained_results_path: str | Path = DEFAULT_CONSTRAINED_RESULTS,
    requirement_results_path: str | Path = DEFAULT_REQUIREMENT_RESULTS,
) -> dict[str, Any]:
    root = Path(results_directory)
    runs = discover_results(root)
    grouped = _validate_complete(runs, configuration)
    summaries = {
        f"{split}/step-{step}": _summary(values)
        for (split, step), values in sorted(grouped.items())
    }
    validation_runs = [
        run for run in runs if run.evaluation_split_id == "validation-3000-3008-v1"
    ]
    learning_curves = [
        {
            "training_seed": run.training_seed,
            "training_steps": run.training_steps,
            "rsr": run.summary.requirement_satisfaction_rate,
            "completion_rate": run.summary.completion_rate,
            "deadline_compliance_rate": 1.0 - run.summary.time_violation_rate,
            "speed_compliance_rate": run.summary.speed_compliance_rate,
        }
        for run in sorted(
            validation_runs, key=lambda item: (item.training_seed, item.training_steps)
        )
    ]
    final_step = configuration.total_training_steps
    final_runs = sorted(
        grouped[("validation-3000-3008-v1", final_step)],
        key=lambda item: item.training_seed,
    )
    final_by_seed = {
        str(run.training_seed): {
            **_summary((run,)),
            "policy_updates": run.policy_updates,
            "training_wall_time_s": run.training_wall_time_s,
            "diagnostics": run.diagnostics,
        }
        for run in final_runs
    }
    final_summary = _summary(final_runs)
    stability = []
    for seed in configuration.training_seeds:
        rows = [row for row in learning_curves if row["training_seed"] == seed]
        peak = max(rows, key=lambda item: item["rsr"])
        final = rows[-1]
        stability.append(
            {
                "training_seed": seed,
                "peak_rsr": peak["rsr"],
                "peak_training_steps": peak["training_steps"],
                "final_rsr": final["rsr"],
                "peak_to_final_rsr_drop": peak["rsr"] - final["rsr"],
                "final_minus_250k_rsr": final["rsr"] - rows[-2]["rsr"],
            }
        )
    training = _training_outcomes(root, configuration)
    acceptance = configuration.acceptance
    seed_rsr = [
        final_by_seed[str(seed)]["requirement_satisfaction_rate"]
        for seed in configuration.training_seeds
    ]
    training_successes = [
        training["by_training_seed"][str(seed)]["successful_training_episode_count"]
        for seed in configuration.training_seeds
    ]
    criteria = {
        "pooled_validation_rsr": _criterion(
            final_summary["requirement_satisfaction_rate"],
            acceptance.minimum_pooled_validation_rsr,
        ),
        "minimum_seed_rsr": _criterion(
            min(seed_rsr), acceptance.minimum_rsr_per_training_seed
        ),
        "seed_count_at_half_rsr": _criterion(
            sum(value >= 0.5 for value in seed_rsr),
            acceptance.minimum_seed_count_at_half_rsr,
        ),
        "minimum_training_successes_per_seed": _criterion(
            min(training_successes), acceptance.minimum_training_successes_per_seed
        ),
        "maximum_peak_to_final_rsr_drop": _criterion(
            max(item["peak_to_final_rsr_drop"] for item in stability),
            acceptance.maximum_peak_to_final_rsr_drop,
            maximum=True,
        ),
    }
    gate, reason = decision_gate(
        criteria,
        final_by_seed,
        training,
        final_summary["feasible_count"],
    )

    scalar = _read(scalar_results_path)
    scalar_final = scalar["summaries"]["sac/validation-3000-3008-v1/step-300000"]
    constrained = _read(constrained_results_path)["summaries"][
        "validation-3000-3008-v2/simulator-target-300000"
    ]
    requirement = _read(requirement_results_path)["final_validation"]["overall"]
    if round(scalar_final["mean_rsr"] * 27) != 5:
        raise ValueError("Frozen Scalar SB3 SAC no longer equals 5/27")
    if round(constrained["mean_rsr"] * 27) != 21:
        raise ValueError("Frozen Constrained V2 no longer equals 21/27")
    scalar_curve = [
        row for row in scalar["learning_curves"] if row["algorithm"] == "sac"
    ]
    return {
        "schema_version": 1,
        "benchmark_name": configuration.name,
        "configuration_sha256": configuration_sha256(configuration),
        "result_file_count": len(runs),
        "reserved_final_test_evaluated": False,
        "summaries": summaries,
        "final_validation": {
            "overall": final_summary,
            "by_training_seed": final_by_seed,
        },
        "learning_curves": learning_curves,
        "learning_stability": stability,
        "training_outcomes": training,
        "acceptance": criteria,
        "decision_gate": gate,
        "decision_reason": reason,
        "frozen_comparisons": {
            "scalar_sb3_sac": {
                "feasible_count": 5,
                "episode_count": 27,
                "requirement_satisfaction_rate": scalar_final["mean_rsr"],
                "learning_curves": scalar_curve,
                "source": str(scalar_results_path),
            },
            "constrained_v2": {
                "feasible_count": 21,
                "episode_count": 27,
                "requirement_satisfaction_rate": constrained["mean_rsr"],
                "source": str(constrained_results_path),
            },
            "requirement_conditioned_v1": {
                "feasible_count": requirement["feasible_count"],
                "episode_count": requirement["episode_count"],
                "requirement_satisfaction_rate": requirement[
                    "requirement_satisfaction_rate"
                ],
                "source": str(requirement_results_path),
                "note": "Different five-requirement validation matrix",
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--scalar-results", type=Path, default=DEFAULT_SCALAR_RESULTS)
    parser.add_argument(
        "--constrained-results", type=Path, default=DEFAULT_CONSTRAINED_RESULTS
    )
    parser.add_argument(
        "--requirement-results", type=Path, default=DEFAULT_REQUIREMENT_RESULTS
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = analyze(
        args.results_directory,
        load_configuration(args.config),
        scalar_results_path=args.scalar_results,
        constrained_results_path=args.constrained_results,
        requirement_results_path=args.requirement_results,
    )
    print(_write(args.output, payload))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
