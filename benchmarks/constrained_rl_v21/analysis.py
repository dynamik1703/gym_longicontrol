"""Aggregate the frozen V2 trajectory diagnosis into a reviewable artifact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_rl_v2.config import (
    configuration_sha256,
    load_configuration,
)
from benchmarks.constrained_rl_v2.results import load_result

from .diagnosis import (
    comparison_track,
    extract_violation_events,
    speed_limit_transitions,
    trajectory_smoothness,
)

RESULT_SCHEMA_VERSION = 1


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return a compact deterministic five-number-style summary."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("Distribution values must be finite")
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Return Pearson r, or None where a constant/short vector makes it undefined."""

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Correlation inputs must be matching vectors")
    if left.size < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def event_mechanism(event: Mapping[str, Any], *, dt_s: float = 0.1) -> str:
    """Classify a physical event without changing benchmark feasibility.

    A reduction event begins no later than one native control interval after the
    crossing. Boundary tracking is limited by the maximum possible one-step speed
    change (3 m/s² * 0.1 s = 0.3 m/s), a physical rather than tuned threshold.
    """

    transition = event.get("nearest_speed_limit_transition")
    if transition and transition["kind"] == "reduction":
        signed_time = float(transition["signed_time_s"])
        if 0 <= signed_time <= dt_s + 1e-12:
            return "late_braking_after_speed_limit_reduction"
    if float(event["peak_overspeed_m_s"]) <= 3.0 * dt_s + 1e-12:
        return "small_constant_section_boundary_overshoot"
    return "other"


def dominant_failure_classification(failure_mechanisms: Sequence[str]) -> str:
    """Map homogeneous mechanisms to the requested A/B classes; mixtures stop."""

    mechanisms = set(failure_mechanisms)
    if mechanisms == {"small_constant_section_boundary_overshoot"}:
        return "A"
    if mechanisms == {"late_braking_after_speed_limit_reduction"}:
        return "B"
    return "F"


def _episode_record(trajectory: Mapping[str, Any]) -> dict[str, Any]:
    transitions = speed_limit_transitions(
        trajectory["track"]["positions_m"], trajectory["track"]["limits_m_s"]
    )
    events = [
        {**event, "mechanism": event_mechanism(event)}
        for event in extract_violation_events(trajectory["samples"], transitions)
    ]
    integrated = sum(float(event["integrated_overspeed_m"]) for event in events)
    if len(events) != int(trajectory["speed_violation_count"]) or not np.isclose(
        integrated,
        float(trajectory["integrated_speed_violation_m"]),
        atol=1e-10,
    ):
        raise ValueError("Reconstructed violation events differ from EpisodeMetrics")
    samples = trajectory["samples"]
    return {
        "training_seed": int(trajectory["training_seed"]),
        "track_seed": int(trajectory["track_seed"]),
        "feasible": bool(trajectory["feasible"]),
        "travel_time_s": float(trajectory["travel_time_s"]),
        "energy_kwh": float(trajectory["energy_kwh"]),
        "speed_violation_count": int(trajectory["speed_violation_count"]),
        "max_speed_violation_m_s": float(trajectory["max_speed_violation_m_s"]),
        "integrated_speed_violation_m": float(
            trajectory["integrated_speed_violation_m"]
        ),
        "minimum_deadline_slack_s": float(
            min(item["deadline_slack_s"] for item in samples)
        ),
        "maximum_deadline_deficit_s": float(
            max(item["deadline_deficit_s"] for item in samples)
        ),
        "deadline_deficit_integral_s": float(
            sum(item["deadline_cost_s"] for item in samples)
        ),
        "smoothness": trajectory_smoothness(samples),
        "track": trajectory["track"],
        "violation_events": events,
    }


def _training_diagnostics(results_directory: Path, configuration) -> dict[str, Any]:
    by_seed = {}
    for seed in configuration.training_seeds:
        seed_directory = results_directory / f"training-seed-{seed}"
        rows = json.loads(
            (seed_directory / "training-diagnostics.json").read_text(encoding="utf-8")
        )
        curve = []
        for target in configuration.simulator_step_checkpoints:
            result = load_result(
                seed_directory
                / f"target-{target:06d}"
                / "validation-3000-3008-result.json"
            )
            curve.append(
                {
                    "simulator_step_target": target,
                    "actual_simulator_steps": result.simulator_steps,
                    "rsr": result.summary.requirement_satisfaction_rate,
                    "speed_compliance_rate": result.summary.speed_compliance_rate,
                    "completion_rate": result.summary.completion_rate,
                }
            )
        speed_lambdas = [float(row["lagrange_speed"]) for row in rows]
        deadline_lambdas = [float(row["lagrange_deadline"]) for row in rows]
        speed_costs = [float(row["speed_integral_m"]) for row in rows]
        deadline_costs = [float(row["deadline_deficit_integral_s"]) for row in rows]
        by_seed[str(seed)] = {
            "episode_count": len(rows),
            "first_positive_speed_cost_step": next(
                (
                    int(row["simulator_steps"])
                    for row in rows
                    if float(row["speed_integral_m"]) > 0
                ),
                None,
            ),
            "first_positive_deadline_cost_step": next(
                (
                    int(row["simulator_steps"])
                    for row in rows
                    if float(row["deadline_deficit_integral_s"]) > 0
                ),
                None,
            ),
            "lambda_speed": {
                **distribution(speed_lambdas),
                "final": speed_lambdas[-1],
            },
            "lambda_deadline": {
                **distribution(deadline_lambdas),
                "final": deadline_lambdas[-1],
            },
            "training_speed_cost_return": distribution(speed_costs),
            "training_deadline_cost_return": distribution(deadline_costs),
            "checkpoint_validation": curve,
        }
    return by_seed


def analyze(
    trajectory_path: str | Path,
    v2_results_directory: str | Path,
) -> dict[str, Any]:
    """Build the formal post-hoc diagnosis from frozen artifacts."""

    source = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    trajectories = source["trajectories"]
    if len(trajectories) != 27:
        raise ValueError("The diagnosis requires all 27 frozen Validation episodes")
    records = [_episode_record(item) for item in trajectories]
    failed = [record for record in records if not record["feasible"]]
    successful = [record for record in records if record["feasible"]]
    if len(failed) != 6 or len(successful) != 21:
        raise ValueError("Frozen V2 must contain exactly 21 successes and 6 failures")

    for failure in failed:
        same_seed = [
            item
            for item in trajectories
            if item["training_seed"] == failure["training_seed"] and item["feasible"]
        ]
        original = next(
            item
            for item in trajectories
            if item["training_seed"] == failure["training_seed"]
            and item["track_seed"] == failure["track_seed"]
        )
        failure["comparison_success_track_seed"] = comparison_track(original, same_seed)
        comparison = next(
            item
            for item in records
            if item["training_seed"] == failure["training_seed"]
            and item["track_seed"] == failure["comparison_success_track_seed"]
        )
        failure["comparison_success"] = {
            key: value
            for key, value in comparison.items()
            if key not in {"track", "violation_events"}
        }

    episode_fields = (
        "travel_time_s",
        "energy_kwh",
        "speed_violation_count",
        "max_speed_violation_m_s",
        "integrated_speed_violation_m",
        "minimum_deadline_slack_s",
        "maximum_deadline_deficit_s",
    )
    smoothness_fields = tuple(records[0]["smoothness"])
    group_summaries = {}
    for name, group in (("successful", successful), ("failed", failed)):
        group_summaries[name] = {
            field: distribution([float(item[field]) for item in group])
            for field in episode_fields
        }
        group_summaries[name]["smoothness"] = {
            field: distribution([float(item["smoothness"][field]) for item in group])
            for field in smoothness_fields
        }

    events = [event for failure in failed for event in failure["violation_events"]]
    mechanisms = [event["mechanism"] for event in events]
    episode_mechanisms = [
        "+".join(sorted({event["mechanism"] for event in item["violation_events"]}))
        for item in failed
    ]
    violating_samples = []
    for trajectory in trajectories:
        if trajectory["feasible"]:
            continue
        violating_samples.extend(
            item
            for item in trajectory["samples"]
            if float(item["speed_excess_m_s"]) > 0
        )

    by_training_seed = Counter(str(item["training_seed"]) for item in failed)
    by_track_seed = {
        str(track): [
            int(item["training_seed"]) for item in failed if item["track_seed"] == track
        ]
        for track in sorted({item["track_seed"] for item in failed})
    }
    deadline_deficits = [
        float(item["deadline_deficit_s"]) for item in violating_samples
    ]
    speed_excess = [float(item["speed_excess_m_s"]) for item in violating_samples]
    deadline_slack = [float(item["deadline_slack_s"]) for item in violating_samples]

    classification = dominant_failure_classification(mechanisms)
    configuration = load_configuration()
    if source["configuration_sha256"] != configuration_sha256(configuration):
        raise ValueError("Trajectory source does not match the frozen V2 configuration")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study": "constrained-rl-v2-post-hoc-speed-diagnosis",
        "source_configuration_sha256": source["configuration_sha256"],
        "source_episode_counts": {"successful": 21, "failed": 6},
        "failed_episodes": failed,
        "group_distributions": group_summaries,
        "event_summary": {
            "event_count": len(events),
            "mechanism_counts": dict(sorted(Counter(mechanisms).items())),
            "episode_mechanism_counts": dict(
                sorted(Counter(episode_mechanisms).items())
            ),
            "peak_overspeed_m_s": distribution(
                [float(item["peak_overspeed_m_s"]) for item in events]
            ),
            "integrated_overspeed_m": distribution(
                [float(item["integrated_overspeed_m"]) for item in events]
            ),
            "duration_s": distribution([float(item["duration_s"]) for item in events]),
            "distance_while_violating_m": distribution(
                [float(item["distance_while_violating_m"]) for item in events]
            ),
        },
        "deadline_interaction": {
            "violating_step_count": len(violating_samples),
            "violating_steps_with_positive_deadline_deficit": int(
                np.count_nonzero(np.asarray(deadline_deficits) > 0)
            ),
            "minimum_slack_during_speed_violation_s": min(deadline_slack),
            "speed_excess_vs_deadline_slack_pearson_r": pearson_correlation(
                speed_excess, deadline_slack
            ),
            "speed_excess_vs_deadline_deficit_pearson_r": pearson_correlation(
                speed_excess, deadline_deficits
            ),
            "interpretation": (
                "All violating steps have zero deadline deficit and substantial "
                "positive slack; correlation does not establish causality."
            ),
        },
        "failure_concentration": {
            "by_training_seed": dict(sorted(by_training_seed.items())),
            "by_track_seed": by_track_seed,
            "repeated_failed_track_across_training_seeds": False,
        },
        "training_diagnostics": _training_diagnostics(
            Path(v2_results_directory), configuration
        ),
        "formal_diagnosis": {
            "classification": classification,
            "label": "mixed / no single intervention follows",
            "v21_training_justified": False,
            "reason": (
                "Four failed episodes are small constant-section boundary tracking "
                "overshoots, while two contain physically meaningful late braking "
                "after speed-limit reductions. A margin-only intervention does not "
                "address late braking, and an anticipatory-braking-only intervention "
                "does not address the four constant-section failures."
            ),
            "decision_gate": "F",
            "freeze_constrained_rl": True,
            "next_research_stage": "Requirement-Conditioned RL",
        },
        "reserved_final_test_evaluated": False,
    }


def write_analysis(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument(
        "--v2-results",
        type=Path,
        default=Path("runs/constrained-rl-v2-20261002"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        write_analysis(
            analyze(args.trajectories, args.v2_results),
            args.output,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
