"""Physics-only selection and verification of requirement margins."""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path

from benchmarks.constrained_rl_v2.costs import optimistic_remaining_time_s
from benchmarks.scalar_sac.experiment import _base_environment
from benchmarks.scalar_sac.references import privileged_speed_action

from .config import DEFAULT_CONFIG_PATH, load_configuration


def analyze_tracks(configuration, seeds):
    environment = _base_environment(configuration)
    rows = []
    try:
        for seed in seeds:
            _observation, info = environment.reset(seed=seed)
            base = environment.unwrapped
            t_min = optimistic_remaining_time_s(
                base.track,
                position_m=0.0,
                track_length_m=base.config.track_length_m,
            )
            while True:
                action = privileged_speed_action(
                    environment, info, margin_m_s=0.5, braking_m_s2=0.75
                )
                _observation, _reward, terminated, truncated, info = environment.step(
                    action
                )
                if terminated or truncated:
                    metrics = info["episode_metrics"]
                    break
            rows.append(
                {
                    "track_seed": seed,
                    "t_min_start_s": t_min,
                    "fast_compliant_reference_time_s": metrics["travel_time_s"],
                    "reference_gap_over_t_min_s": metrics["travel_time_s"] - t_min,
                    "reference_completed": metrics["completed"],
                    "reference_max_speed_violation_m_s": metrics[
                        "max_speed_violation_m_s"
                    ],
                }
            )
    finally:
        environment.close()
    return rows


def development_analysis(configuration):
    rows = analyze_tracks(
        configuration, configuration.track_splits.development_calibration
    )
    maximum_gap = max(item["reference_gap_over_t_min_s"] for item in rows)
    selected_tight = ceil(maximum_gap / 10.0) * 10.0
    selected_training = (selected_tight, selected_tight + 20, selected_tight + 40)
    selected_interpolation = (selected_tight + 10, selected_tight + 30)
    if selected_training != configuration.requirements.training_margins_s:
        raise RuntimeError("Physics-only rule differs from canonical training margins")
    if selected_interpolation != configuration.requirements.interpolation_margins_s:
        raise RuntimeError("Physics-only rule differs from interpolation margins")
    maximum_deadline = max(item["t_min_start_s"] for item in rows) + max(
        selected_training
    )
    if maximum_deadline > configuration.requirements.time_scale_s:
        raise RuntimeError("Selected Development deadline exceeds the episode horizon")
    validation_rows = analyze_tracks(
        configuration, configuration.track_splits.validation
    )
    validation_primary_feasible = all(
        item["reference_completed"]
        and item["reference_max_speed_violation_m_s"] == 0.0
        and item["fast_compliant_reference_time_s"]
        <= item["t_min_start_s"] + min(selected_training)
        for item in validation_rows
    )
    if not validation_primary_feasible:
        raise RuntimeError("A primary Validation requirement failed physics precheck")
    return {
        "schema_version": 1,
        "selection_data": "Development tracks only",
        "reference_controller": {
            "privileged_track_access": True,
            "speed_margin_m_s": 0.5,
            "braking_m_s2": 0.75,
            "purpose": "physics-only feasibility anchor; never a learned baseline",
        },
        "selection_rule": (
            "tight margin = ceil(max Development fast-compliant-reference gap "
            "/ 10 s) * 10 s; anchors add 0/20/40 s; interpolation adds 10/30 s"
        ),
        "maximum_reference_gap_s": maximum_gap,
        "training_margins_s": list(selected_training),
        "interpolation_margins_s": list(selected_interpolation),
        "maximum_development_deadline_s": maximum_deadline,
        "episode_horizon_s": configuration.requirements.time_scale_s,
        "tracks": rows,
        "validation_precheck": {
            "performed_after_requirement_selection": True,
            "influenced_requirement_selection": False,
            "all_tight_requirements_reached_by_fast_compliant_reference": True,
            "maximum_reference_gap_s": max(
                item["reference_gap_over_t_min_s"] for item in validation_rows
            ),
            "maximum_primary_deadline_s": max(
                item["t_min_start_s"] + max(selected_training)
                for item in validation_rows
            ),
            "tracks": validation_rows,
        },
        "reserved_final_test_evaluated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = development_analysis(load_configuration(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
