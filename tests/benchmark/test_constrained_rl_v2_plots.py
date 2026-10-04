import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from benchmarks.constrained_rl_v2.plot_results import (  # noqa: E402
    generate_result_plots,
    generate_trajectory_plots,
)


def _seed_summary(rsr):
    return {
        "requirement_satisfaction_rate": rsr,
        "completion_rate": 1.0,
        "speed_compliance_rate": rsr,
    }


def test_v2_result_plots_are_generated_from_persisted_artifacts(tmp_path):
    seeds = (11, 29, 47)
    v2_curves = [
        {
            "training_seed": seed,
            "simulator_step_target": step,
            "rsr": (seed + step) % 2,
        }
        for seed in seeds
        for step in (50_000, 300_000)
    ]
    v1_curves = [
        {"training_seed": seed, "simulator_step_target": step, "rsr": 0.0}
        for seed in seeds
        for step in (50_000, 300_000)
    ]
    scalar_curves = [
        {"training_seed": seed, "training_steps": step, "rsr": seed == 11}
        for seed in seeds
        for step in (50_000, 300_000)
    ]
    v2_summary = {
        "mean_rsr": 7 / 9,
        "mean_completion_rate": 1.0,
        "mean_time_compliance_rate": 1.0,
        "mean_speed_compliance_rate": 7 / 9,
        "behavior_counts": {
            "fully_feasible": 21,
            "completed_with_speed_violation": 6,
        },
        "by_training_seed": {
            str(seed): _seed_summary(value)
            for seed, value in zip(seeds, (1.0, 5 / 9, 7 / 9))
        },
    }
    v1_summary = {
        "mean_rsr": 0.0,
        "mean_completion_rate": 0.0,
        "mean_time_compliance_rate": 0.0,
        "mean_speed_compliance_rate": 1.0,
        "by_training_seed": {
            str(seed): _seed_summary(0.0) for seed in seeds
        },
    }
    scalar_summary = {
        "mean_rsr": 5 / 27,
        "mean_completion_rate": 5 / 27,
        "mean_time_compliance_rate": 5 / 27,
        "mean_speed_compliance_rate": 1.0,
        "by_training_seed": {
            str(seed): _seed_summary(value)
            for seed, value in zip(seeds, (4 / 9, 0.0, 1 / 9))
        },
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "learning_curves": v2_curves,
                "summaries": {
                    "validation-3000-3008-v2/simulator-target-300000": v2_summary
                },
                "frozen_comparisons": {
                    "constrained_v1_learning_curves": v1_curves,
                    "constrained_v1_300k": v1_summary,
                    "constrained_v1_behavior_counts": {"standstill": 27},
                    "scalar_sb3_sac_v2b_learning_curves": scalar_curves,
                    "scalar_sb3_sac_v2b_300k": scalar_summary,
                },
            }
        ),
        encoding="utf-8",
    )
    for seed in seeds:
        directory = tmp_path / "diagnostics" / f"training-seed-{seed}"
        directory.mkdir(parents=True)
        (directory / "training-diagnostics.json").write_text(
            json.dumps(
                [
                    {
                        "simulator_steps": step,
                        "objective_return": -0.5,
                        "speed_integral_m": 0.1,
                        "deadline_deficit_integral_s": 0.2,
                        "lagrange_speed": 1.0,
                        "lagrange_deadline": 2.0,
                        "loss/q0": 0.01,
                        "loss/q1": 0.01,
                        "loss/q2": 0.01,
                        "loss/actor_total": 0.02,
                        "alpha": 0.1,
                    }
                    for step in (50_000, 300_000)
                ]
            ),
            encoding="utf-8",
        )
    outputs = generate_result_plots(
        analysis_path, tmp_path / "diagnostics", tmp_path / "plots"
    )
    assert {path.name for path in outputs} == {
        "behavior-categories-300k.png",
        "final-requirement-breakdown.png",
        "objective-cost-multiplier-dynamics.png",
        "optimizer-diagnostics.png",
        "seed-reliability-300k.png",
        "validation-rsr-v2-v1-scalar.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_v2_trajectory_plots_include_deadline_and_multiplier_signals(tmp_path):
    path = tmp_path / "trajectories.json"
    sample = {
        "time_s": 0.1,
        "position_m": 0.01,
        "velocity_m_s": 0.2,
        "speed_limit_m_s": 10.0,
        "action": 0.5,
        "acceleration_m_s2": 2.0,
        "jerk_m_s3": 10.0,
        "cumulative_energy_kwh": 0.001,
        "deadline_deficit_s": 0.0,
        "cumulative_deadline_cost_s": 0.0,
        "lambda_speed": 1.5,
        "lambda_deadline": 2.5,
    }
    path.write_text(
        json.dumps(
            {
                "training_seed": 47,
                "trajectories": [
                    {
                        "evaluation_seed": seed,
                        "behavior_category": category,
                        "samples": [sample],
                    }
                    for seed, category in (
                        (3000, "fully_feasible"),
                        (3003, "completed_with_speed_violation"),
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    outputs = generate_trajectory_plots(path, tmp_path / "plots")
    assert {Path(output).name for output in outputs} == {
        "representative-trajectory-track-3000.png",
        "representative-trajectory-track-3003.png",
    }
    assert all(output.stat().st_size > 0 for output in outputs)
