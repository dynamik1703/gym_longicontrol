import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from benchmarks.constrained_rl.plot_results import (  # noqa: E402
    generate_result_plots,
    generate_trajectory_plots,
)


def test_constrained_plots_are_generated_from_persisted_artifacts(tmp_path):
    seeds = (11, 29, 47)
    constrained_curves = [
        {
            "training_seed": seed,
            "simulator_step_target": step,
            "rsr": 0.0,
        }
        for seed in seeds
        for step in (50_000, 300_000)
    ]
    scalar_curves = [
        {
            "training_seed": seed,
            "training_steps": step,
            "rsr": seed == 11 and step == 300_000,
        }
        for seed in seeds
        for step in (50_000, 300_000)
    ]
    constrained_summary = {
        "episode_count": 27,
        "failure_mode_counts": {"incomplete+time": 27},
    }
    comparison_summary = {
        "episode_count": 27,
        "failure_mode_counts": {"feasible": 5, "incomplete+time": 22},
    }
    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "learning_curves": constrained_curves,
                "summaries": {
                    "validation-3000-3008-v1/simulator-target-300000": (
                        constrained_summary
                    )
                },
                "material_improvement": {
                    "constrained_rsr_by_seed": {str(seed): 0.0 for seed in seeds},
                    "frozen_scalar_sac_rsr_by_seed": {
                        "11": 4 / 9,
                        "29": 0.0,
                        "47": 1 / 9,
                    },
                },
                "frozen_comparisons": {
                    "sb3_sac_v2b_learning_curves": scalar_curves,
                    "sb3_sac_v2b_300k": comparison_summary,
                    "credit_condition_c_300k": comparison_summary,
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
                        "speed_integral_m": 0.0,
                        "task_failure": 1.0,
                        "lagrange_speed": 0.0,
                        "lagrange_task": 0.1,
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
        "failure-modes-300k.png",
        "objective-cost-multiplier-dynamics.png",
        "optimizer-diagnostics.png",
        "seed-reliability-300k.png",
        "validation-rsr-constrained-vs-scalar.png",
    }
    assert all(path.stat().st_size > 0 for path in outputs)


def test_constrained_trajectory_plots_include_all_required_signals(tmp_path):
    path = tmp_path / "trajectories.json"
    samples = [
        {
            "time_s": 0.1,
            "position_m": 0.0,
            "velocity_m_s": 0.0,
            "speed_limit_m_s": 10.0,
            "acceleration_m_s2": 0.0,
            "jerk_m_s3": 0.0,
            "action": -0.2,
            "cumulative_energy_kwh": 0.001,
        }
    ]
    path.write_text(
        json.dumps(
            {
                "training_seed": 11,
                "trajectories": [
                    {
                        "evaluation_seed": seed,
                        "failure_mode": "incomplete+time",
                        "samples": samples,
                    }
                    for seed in (3000, 3005)
                ],
            }
        ),
        encoding="utf-8",
    )
    outputs = generate_trajectory_plots(path, tmp_path / "plots")
    assert {Path(output).name for output in outputs} == {
        "representative-trajectory-track-3000.png",
        "representative-trajectory-track-3005.png",
    }
    assert all(output.stat().st_size > 0 for output in outputs)
