import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from benchmarks.scalar_sac.config import RewardParameters  # noqa: E402
from benchmarks.scalar_sac.evaluation import (  # noqa: E402
    BenchmarkRunResult,
    EpisodeEvaluation,
    EvaluationSummary,
    save_run_result,
)
from benchmarks.scalar_sac.plot_results import generate_plots  # noqa: E402
from benchmarks.scalar_sac.trajectories import (  # noqa: E402
    plot_trajectories,
    select_representatives,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics  # noqa: E402
from gym_longicontrol.domain.task import TaskSpecification  # noqa: E402


def test_headless_plots_are_generated_from_stored_results(tmp_path):
    task = TaskSpecification(60)
    metrics = EpisodeMetrics(True, 50, 0.2, 0, 0, 0)
    episodes = (EpisodeEvaluation.from_metrics(1000, metrics, task),)
    result = BenchmarkRunResult(
        benchmark_name="test",
        configuration_sha256="0" * 64,
        environment_id="StochasticTrack-v1",
        evaluation_set_id="test-seeds",
        training_seed=11,
        training_steps=1,
        task=task,
        reward_parameters=RewardParameters("test", 1, 1, 1, 1),
        energy_normalization_kwh=0.25,
        speed_violation_normalization_m=1.0,
        episodes=episodes,
        summary=EvaluationSummary.from_episodes(episodes, task),
    )
    save_run_result(tmp_path / "input" / "result.json", result)
    outputs = generate_plots(tmp_path / "input", tmp_path / "plots")
    assert {Path(path).name for path in outputs} == {
        "reward-sensitivity.png",
        "energy-vs-satisfaction.png",
        "failure-diagnostics.png",
    }
    assert all(Path(path).stat().st_size > 0 for path in outputs)


def test_headless_trajectory_plot_uses_persisted_samples(tmp_path):
    trajectory_path = tmp_path / "trajectories.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trajectories": [
                    {
                        "selection": {"label": "representative policy"},
                        "samples": [
                            {
                                "time_s": 0.1,
                                "position_m": 0.01,
                                "velocity_m_s": 0.2,
                                "speed_limit_m_s": 10.0,
                                "net_energy_kwh": 0.001,
                                "action": 0.5,
                                "acceleration_m_s2": 2.0,
                            },
                            {
                                "time_s": 0.2,
                                "position_m": 0.04,
                                "velocity_m_s": 0.4,
                                "speed_limit_m_s": 10.0,
                                "net_energy_kwh": 0.002,
                                "action": 0.4,
                                "acceleration_m_s2": 1.8,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = plot_trajectories(trajectory_path, tmp_path / "trajectory.png")
    assert output.stat().st_size > 0


def test_representative_selection_uses_common_feasible_track_and_failure_control(
    tmp_path,
):
    task = TaskSpecification(60)

    def save(identifier, training_seed, parameters, episodes):
        result = BenchmarkRunResult(
            benchmark_name="test",
            configuration_sha256="0" * 64,
            environment_id="StochasticTrack-v1",
            evaluation_set_id="test-seeds",
            training_seed=training_seed,
            training_steps=1,
            task=task,
            reward_parameters=parameters,
            energy_normalization_kwh=0.25,
            speed_violation_normalization_m=1.0,
            episodes=episodes,
            summary=EvaluationSummary.from_episodes(episodes, task),
        )
        save_run_result(
            tmp_path / identifier / f"training-seed-{training_seed}" / "result.json",
            result,
        )

    feasible = EpisodeMetrics(True, 50, 0.2, 0, 0, 0)
    incomplete = EpisodeMetrics(False, 60, 0.04, 0, 0, 0)
    save(
        "successful-a",
        11,
        RewardParameters("successful-a", 1, 0.5, 0.25, 0),
        (
            EpisodeEvaluation.from_metrics(1000, incomplete, task),
            EpisodeEvaluation.from_metrics(1001, feasible, task),
        ),
    )
    save(
        "successful-b",
        29,
        RewardParameters("successful-b", 1, 0.5, 1, 2),
        (
            EpisodeEvaluation.from_metrics(1000, feasible, task),
            EpisodeEvaluation.from_metrics(1001, feasible, task),
        ),
    )
    save(
        "failed",
        47,
        RewardParameters("failed", 1, 2, 1, 0),
        (
            EpisodeEvaluation.from_metrics(1000, incomplete, task),
            EpisodeEvaluation.from_metrics(1001, incomplete, task),
        ),
    )

    selections = select_representatives(tmp_path)
    assert [item.configuration_id for item in selections] == [
        "successful-a",
        "successful-b",
        "failed",
    ]
    assert {item.evaluation_seed for item in selections} == {1001}
    assert selections[-1].label.endswith("(failure control)")
