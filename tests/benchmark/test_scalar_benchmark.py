import json
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.scalar_sac.analysis import analyze_results, failure_mode, paired_energy
from benchmarks.scalar_sac.config import (
    DEFAULT_CONFIG_PATH,
    RewardParameters,
    configuration_sha256,
    load_configuration,
)
from benchmarks.scalar_sac.evaluation import (
    BenchmarkRunResult,
    EpisodeEvaluation,
    EvaluationSummary,
    evaluate_policy,
    load_run_result,
    save_run_result,
)
from benchmarks.scalar_sac.reward import ScalarBenchmarkReward
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification


def reward_parameters():
    return RewardParameters("test-grid-point", 1.0, 2.0, 0.5, 3.0)


def test_canonical_configuration_loads_and_freezes_the_protocol():
    configuration = load_configuration()
    assert DEFAULT_CONFIG_PATH.name == "canonical.json"
    assert configuration.environment_id == "StochasticTrack-v1"
    assert configuration.task == TaskSpecification(140.0, 0.0)
    assert configuration.time_budget_sweep == (120.0, 140.0, 160.0)
    assert configuration.training_seeds == (11, 29, 47)
    assert configuration.calibration_seeds == tuple(range(2000, 2009))
    assert configuration.evaluation_seeds == tuple(range(1000, 1009))
    assert len(configuration.reward_grid) == 8
    assert len({item.configuration_id for item in configuration.reward_grid}) == 8
    assert len(configuration_sha256(configuration)) == 64


def test_configuration_loading_rejects_invalid_task(tmp_path):
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["task"]["max_time_s"] = 0
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="max_time_s"):
        load_configuration(path)


def test_reward_wrapper_replaces_only_reward_and_preserves_dynamics():
    base = gym.make("DeterministicTrack-v1", max_episode_steps=10)
    compared = gym.make("DeterministicTrack-v1", max_episode_steps=10)
    wrapped = ScalarBenchmarkReward(
        compared,
        parameters=reward_parameters(),
        task=TaskSpecification(120),
        energy_normalization_kwh=0.25,
        speed_violation_normalization_m=1.0,
    )
    try:
        first_observation, first_info = base.reset(seed=9)
        second_observation, second_info = wrapped.reset(seed=9)
        np.testing.assert_array_equal(first_observation, second_observation)
        assert first_info == second_info
        for action in ([1.0], [0.2], [-0.5], [0.8]):
            first = base.step(action)
            second = wrapped.step(action)
            np.testing.assert_array_equal(first[0], second[0])
            assert first[2:4] == second[2:4]
            assert base.unwrapped.state == wrapped.unwrapped.state
            assert second[4]["historical_reward"] == first[1]
            assert second[4]["reward_components"] == first[4]["reward_components"]
            assert np.isfinite(second[1])
            assert second[1] == pytest.approx(
                sum(second[4]["benchmark_reward_components"].values())
            )
    finally:
        base.close()
        wrapped.close()


def test_scalar_reward_formula_uses_documented_normalizations():
    environment = gym.make("DeterministicTrack-v1", max_episode_steps=2)
    wrapped = ScalarBenchmarkReward(
        environment,
        parameters=reward_parameters(),
        task=TaskSpecification(100),
        energy_normalization_kwh=0.5,
        speed_violation_normalization_m=2.0,
    )
    try:
        wrapped.reset(seed=1)
        _, reward, _, _, info = wrapped.step([1.0])
        expected = {
            "progress": info["position_m"] / 1000.0,
            "energy": -2.0 * info["step_energy_kwh"] / 0.5,
            "time": -0.5 * info["elapsed_time_s"] / 100.0,
            "speed_violation": (
                -3.0
                * info["speed_excess_m_s"]
                * info["elapsed_time_s"]
                / 2.0
            ),
        }
        assert info["benchmark_reward_components"] == pytest.approx(expected)
        assert reward == pytest.approx(sum(expected.values()))
    finally:
        wrapped.close()


def _episode(seed, metrics, task):
    return EpisodeEvaluation.from_metrics(seed, metrics, task)


def test_evaluation_aggregates_energy_only_over_feasible_episodes():
    task = TaskSpecification(60.0, 0.0)
    episodes = (
        _episode(1, EpisodeMetrics(True, 50, 0.2, 0, 0, 0), task),
        _episode(2, EpisodeMetrics(True, 70, 0.1, 0, 0, 0), task),
        _episode(3, EpisodeMetrics(True, 55, 0.05, 1, 0.1, 0.5), task),
        _episode(4, EpisodeMetrics(False, 60, 0.0, 0, 0, 0), task),
    )
    summary = EvaluationSummary.from_episodes(episodes, task)
    assert summary.completion_rate == 0.75
    assert summary.requirement_satisfaction_rate == 0.25
    assert summary.mean_feasible_energy_kwh == 0.2
    assert summary.median_feasible_energy_kwh == 0.2
    assert summary.incomplete_rate == 0.25
    assert summary.time_violation_rate == 0.25
    assert summary.speed_violation_rate == 0.25
    assert summary.mean_integrated_speed_violation_m == 0.125


def test_no_feasible_episode_reports_missing_energy_not_zero():
    task = TaskSpecification(10)
    episodes = (_episode(1, EpisodeMetrics(False, 10, 0, 0, 0, 0), task),)
    summary = EvaluationSummary.from_episodes(episodes, task)
    assert summary.mean_feasible_energy_kwh is None
    assert summary.median_feasible_energy_kwh is None


def test_fixed_seed_evaluation_is_deterministic_and_ignores_reward():
    task = TaskSpecification(120)

    def policy(_observation):
        return np.array([0.25])

    first = gym.make("StochasticTrack-v1", max_episode_steps=4)
    second = gym.make(
        "StochasticTrack-v1",
        max_episode_steps=4,
        reward_weights=[9.0, -2.0, 3.0, 7.0],
    )
    try:
        result_a = evaluate_policy(
            policy, first, task=task, evaluation_seeds=(101, 202)
        )
        result_b = evaluate_policy(
            policy, second, task=task, evaluation_seeds=(101, 202)
        )
        assert result_a == result_b
        for episode in result_a[0]:
            assert episode.step_count == 4
            assert episode.final_position_m > 0
            assert episode.traction_energy_kwh - episode.regenerative_energy_kwh == (
                pytest.approx(episode.energy_kwh)
            )
    finally:
        first.close()
        second.close()


def test_result_serialization_roundtrip_and_summary_validation(tmp_path):
    task = TaskSpecification(60)
    episodes = (
        _episode(1000, EpisodeMetrics(True, 55, 0.25, 0, 0, 0), task),
    )
    result = BenchmarkRunResult(
        benchmark_name="test",
        configuration_sha256="0" * 64,
        environment_id="StochasticTrack-v1",
        evaluation_set_id="test-seeds",
        training_seed=11,
        training_steps=123,
        task=task,
        reward_parameters=reward_parameters(),
        energy_normalization_kwh=0.25,
        speed_violation_normalization_m=1.0,
        episodes=episodes,
        summary=EvaluationSummary.from_episodes(episodes, task),
    )
    path = save_run_result(tmp_path / "result.json", result)
    assert load_run_result(path) == result
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["summary"]["completion_rate"] = 0.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="summary"):
        load_run_result(path)


def test_reward_parameter_validation():
    with pytest.raises(ValueError):
        replace(reward_parameters(), speed_violation_weight=-1)
    with pytest.raises(ValueError):
        replace(reward_parameters(), configuration_id="")


def test_failure_modes_and_paired_energy_require_joint_feasibility():
    task = TaskSpecification(60)
    feasible = _episode(1, EpisodeMetrics(True, 50, 0.2, 0, 0, 0), task)
    slow = _episode(2, EpisodeMetrics(True, 70, 0.1, 0, 0, 0), task)
    speeding = _episode(3, EpisodeMetrics(True, 50, 0.1, 1, 0.2, 1), task)
    assert failure_mode(feasible, task) == "feasible"
    assert failure_mode(slow, task) == "time"
    assert failure_mode(speeding, task) == "speed"
    comparison = paired_energy(
        (feasible, slow),
        (
            replace(feasible, energy_kwh=0.15),
            replace(slow, feasible=True, energy_kwh=0.05),
        ),
    )
    assert comparison is not None
    assert comparison["paired_track_count"] == 1
    assert comparison["mean_delta_energy_kwh"] == pytest.approx(0.05)


def test_analysis_rejects_results_from_a_different_configuration():
    configuration = load_configuration()
    task = configuration.task
    episodes = (
        _episode(1000, EpisodeMetrics(True, 100, 0.2, 0, 0, 0), task),
    )
    run = BenchmarkRunResult(
        benchmark_name=configuration.name,
        configuration_sha256="0" * 64,
        environment_id=configuration.environment_id,
        evaluation_set_id=configuration.evaluation_set_id,
        training_seed=11,
        training_steps=1,
        task=task,
        reward_parameters=configuration.reward_grid[0],
        energy_normalization_kwh=configuration.energy_normalization_kwh,
        speed_violation_normalization_m=(
            configuration.speed_violation_normalization_m
        ),
        episodes=episodes,
        summary=EvaluationSummary.from_episodes(episodes, task),
    )
    with pytest.raises(ValueError, match="hash"):
        analyze_results((run,), configuration, require_complete=False)
