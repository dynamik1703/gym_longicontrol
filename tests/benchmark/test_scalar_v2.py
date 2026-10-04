import json

import gymnasium as gym
import pytest

from benchmarks.scalar_sac.evaluation import (
    EpisodeEvaluation,
    EvaluationSummary,
)
from benchmarks.scalar_sac.v2_config import (
    DEFAULT_V2_CONFIG_PATH,
    load_v2_configuration,
    v2_configuration_sha256,
)
from benchmarks.scalar_sac.v2_evaluation import (
    V2BenchmarkRunResult,
    load_v2_result,
    save_v2_result,
)
from benchmarks.scalar_sac.v2_reward import (
    ScalarBenchmarkRewardV2,
    approximate_episode_return,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.state import VehicleState


@pytest.mark.parametrize("config_name", ["canonical_v2.json", "canonical_v2b.json"])
def test_v2_configuration_freezes_disjoint_splits_and_acceptance_thresholds(
    config_name,
):
    configuration = load_v2_configuration(DEFAULT_V2_CONFIG_PATH.with_name(config_name))
    assert configuration.training.total_training_steps == 300_000
    assert configuration.comparison_steps == (100_000, 300_000)
    assert len(configuration.reward_candidates) == 3
    assert configuration.track_splits.validation == tuple(range(3000, 3009))
    assert configuration.track_splits.paper_final_test_reserved == tuple(
        range(4000, 4018)
    )
    assert len(v2_configuration_sha256(configuration)) == 64


def test_v2_configuration_rejects_overlapping_track_splits(tmp_path):
    raw = json.loads(DEFAULT_V2_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["track_splits"]["validation"][0] = 2000
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_v2_configuration(path)


@pytest.mark.parametrize("config_name", ["canonical_v2.json", "canonical_v2b.json"])
@pytest.mark.parametrize("candidate_index", range(3))
def test_all_v2_candidates_prefer_representative_feasible_drive_to_standstill(
    config_name, candidate_index
):
    configuration = load_v2_configuration(
        DEFAULT_V2_CONFIG_PATH.with_name(config_name)
    )
    parameters = configuration.reward_candidates[candidate_index]
    standstill = approximate_episode_return(
        parameters,
        completed=False,
        on_time=False,
        progress_fraction=0,
        travel_time_s=180,
        energy_kwh=0.0404,
        integrated_speed_violation_m=0,
        had_speed_violation=False,
    )
    efficient = approximate_episode_return(
        parameters,
        completed=True,
        on_time=True,
        progress_fraction=1,
        travel_time_s=115,
        energy_kwh=0.23,
        integrated_speed_violation_m=0,
        had_speed_violation=False,
    )
    inefficient = approximate_episode_return(
        parameters,
        completed=True,
        on_time=True,
        progress_fraction=1,
        travel_time_s=115,
        energy_kwh=0.30,
        integrated_speed_violation_m=0,
        had_speed_violation=False,
    )
    violating = approximate_episode_return(
        parameters,
        completed=True,
        on_time=True,
        progress_fraction=1,
        travel_time_s=90,
        energy_kwh=0.28,
        integrated_speed_violation_m=5,
        had_speed_violation=True,
    )
    assert efficient > inefficient > standstill
    assert efficient > violating


def _wrapped_environment(parameters, *, max_episode_steps):
    return ScalarBenchmarkRewardV2(
        gym.make("DeterministicTrack-v1", max_episode_steps=max_episode_steps),
        parameters=parameters,
        max_time_s=140,
        max_speed_violation_m_s=0,
        energy_normalization_kwh=0.25,
        speed_violation_normalization_m=1,
    )


def test_v2_terminal_completion_bonus_is_reported_in_route_component():
    parameters = load_v2_configuration().reward_candidates[0]
    environment = _wrapped_environment(parameters, max_episode_steps=2)
    try:
        environment.reset(seed=1)
        environment.unwrapped.state = VehicleState(position_m=999.9, velocity_m_s=10)
        _, reward, terminated, _, info = environment.step([0])
        assert terminated
        components = info["benchmark_reward_components"]
        assert components["route_achievement"] > parameters.on_time_completion_bonus
        assert reward == pytest.approx(sum(components.values()))
        assert info["benchmark_reward_formula_version"] == "scalar-v2"
    finally:
        environment.close()


def test_v2_terminal_speed_event_penalty_occurs_once_at_episode_end():
    parameters = load_v2_configuration().reward_candidates[0]
    environment = _wrapped_environment(parameters, max_episode_steps=1)
    try:
        environment.reset(seed=1)
        environment.unwrapped.state = VehicleState(velocity_m_s=30)
        _, _, _, truncated, info = environment.step([0])
        assert truncated
        dense_penalty = (
            -parameters.speed_integral_weight
            * info["speed_excess_m_s"]
            * info["elapsed_time_s"]
        )
        assert info["benchmark_reward_components"]["speed_violation"] == (
            pytest.approx(dense_penalty - parameters.speed_violation_event_penalty)
        )
    finally:
        environment.close()


def test_v2_result_roundtrip(tmp_path):
    configuration = load_v2_configuration()
    task = configuration.task
    metrics = EpisodeMetrics(True, 100, 0.2, 0, 0, 0)
    episodes = (EpisodeEvaluation.from_metrics(3000, metrics, task),)
    result = V2BenchmarkRunResult(
        benchmark_name=configuration.name,
        configuration_sha256=v2_configuration_sha256(configuration),
        environment_id=configuration.environment_id,
        evaluation_split_id="validation-test",
        training_seed=11,
        training_steps=100,
        task=task,
        reward_parameters=configuration.reward_candidates[0],
        energy_normalization_kwh=configuration.energy_normalization_kwh,
        speed_violation_normalization_m=(
            configuration.speed_violation_normalization_m
        ),
        episodes=episodes,
        summary=EvaluationSummary.from_episodes(episodes, task),
    )
    path = save_v2_result(tmp_path / "result.json", result)
    assert load_v2_result(path) == result
