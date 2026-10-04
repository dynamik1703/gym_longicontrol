from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np
import pytest

import gym_longicontrol  # noqa: F401
from benchmarks.constrained_rl.config import load_configuration as load_v1
from benchmarks.constrained_rl.costs import objective_reward, speed_cost
from benchmarks.constrained_rl_v2.analysis import behavior_category
from benchmarks.constrained_rl_v2.config import load_configuration
from benchmarks.constrained_rl_v2.costs import (
    COST_NAMES,
    DenseDeadlineTaskWrapper,
    deadline_deficit_cost,
    deadline_state,
    discounted_elapsed_cost,
    optimistic_remaining_time_s,
)
from benchmarks.scalar_sac.evaluation import EpisodeEvaluation
from gym_longicontrol.domain.task import TaskSpecification
from gym_longicontrol.domain.track import Track


@pytest.fixture
def simple_track():
    return Track(
        positions_m=np.array([0.0, 100.0]),
        limits_m_s=np.array([10.0, 20.0]),
    )


def test_raw_time_discount_mapping_matches_preregistered_values():
    expected = {
        100.0: 9.999568287526,
        120.0: 9.999942159303,
        140.0: 9.999992250522,
        160.0: 9.999998961727,
        180.0: 9.999999860893,
    }
    actual = {
        duration: discounted_elapsed_cost(duration, 0.1, 0.99)
        for duration in expected
    }
    assert actual == pytest.approx(expected, abs=5e-13)
    assert actual[160.0] - actual[140.0] < 7e-6
    assert actual[180.0] - actual[160.0] < 1e-6


@pytest.mark.parametrize(
    ("position_m", "expected_s"),
    ((0.0, 15.0), (50.0, 10.0), (100.0, 5.0), (150.0, 2.5), (200.0, 0.0)),
)
def test_optimistic_remaining_time_integrates_speed_limit_segments(
    simple_track, position_m, expected_s
):
    assert optimistic_remaining_time_s(
        simple_track, position_m=position_m, track_length_m=200.0
    ) == pytest.approx(expected_s)


def test_deadline_state_before_deadline_and_on_exact_boundary(simple_track):
    remaining, slack, deficit = deadline_state(
        elapsed_time_s=125.0,
        position_m=0.0,
        track=simple_track,
        track_length_m=200.0,
        deadline_s=140.0,
    )
    assert (remaining, slack, deficit) == pytest.approx((15.0, 0.0, 0.0))


def test_deadline_state_detects_incomplete_and_late_completion(simple_track):
    incomplete = deadline_state(
        elapsed_time_s=180.0,
        position_m=0.0,
        track=simple_track,
        track_length_m=200.0,
        deadline_s=140.0,
    )
    late_completion = deadline_state(
        elapsed_time_s=141.0,
        position_m=200.0,
        track=simple_track,
        track_length_m=200.0,
        deadline_s=140.0,
    )
    assert incomplete == pytest.approx((15.0, -55.0, 55.0))
    assert late_completion == pytest.approx((0.0, -1.0, 1.0))


def test_completion_before_deadline_has_no_deadline_deficit(simple_track):
    assert deadline_state(
        elapsed_time_s=139.0,
        position_m=200.0,
        track=simple_track,
        track_length_m=200.0,
        deadline_s=140.0,
    ) == pytest.approx((0.0, 1.0, 0.0))


def test_deadline_cost_has_seconds_units_and_accumulates_right_endpoints():
    deficits = (0.0, 1.0, 2.0, 3.0)
    costs = [
        deadline_deficit_cost(deficit_s=value, dt_s=0.1, normalization_s=2.0)
        for value in deficits
    ]
    assert costs == pytest.approx((0.0, 0.05, 0.1, 0.15))
    assert sum(costs) == pytest.approx(0.3)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"duration_s": -1.0, "dt_s": 0.1, "gamma": 0.99},
        {"duration_s": 1.05, "dt_s": 0.1, "gamma": 0.99},
        {"duration_s": 1.0, "dt_s": 0.0, "gamma": 0.99},
        {"duration_s": 1.0, "dt_s": 0.1, "gamma": 1.1},
    ),
)
def test_discount_mapping_rejects_invalid_physical_inputs(kwargs):
    with pytest.raises(ValueError):
        discounted_elapsed_cost(**kwargs)


def test_v2_keeps_v1_objective_speed_cost_and_algorithm_exactly():
    v1 = load_v1()
    v2 = load_configuration()
    assert asdict(v2.objective) == asdict(v1.objective)
    assert asdict(v2.algorithm) == asdict(v1.algorithm)
    assert objective_reward(0.01, v2.objective.energy_scale_kwh) == pytest.approx(
        -0.04
    )
    assert speed_cost(2.0, 0.1) == pytest.approx(0.2)
    assert COST_NAMES == ("speed_integral_m", "deadline_deficit_integral_s")
    assert v2.constraints.cost_limits == (0.0, 0.0)


def _wrapped(configuration, *, max_steps=None):
    base = gym.make(
        configuration.environment_id,
        max_episode_steps=configuration.max_episode_steps,
    )
    return DenseDeadlineTaskWrapper(
        base,
        task=configuration.task,
        energy_scale_kwh=configuration.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.deadline_cost.normalization_s,
        max_simulator_steps=max_steps,
    )


def test_wrapper_preserves_observation_action_and_physical_transition():
    configuration = load_configuration()
    raw = gym.make(
        configuration.environment_id,
        max_episode_steps=configuration.max_episode_steps,
    )
    wrapped = _wrapped(configuration)
    try:
        raw_observation, raw_info = raw.reset(seed=2000)
        wrapped_observation, wrapped_info = wrapped.reset(seed=2000)
        assert raw.observation_space == wrapped.observation_space
        assert raw.action_space == wrapped.action_space
        np.testing.assert_array_equal(raw_observation, wrapped_observation)
        action = np.array([0.25])
        raw_next, _raw_reward, raw_terminated, raw_truncated, raw_step = raw.step(
            action
        )
        next_observation, _objective, terminated, truncated, step = wrapped.step(action)
        np.testing.assert_array_equal(raw_next, next_observation)
        assert (terminated, truncated) == (raw_terminated, raw_truncated)
        for key in (
            "position_m",
            "velocity_m_s",
            "acceleration_m_s2",
            "jerk_m_s3",
            "elapsed_time_s",
            "step_energy_kwh",
        ):
            assert step[key] == raw_step[key]
        assert "constraint_costs" not in raw_info
        assert "constraint_costs" not in wrapped_info
    finally:
        raw.close()
        wrapped.close()


def test_wrapper_is_deterministic_and_has_no_terminal_failure_impulse():
    configuration = load_configuration()
    first = _wrapped(configuration, max_steps=4)
    second = _wrapped(configuration, max_steps=4)
    try:
        observations = []
        costs = []
        for environment in (first, second):
            observation, _ = environment.reset(seed=2000)
            rollout = [observation.copy()]
            episode_costs = []
            for _ in range(4):
                observation, _reward, _terminated, truncated, info = environment.step(
                    np.array([0.0])
                )
                rollout.append(observation.copy())
                episode_costs.append(np.asarray(info["cost"]).copy())
            assert truncated
            observations.append(rollout)
            costs.append(episode_costs)
        np.testing.assert_array_equal(observations[0], observations[1])
        np.testing.assert_array_equal(costs[0], costs[1])
        # Seed 2000 starts inside the optimistic envelope, so truncating this tiny
        # implementation rollout does not inject V1's binary terminal cost.
        assert costs[0][-1][1] == 0.0
    finally:
        first.close()
        second.close()


def test_wrapper_exposes_positive_dense_cost_for_an_impossible_deadline():
    configuration = load_configuration()
    task = TaskSpecification(max_time_s=1.0, max_speed_violation_m_s=0.0)
    base = gym.make(
        configuration.environment_id,
        max_episode_steps=configuration.max_episode_steps,
    )
    wrapped = DenseDeadlineTaskWrapper(
        base,
        task=task,
        energy_scale_kwh=configuration.objective.energy_scale_kwh,
        deadline_normalization_s=1.0,
        max_simulator_steps=2,
    )
    try:
        wrapped.reset(seed=2000)
        _obs, _reward, _terminated, _truncated, first = wrapped.step(np.array([0.0]))
        _obs, _reward, _terminated, truncated, second = wrapped.step(np.array([0.0]))
        assert first["deadline_deficit_cost_s"] > 0
        assert second["deadline_deficit_integral_s"] > first[
            "deadline_deficit_integral_s"
        ]
        assert truncated
        assert wrapped.last_completed_episode is not None
        assert wrapped.last_completed_episode.costs[1] == pytest.approx(
            second["deadline_deficit_integral_s"]
        )
    finally:
        wrapped.close()


def test_track_splits_are_disjoint_and_sealed_tracks_are_not_training_metadata():
    configuration = load_configuration()
    splits = configuration.track_splits
    used = set(splits.development_calibration) | set(splits.validation)
    assert not used & set(splits.paper_final_test_reserved)
    assert splits.development_calibration == tuple(range(2000, 2009))
    assert splits.validation == tuple(range(3000, 3009))
    assert splits.paper_final_test_reserved == tuple(range(4000, 4018))


def test_canonical_configuration_rejects_action_repeat_change():
    configuration = load_configuration()
    with pytest.raises(ValueError, match="one native episode|native 10-Hz"):
        replace(
            configuration,
            algorithm=replace(configuration.algorithm, action_repeat=2),
        )


def _episode(**overrides):
    values = {
        "evaluation_seed": 3000,
        "completed": False,
        "feasible": False,
        "travel_time_s": 180.0,
        "energy_kwh": 0.1,
        "speed_violation_count": 0,
        "max_speed_violation_m_s": 0.0,
        "integrated_speed_violation_m": 0.0,
        "final_position_m": 0.0,
    }
    values.update(overrides)
    return EpisodeEvaluation(**values)


@pytest.mark.parametrize(
    ("episode", "expected"),
    (
        (
            _episode(
                completed=True,
                feasible=True,
                travel_time_s=120.0,
                final_position_m=1000.0,
            ),
            "fully_feasible",
        ),
        (
            _episode(
                completed=True,
                travel_time_s=120.0,
                max_speed_violation_m_s=0.1,
                speed_violation_count=1,
                final_position_m=1000.0,
            ),
            "completed_with_speed_violation",
        ),
        (
            _episode(completed=True, travel_time_s=150.0, final_position_m=1000.0),
            "completed_too_slowly",
        ),
        (_episode(final_position_m=1.0), "standstill"),
        (_episode(final_position_m=1.0001), "partial_progress_or_crawling"),
    ),
)
def test_behavior_categories_are_exclusive_and_reproducible(episode, expected):
    assert behavior_category(episode, load_configuration().task) == expected
