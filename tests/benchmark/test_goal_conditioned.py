from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

from benchmarks.goal_conditioned.config import (
    configuration_sha256,
    load_configuration,
)
from benchmarks.goal_conditioned.environment import GoalConditionedTask
from benchmarks.goal_conditioned.experiment import (
    build_model,
    make_environment,
    sac_model_kwargs,
)
from benchmarks.goal_conditioned.goal import (
    GoalScales,
    canonical_desired_goal,
    desired_goal,
    encode_achieved_goal,
    goal_success,
    goal_transition_reward,
    relabel_desired_goal,
    relabeled_terminal_mask,
)
from benchmarks.goal_conditioned.replay_buffer import GoalReplayBuffer
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.envs.longicontrol import DeterministicTrack


def _achieved(position, previous, time_s, violation, scales=GoalScales()):
    return encode_achieved_goal(
        position_m=position,
        previous_position_m=previous,
        elapsed_time_s=time_s,
        max_speed_violation_m_s=violation,
        scales=scales,
    )


@pytest.mark.parametrize(
    ("completed", "position", "previous", "time_s", "violation"),
    [
        (True, 1000.0, 999.0, 140.0, 0.0),
        (True, 1000.0, 999.0, 140.1, 0.0),
        (True, 1000.0, 999.0, 100.0, 0.01),
        (False, 900.0, 899.0, 100.0, 0.0),
    ],
)
def test_canonical_success_matches_external_evaluator(
    completed, position, previous, time_s, violation
):
    task = TaskSpecification(140.0, 0.0)
    scales = GoalScales()
    metrics = EpisodeMetrics(
        completed=completed,
        travel_time_s=time_s,
        energy_kwh=0.2,
        speed_violation_count=int(violation > 0),
        max_speed_violation_m_s=violation,
        integrated_speed_violation_m=violation * 0.1,
    )
    success = goal_success(
        _achieved(position, previous, time_s, violation),
        canonical_desired_goal(task, scales),
    )
    assert success is is_feasible(metrics, task)


def test_prior_overspeed_survives_slowing_and_invalidates_prefix():
    goal = desired_goal(
        target_position_m=500.0,
        max_time_s=140.0,
        max_speed_violation_m_s=0.0,
        scales=GoalScales(),
    )
    after_slowing = _achieved(500.0, 499.0, 80.0, 0.25)
    assert goal_transition_reward(after_slowing, goal) == 0.0


def test_environment_keeps_prior_overspeed_after_current_excess_returns_to_zero():
    base = TimeLimit(
        DeterministicTrack(speed_limits=(7, 7, 7, 7)), max_episode_steps=1800
    )
    environment = GoalConditionedTask(
        base,
        task=TaskSpecification(140.0, 0.0),
        scales=GoalScales(),
    )
    try:
        environment.reset(seed=2)
        for _ in range(200):
            observation, _, _, _, info = environment.step([1.0])
            if info["speed_excess_m_s"] > 0:
                break
        else:
            raise AssertionError("Synthetic low-limit track never produced overspeed")
        recorded_maximum = observation["achieved_goal"][3]
        assert recorded_maximum > 0

        for _ in range(100):
            observation, _, _, _, info = environment.step([-1.0])
            if info["speed_excess_m_s"] == 0:
                break
        assert info["speed_excess_m_s"] == 0
        assert observation["achieved_goal"][3] >= recorded_maximum
    finally:
        environment.close()


def test_first_arrival_is_rewarded_once_not_repeated():
    goal = desired_goal(
        target_position_m=500.0,
        max_time_s=140.0,
        max_speed_violation_m_s=0.0,
        scales=GoalScales(),
    )
    arrival = _achieved(500.0, 499.0, 80.0, 0.0)
    later = _achieved(600.0, 500.0, 90.0, 0.0)
    assert goal_transition_reward(arrival, goal) == 1.0
    assert goal_transition_reward(later, goal) == 0.0


def test_reward_scalar_and_batch_agree():
    task = TaskSpecification(140.0, 0.0)
    goal = canonical_desired_goal(task, GoalScales())
    achieved = np.stack(
        [
            _achieved(1000.0, 999.0, 120.0, 0.0),
            _achieved(900.0, 899.0, 120.0, 0.0),
            _achieved(1000.0, 999.0, 141.0, 0.0),
        ]
    )
    batch = goal_transition_reward(achieved, goal)
    scalar = np.array(
        [goal_transition_reward(row, goal) for row in achieved], dtype=np.float32
    )
    np.testing.assert_array_equal(batch, scalar)
    np.testing.assert_array_equal(batch, [1.0, 0.0, 0.0])


def test_relabel_changes_position_only_and_keeps_constraints():
    task = TaskSpecification(140.0, 0.0)
    original = canonical_desired_goal(task, GoalScales())
    future = _achieved(450.0, 440.0, 70.0, 0.3)
    relabeled = relabel_desired_goal(original, future)
    np.testing.assert_allclose(relabeled[:2], [0.45, 0.45])
    np.testing.assert_array_equal(relabeled[2:], original[2:])
    assert goal_transition_reward(_achieved(450.0, 440.0, 70.0, 0.3), relabeled) == 0
    assert goal_transition_reward(_achieved(450.0, 440.0, 70.0, 0.0), relabeled) == 1


def test_terminal_mask_ends_on_relabelled_arrival_and_finite_timeout():
    goal = desired_goal(
        target_position_m=500.0,
        max_time_s=140.0,
        max_speed_violation_m_s=0.0,
        scales=GoalScales(),
    )
    arrived = _achieved(500.0, 499.0, 80.0, 0.0)
    not_arrived = _achieved(400.0, 399.0, 180.0, 0.0)
    assert relabeled_terminal_mask(False, arrived, goal) is True
    assert relabeled_terminal_mask(True, not_arrived, goal) is True
    assert relabeled_terminal_mask(False, not_arrived, goal) is False


def test_configuration_freezes_matched_conditions_and_protected_splits():
    configuration = load_configuration()
    assert configuration_sha256(configuration) == (
        "87fee0a3f74b9f8365b9f4c5004e3b7bf880a3739d890462780fc1d70294f6ec"
    )
    assert tuple(item.condition_id for item in configuration.conditions) == (
        "sac-no-her",
        "sac-her",
    )
    assert configuration.sac.learning_starts == 1800
    assert configuration.validation_evaluation_steps == (300_000,)
    assert set(configuration.track_splits.validation).isdisjoint(
        configuration.track_splits.paper_final_test_reserved
    )
    with pytest.raises(ValueError, match="Validation"):
        replace(configuration, validation_evaluation_steps=(50_000, 300_000))


def test_only_replay_mechanism_differs_between_model_arguments():
    configuration = load_configuration()
    control = sac_model_kwargs(configuration, "sac-no-her")
    hindsight = sac_model_kwargs(configuration, "sac-her")
    shared_keys = set(control) - {"replay_buffer_class", "replay_buffer_kwargs"}
    assert shared_keys == set(hindsight) - {
        "replay_buffer_class",
        "replay_buffer_kwargs",
    }
    assert {key: control[key] for key in shared_keys} == {
        key: hindsight[key] for key in shared_keys
    }
    assert control["replay_buffer_class"] is GoalReplayBuffer
    assert hindsight["replay_buffer_class"] is GoalReplayBuffer
    assert control["replay_buffer_kwargs"] == {
        "handle_timeout_termination": False,
        "n_sampled_goal": 0,
        "goal_selection_strategy": "future",
        "copy_info_dict": False,
    }
    assert hindsight["replay_buffer_kwargs"]["handle_timeout_termination"] is False
    assert hindsight["replay_buffer_kwargs"]["n_sampled_goal"] == 4
    differing_replay_arguments = {
        key
        for key in control["replay_buffer_kwargs"]
        if control["replay_buffer_kwargs"][key]
        != hindsight["replay_buffer_kwargs"][key]
    }
    assert differing_replay_arguments == {"n_sampled_goal"}


@pytest.mark.parametrize(
    ("condition_id", "replay_buffer_class"),
    [
        ("sac-no-her", GoalReplayBuffer),
        ("sac-her", GoalReplayBuffer),
    ],
)
def test_both_preregistered_models_build_without_learning(
    condition_id, replay_buffer_class
):
    configuration = load_configuration()
    configuration = replace(
        configuration,
        sac=replace(configuration.sac, buffer_size=2048, batch_size=32),
    )
    environment = make_environment(configuration)
    try:
        model = build_model(
            configuration, condition_id, environment, seed=11, device="cpu"
        )
        assert isinstance(model.replay_buffer, replay_buffer_class)
        assert model.num_timesteps == 0
    finally:
        environment.close()


def test_goal_environment_resets_and_steps_deterministically_without_oracles():
    configuration = load_configuration()
    first = make_environment(configuration)
    second = make_environment(configuration)
    try:
        first_obs, first_info = first.reset(seed=2000)
        second_obs, second_info = second.reset(seed=2000)
        assert set(first_obs) == {"observation", "achieved_goal", "desired_goal"}
        assert first.observation_space.contains(first_obs)
        assert not any("track" in key or "oracle" in key for key in first_obs)
        assert not any("track" in key or "oracle" in key for key in first_info)
        for key in first_obs:
            np.testing.assert_array_equal(first_obs[key], second_obs[key])
        np.testing.assert_array_equal(
            first_obs["desired_goal"],
            canonical_desired_goal(configuration.task, configuration.goal_scales),
        )
        for action in ([0.2], [0.5], [-0.3], [0.1]):
            first_step = first.step(action)
            second_step = second.step(action)
            for key in first_step[0]:
                np.testing.assert_array_equal(first_step[0][key], second_step[0][key])
            assert first_step[1:4] == second_step[1:4]
    finally:
        first.close()
        second.close()


class _NoLiveRewardEnvironment:
    def env_method(self, *args, **kwargs):
        raise AssertionError("Virtual reward must not query a live environment")


def _replay_buffer(*, n_sampled_goal=4):
    observation_space = spaces.Dict(
        {
            "observation": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float64),
            "achieved_goal": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float64),
            "desired_goal": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float64),
        }
    )
    return GoalReplayBuffer(
        buffer_size=16,
        observation_space=observation_space,
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64),
        env=_NoLiveRewardEnvironment(),
        device="cpu",
        n_envs=1,
        n_sampled_goal=n_sampled_goal,
        handle_timeout_termination=False,
    )


def _add_transition(buffer, achieved, next_achieved, *, done, timeout=False):
    canonical = canonical_desired_goal(TaskSpecification(140.0, 0.0), GoalScales())
    obs = {
        "observation": np.array([[0.0, 0.0]]),
        "achieved_goal": np.array([achieved]),
        "desired_goal": np.array([canonical]),
    }
    next_obs = {
        "observation": np.array([[0.1, 0.0]]),
        "achieved_goal": np.array([next_achieved]),
        "desired_goal": np.array([canonical]),
    }
    buffer.add(
        obs,
        next_obs,
        np.array([[0.0]]),
        np.array([0.0]),
        np.array([done]),
        [{"TimeLimit.truncated": timeout}],
    )


def test_virtual_reward_uses_stored_goals_and_recomputes_done():
    buffer = _replay_buffer()
    start = _achieved(0.0, 0.0, 0.0, 0.0)
    arrival = _achieved(500.0, 0.0, 80.0, 0.0)
    timeout = _achieved(800.0, 500.0, 180.0, 0.0)
    _add_transition(buffer, start, arrival, done=False)
    _add_transition(buffer, arrival, timeout, done=True, timeout=True)

    target = desired_goal(
        target_position_m=500.0,
        max_time_s=140.0,
        max_speed_violation_m_s=0.0,
        scales=GoalScales(),
    )
    buffer._sample_goals = lambda batch, env: np.array([target])
    virtual = buffer._get_virtual_samples(np.array([0]), np.array([0]))
    assert virtual.rewards.cpu().numpy().item() == 1.0
    assert virtual.dones.cpu().numpy().item() == 1.0

    canonical = canonical_desired_goal(TaskSpecification(140.0, 0.0), GoalScales())
    buffer._sample_goals = lambda batch, env: np.array([canonical])
    timeout_sample = buffer._get_virtual_samples(np.array([1]), np.array([0]))
    assert timeout_sample.rewards.cpu().numpy().item() == 0.0
    assert timeout_sample.dones.cpu().numpy().item() == 1.0
    assert buffer.handle_timeout_termination is False
    assert buffer.timeouts[1, 0] == 0.0


def test_goal_sampling_is_deterministic_and_rejects_already_reached_prefix():
    buffer = _replay_buffer()
    zero = _achieved(0.0, 0.0, 0.0, 0.0)
    at_200 = _achieved(200.0, 0.0, 20.0, 0.0)
    at_400 = _achieved(400.0, 200.0, 40.0, 0.0)
    _add_transition(buffer, zero, at_200, done=False)
    _add_transition(buffer, at_200, at_400, done=True)
    np.random.seed(123)
    first = buffer._sample_goals(np.array([0]), np.array([0]))
    np.random.seed(123)
    second = buffer._sample_goals(np.array([0]), np.array([0]))
    np.testing.assert_array_equal(first, second)
    assert first[0, 0] > zero[0]
    sample = buffer.sample(8)
    assert np.isfinite(sample.rewards.cpu().numpy()).all()
    assert np.isfinite(sample.dones.cpu().numpy()).all()

    stationary = _replay_buffer()
    _add_transition(stationary, zero, zero, done=True)
    sampled = stationary._sample_goals(np.array([0]), np.array([0]))
    canonical = canonical_desired_goal(TaskSpecification(140.0, 0.0), GoalScales())
    np.testing.assert_array_equal(sampled[0], canonical)


def test_replay_requires_completed_episode_before_sampling():
    buffer = _replay_buffer()
    zero = _achieved(0.0, 0.0, 0.0, 0.0)
    at_100 = _achieved(100.0, 0.0, 10.0, 0.0)
    _add_transition(buffer, zero, at_100, done=False)
    with pytest.raises(RuntimeError, match="end of the first episode"):
        buffer.sample(1)


def test_no_her_control_uses_same_completed_episode_replay_without_virtual_goals():
    buffer = _replay_buffer(n_sampled_goal=0)
    zero = _achieved(0.0, 0.0, 0.0, 0.0)
    at_100 = _achieved(100.0, 0.0, 10.0, 0.0)
    _add_transition(buffer, zero, at_100, done=True)
    sample = buffer.sample(8)
    assert buffer.her_ratio == 0.0
    assert sample.rewards.shape == (8, 1)
    assert np.isfinite(sample.rewards.cpu().numpy()).all()
