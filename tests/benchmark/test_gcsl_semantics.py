import numpy as np
import pytest

from benchmarks.contrastive_rl.goal_adapter import project_outcome as crl_project
from benchmarks.gcsl.config import GCSLConfig
from benchmarks.gcsl.diagnostics import sampled_tuple_diagnostics
from benchmarks.gcsl.goal_adapter import (
    canonical_command,
    canonical_goal_set_membership,
    physical_outcome,
    project_outcome,
    projected_goal_is_canonical,
)
from benchmarks.gcsl.replay import TrajectoryReplay
from gym_longicontrol.domain.task import TaskSpecification

TASK = TaskSpecification(140.0, 0.0)


def outcome(position, previous, elapsed, violation=0.0):
    return physical_outcome(
        position_m=position,
        previous_position_m=previous,
        elapsed_time_s=elapsed,
        max_speed_violation_m_s=violation,
    )


def append(replay, episode, step, position, previous, elapsed, **flags):
    replay.append(
        state=np.full(12, episode * 10 + step),
        action=np.asarray([(episode * 10 + step) / 100]),
        source_outcome=outcome(previous, max(previous - 1, 0), max(elapsed - 0.1, 0)),
        outcome=outcome(position, previous, elapsed, flags.pop("violation", 0.0)),
        historical_reward=123.0,
        terminated=flags.pop("terminated", False),
        truncated=flags.pop("truncated", False),
        episode_id=episode,
        episode_step=step,
    )


def test_tuple_indexing_uses_source_action_and_strict_future():
    replay = TrajectoryReplay(10)
    append(replay, 0, 0, 100, 0, 1)
    append(replay, 0, 1, 200, 100, 2)
    append(replay, 0, 2, 300, 200, 3, truncated=True)
    sampled = replay.sample(100, gamma=0.99, rng=np.random.default_rng(2), task=TASK)
    assert np.all(sampled.future_indices >= sampled.source_indices)
    np.testing.assert_array_equal(
        sampled.batch.lags, sampled.future_indices - sampled.source_indices + 1
    )
    assert np.all(sampled.future_outcomes[:, 2] > sampled.source_outcomes[:, 2])
    np.testing.assert_allclose(
        sampled.batch.actions[:, 0], sampled.source_indices / 100
    )
    np.testing.assert_array_equal(sampled.episode_ids, 0)


def test_one_step_future_and_final_preterminal_action_are_supervised():
    replay = TrajectoryReplay(2)
    append(replay, 0, 0, 1000, 0, 100, terminated=True)
    sampled = replay.sample(8, gamma=0.99, rng=np.random.default_rng(9), task=TASK)
    np.testing.assert_array_equal(sampled.source_indices, 0)
    np.testing.assert_array_equal(sampled.future_indices, 0)
    np.testing.assert_array_equal(sampled.batch.lags, 1)
    assert np.all(sampled.future_outcomes[:, 2] > sampled.source_outcomes[:, 2])
    assert np.all(sampled.batch.goals == canonical_command())
    diagnostics = sampled_tuple_diagnostics(sampled)
    assert diagnostics["physical_future_lag_seconds"]["mean"] == pytest.approx(0.1)
    assert "exact_state_goal_action_ambiguity" in diagnostics


def test_future_never_crosses_reset_and_terminal_is_not_a_source():
    replay = TrajectoryReplay(10)
    append(replay, 0, 0, 100, 0, 1)
    append(replay, 0, 1, 200, 100, 2, truncated=True)
    append(replay, 1, 0, 50, 0, 1)
    append(replay, 1, 1, 60, 50, 2, truncated=True)
    sampled = replay.sample(200, gamma=0.99, rng=np.random.default_rng(8), task=TASK)
    ids = replay.episode_ids
    np.testing.assert_array_equal(
        ids[sampled.source_indices], ids[sampled.future_indices]
    )
    assert set(sampled.source_indices) <= {0, 1, 2, 3}


def test_post_terminal_source_row_and_discontinuous_step_are_rejected():
    replay = TrajectoryReplay(5)
    append(replay, 0, 0, 100, 0, 1, truncated=True)
    with pytest.raises(ValueError, match="post-terminal"):
        append(replay, 0, 1, 200, 100, 2)
    replay = TrajectoryReplay(5)
    append(replay, 0, 0, 100, 0, 1)
    with pytest.raises(ValueError, match="uninterrupted"):
        append(replay, 0, 2, 200, 100, 2)


def test_safe_prefix_later_violation_and_braking_preserve_history():
    safe_prefix = outcome(500, 499, 60, 0.0)
    unsafe_after_braking = outcome(700, 699, 100, 0.01)
    np.testing.assert_array_equal(
        project_outcome(safe_prefix, terminated=False, task=TASK), [0.5, 1, 1]
    )
    np.testing.assert_array_equal(
        project_outcome(unsafe_after_braking, terminated=False, task=TASK),
        [0.7, 1, 0],
    )


def test_late_and_unsafe_futures_remain_honest():
    late = outcome(1000, 999, 140.0001)
    unsafe = outcome(1000, 999, 130, 0.0001)
    np.testing.assert_array_equal(
        project_outcome(late, terminated=True, task=TASK), [1, 0, 1]
    )
    np.testing.assert_array_equal(
        project_outcome(unsafe, terminated=True, task=TASK), [1, 1, 0]
    )


def test_canonical_target_requires_true_feasible_first_arrival():
    feasible = outcome(1000, 999, 140, 0)
    post_terminal = outcome(1001, 1000, 139, 0)
    assert canonical_goal_set_membership(feasible, task=TASK)
    assert projected_goal_is_canonical(
        project_outcome(feasible, terminated=True, task=TASK)
    )
    assert not canonical_goal_set_membership(post_terminal, task=TASK)
    with pytest.raises(ValueError, match="post-terminal"):
        project_outcome(post_terminal, terminated=True, task=TASK)


def test_canonical_command_exists_without_observed_success_or_insertion():
    np.testing.assert_array_equal(canonical_command(), [1, 1, 1])
    replay = TrajectoryReplay(4)
    append(replay, 0, 0, 999, 0, 140)
    append(replay, 0, 1, 1000, 999, 141, terminated=True)
    sampled = replay.sample(64, gamma=0.99, rng=np.random.default_rng(2), task=TASK)
    assert not np.all(sampled.batch.goals == 1.0, axis=1).any()


def test_gcsl_projection_is_the_frozen_crl_mapping():
    rows = np.stack(
        [outcome(500, 499, 100), outcome(1000, 999, 141), outcome(1000, 999, 130, 0.1)]
    )
    terminated = np.asarray([False, True, True])
    np.testing.assert_array_equal(
        project_outcome(rows, terminated=terminated, task=TASK),
        crl_project(rows, terminated=terminated, task=TASK),
    )


def test_future_lag_absolute_time_and_deadline_are_distinct():
    assert GCSLConfig().horizon_conditioning is False
    short_but_late = outcome(1000, 999, 141)
    long_but_timely = outcome(900, 899, 120)
    np.testing.assert_array_equal(
        project_outcome(short_but_late, terminated=True, task=TASK), [1, 0, 1]
    )
    np.testing.assert_array_equal(
        project_outcome(long_but_timely, terminated=False, task=TASK), [0.9, 1, 1]
    )
    assert 1 != 100
