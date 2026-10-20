import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.contrastive_rl.goal_adapter import (  # noqa: E402
    project_outcome as crl_project_outcome,
)
from benchmarks.gcsl_canonical.diagnostics import (  # noqa: E402
    learning_diagnostics,
    sampled_tuple_diagnostics,
)
from benchmarks.gcsl_canonical.goal_adapter import (  # noqa: E402
    OutcomeScales,
    augment_policy_state,
    canonical_command,
    physical_outcome,
    project_outcome,
)
from benchmarks.gcsl_canonical.learner import (  # noqa: E402
    GCSLBatch,
    GCSLLearner,
)
from benchmarks.gcsl_canonical.policy import GCSLPolicy  # noqa: E402
from benchmarks.gcsl_canonical.replay import TrajectoryReplay  # noqa: E402
from gym_longicontrol.domain.task import TaskSpecification  # noqa: E402

TASK = TaskSpecification(140.0, 0.0)


def outcome(position, previous, elapsed, violation=0.0):
    return physical_outcome(
        position_m=position,
        previous_position_m=previous,
        elapsed_time_s=elapsed,
        max_speed_violation_m_s=violation,
    )


def append(
    replay,
    episode,
    step,
    position,
    previous,
    elapsed,
    *,
    reward=123.0,
    terminated=False,
    truncated=False,
    violation=0.0,
):
    replay.append(
        state=np.full(12, episode * 10 + step, dtype=np.float32),
        action=np.asarray([(episode * 10 + step) / 100], dtype=np.float32),
        source_outcome=outcome(
            previous,
            max(previous - 1, 0),
            max(elapsed - 0.1, 0),
        ),
        outcome=outcome(position, previous, elapsed, violation),
        historical_reward=reward,
        terminated=terminated,
        truncated=truncated,
        episode_id=episode,
        episode_step=step,
    )


def make_replay(reward_offset=0.0, *, capacity=8):
    replay = TrajectoryReplay(capacity)
    previous = 0.0
    for index, position in enumerate((100.0, 200.0, 300.0, 400.0)):
        append(
            replay,
            0,
            index,
            position,
            previous,
            index + 1.0,
            reward=reward_offset + index,
            truncated=index == 3,
        )
        previous = position
    return replay


def synthetic_batch(action=0.4):
    return GCSLBatch(
        states=np.zeros((8, 12), dtype=np.float32),
        actions=np.full((8, 1), action, dtype=np.float32),
        goals=np.ones((8, 3), dtype=np.float32),
        lags=np.ones(8, dtype=np.int64),
    )


def test_exact_12d_state_and_goal_projection_match_frozen_crl_mapping():
    observation = np.arange(8, dtype=np.float64)
    achieved = outcome(500.0, 499.0, 90.0, 3.7)
    state = augment_policy_state(observation, achieved, OutcomeScales())
    assert state.shape == (12,)
    np.testing.assert_array_equal(state[:8], observation)
    np.testing.assert_allclose(state[8:], [0.5, 0.499, 0.5, 0.1])

    rows = np.stack(
        (
            outcome(500, 499, 100),
            outcome(1000, 999, 141),
            outcome(1000, 999, 130, 0.1),
        )
    )
    terminated = np.asarray((False, True, True))
    projected = project_outcome(rows, terminated=terminated, task=TASK)
    np.testing.assert_array_equal(
        projected, crl_project_outcome(rows, terminated=terminated, task=TASK)
    )
    np.testing.assert_array_equal(projected[0], [0.5, 1.0, 1.0])
    np.testing.assert_array_equal(projected[1], [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(projected[2], [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(canonical_command(), [1.0, 1.0, 1.0])


def test_policy_count_bounds_stochasticity_and_deterministic_evaluation_action():
    learner = GCSLLearner(seed=2)
    assert learner.parameter_count == 270_338
    states = np.zeros((128, 12), dtype=np.float32)
    goals = np.ones((128, 3), dtype=np.float32)
    generator = torch.Generator().manual_seed(9)
    sampled = learner.sample_action(states, goals, generator=generator)
    deterministic = learner.deterministic_action(states, goals)
    assert np.all(np.abs(sampled) <= 1.0)
    assert np.all(np.abs(deterministic) <= 1.0)
    assert np.std(sampled) > 0.0
    assert not np.array_equal(sampled, deterministic)
    with torch.no_grad():
        mean, _ = learner.policy(torch.as_tensor(states), torch.as_tensor(goals))
    np.testing.assert_allclose(deterministic, torch.tanh(mean).numpy())


def test_transformed_gaussian_nll_is_exact_and_endpoints_are_finite():
    policy = GCSLPolicy(state_dim=2, goal_dim=1, width=8, depth=4)
    states = torch.tensor([[0.2, -0.4], [0.1, 0.7]], dtype=torch.float32)
    goals = torch.tensor([[0.3], [0.8]], dtype=torch.float32)
    actions = torch.tensor([[0.25], [-0.6]], dtype=torch.float32)
    mean, log_std = policy(states, goals)
    z = torch.atanh(actions)
    normal = torch.distributions.Normal(mean, torch.exp(log_std))
    expected = -(normal.log_prob(z) - torch.log1p(-actions.square())).sum(-1)
    torch.testing.assert_close(policy.nll(states, goals, actions), expected)

    endpoints = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    endpoint_nll = policy.nll(states, goals, endpoints)
    assert torch.isfinite(endpoint_nll).all()
    endpoint_nll.mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_learner_has_no_critic_reward_cost_or_bellman_training_path():
    first = make_replay(-1000.0).sample(
        32, gamma=0.99, rng=np.random.default_rng(4), task=TASK
    )
    second = make_replay(1000.0).sample(
        32, gamma=0.99, rng=np.random.default_rng(4), task=TASK
    )
    assert not np.array_equal(first.historical_rewards, second.historical_rewards)
    np.testing.assert_array_equal(first.batch.states, second.batch.states)
    np.testing.assert_array_equal(first.batch.actions, second.batch.actions)
    np.testing.assert_array_equal(first.batch.goals, second.batch.goals)
    learner = GCSLLearner(seed=4)
    assert set(learner.state_dict()) == {"policy", "optimizer"}
    assert not any(
        token in name.lower()
        for name in vars(learner)
        for token in ("critic", "reward", "cost", "bellman", "q_function")
    )
    loss_a, gradients_a = learner.loss_and_gradients(first.batch)
    loss_b, gradients_b = learner.loss_and_gradients(second.batch)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    for left, right in zip(gradients_a, gradients_b, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_replay_strict_future_same_episode_no_reset_crossing_and_final_action():
    replay = TrajectoryReplay(10)
    append(replay, 0, 0, 100, 0, 1)
    append(replay, 0, 1, 200, 100, 2, truncated=True)
    append(replay, 1, 0, 50, 0, 1)
    append(replay, 1, 1, 1000, 50, 100, terminated=True)
    sampled = replay.sample(1000, gamma=0.99, rng=np.random.default_rng(8), task=TASK)
    assert np.all(sampled.future_indices >= sampled.source_indices)
    np.testing.assert_array_equal(
        sampled.batch.lags,
        sampled.future_indices - sampled.source_indices + 1,
    )
    positions = np.mod(sampled.source_indices, replay.capacity)
    future_positions = np.mod(sampled.future_indices, replay.capacity)
    np.testing.assert_array_equal(
        replay.episode_ids[positions], replay.episode_ids[future_positions]
    )
    assert {1, 3} <= set(sampled.source_indices)
    assert np.all(sampled.future_outcomes[:, 2] > sampled.source_outcomes[:, 2])
    diagnostics = sampled_tuple_diagnostics(sampled)
    assert diagnostics["future_lag_seconds"]["minimum"] == pytest.approx(0.1)
    assert "projected_goal_action_ambiguity" in diagnostics
    assert "exact_state_goal_action_ambiguity" in diagnostics


def test_discounted_future_sampler_matches_frozen_distribution():
    maximum = 6
    gamma = 0.99
    count = 300_000
    lags = TrajectoryReplay._sample_lags(
        np.full(count, maximum), gamma=gamma, rng=np.random.default_rng(12)
    )
    observed = np.bincount(lags, minlength=maximum + 1)[1:] / count
    expected = np.asarray([gamma ** (lag - 1) for lag in range(1, maximum + 1)])
    expected /= expected.sum()
    np.testing.assert_allclose(observed, expected, atol=0.002)


def test_replay_overwrite_roundtrip_keeps_only_valid_links():
    replay = TrajectoryReplay(4)
    for episode in range(3):
        append(replay, episode, 0, 10, 0, 1)
        append(replay, episode, 1, 20, 10, 2, truncated=True)
    assert replay.first_sequence == 2
    report = replay.integrity_report()
    assert report["sequence_index_exact"] is True
    assert report["episode_boundaries_valid"] is True
    restored = TrajectoryReplay.from_state(replay.to_state())
    first = replay.sample(100, gamma=0.99, rng=np.random.default_rng(99), task=TASK)
    second = restored.sample(100, gamma=0.99, rng=np.random.default_rng(99), task=TASK)
    np.testing.assert_array_equal(first.source_indices, second.source_indices)
    np.testing.assert_array_equal(first.future_indices, second.future_indices)
    assert np.min(first.source_indices) >= replay.first_sequence


def test_diagnostics_use_private_rng_and_report_canonical_support():
    replay = make_replay()
    sampled = replay.sample(8, gamma=0.99, rng=np.random.default_rng(4), task=TASK)
    learner = GCSLLearner(seed=5)
    torch_before = torch.get_rng_state().clone()
    numpy_before = np.random.get_state()
    diagnostics = learning_diagnostics(learner, sampled)
    assert np.isfinite(diagnostics["transformed_action_nll"])
    assert torch.equal(torch_before, torch.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    summary = replay.support_summary(TASK)
    assert summary["maximum_replay_progress"] == pytest.approx(0.4)
    assert summary["unique_projected_goals"] == 4


def test_supervised_update_moves_action_toward_label_and_schedule_has_7250():
    learner = GCSLLearner(seed=3)
    for parameter in learner.policy.parameters():
        torch.nn.init.zeros_(parameter)
    batch = synthetic_batch(action=0.5)
    before = learner.deterministic_action(batch.states[:1], batch.goals[:1]).item()
    learner.update(batch)
    after = learner.deterministic_action(batch.states[:1], batch.goals[:1]).item()
    assert before == pytest.approx(0.0)
    assert after > before

    from benchmarks.gcsl_canonical.config import scheduled_update_transitions

    updates = scheduled_update_transitions()
    assert len(updates) == 7_250
    assert updates[0] == 10_040
    assert updates[-1] == 300_000
    assert json.dumps(updates[:3]) == "[10040, 10080, 10120]"
