import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.gcsl.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from benchmarks.gcsl.config import GCSLConfig  # noqa: E402
from benchmarks.gcsl.diagnostics import (  # noqa: E402
    DiagnosticsAccumulator,
    learning_diagnostics,
)
from benchmarks.gcsl.goal_adapter import physical_outcome  # noqa: E402
from benchmarks.gcsl.learner import GCSLBatch, GCSLLearner  # noqa: E402
from benchmarks.gcsl.policy import GCSLPolicy, parameter_count  # noqa: E402
from benchmarks.gcsl.replay import TrajectoryReplay  # noqa: E402
from gym_longicontrol.domain.task import TaskSpecification  # noqa: E402

TASK = TaskSpecification(140.0, 0.0)


def synthetic_batch(config=None, action=0.4):
    config = config or GCSLConfig()
    return GCSLBatch(
        states=np.zeros((8, config.state_dim), dtype=np.float32),
        actions=np.full((8, config.action_dim), action, dtype=np.float32),
        goals=np.ones((8, config.goal_dim), dtype=np.float32),
        lags=np.ones(8, dtype=np.int64),
    )


def runtime_state():
    return {
        "training_seed": 11,
        "transition_count": 0,
        "update_cycle_count": 0,
        "episode_id": 0,
        "episode_step": 0,
        "current_episode_ended": False,
        "observation": np.zeros(8),
        "current_outcome": np.zeros(4),
        "policy_action_rng_state": torch.Generator().get_state(),
        "replay_rng_state": np.random.default_rng(1).bit_generator.state,
        "track_rng_state": np.random.default_rng(2).bit_generator.state,
        "track_seeds_used": [],
        "diagnostics": DiagnosticsAccumulator().to_state(),
        "training_outcomes": [],
        "development_completed": [],
        "first_canonical_target_available_transition": None,
        "first_physical_canonical_success_transition": None,
    }


def test_policy_parameter_count_matches_frozen_architecture():
    learner = GCSLLearner(seed=1)
    assert learner.parameter_count == 270_338
    assert parameter_count(learner.policy) == 270_338


def test_continuous_actions_are_bounded_and_sampling_is_stochastic():
    learner = GCSLLearner(seed=2)
    states = np.zeros((128, 12), dtype=np.float32)
    goals = np.ones((128, 3), dtype=np.float32)
    generator = torch.Generator().manual_seed(9)
    sampled = learner.sample_action(states, goals, generator=generator)
    deterministic = learner.deterministic_action(states, goals)
    assert np.all(sampled >= -1.0) and np.all(sampled <= 1.0)
    assert np.all(deterministic >= -1.0) and np.all(deterministic <= 1.0)
    assert np.std(sampled) > 0
    assert not np.array_equal(sampled, deterministic)


def test_action_nll_has_expected_direction_and_finite_gradients():
    policy = GCSLPolicy(state_dim=2, goal_dim=1, width=8, depth=4)
    for parameter in policy.parameters():
        torch.nn.init.zeros_(parameter)
    states = torch.zeros((4, 2))
    goals = torch.zeros((4, 1))
    at_mode = policy.nll(states, goals, torch.zeros((4, 1))).mean()
    far = policy.nll(states, goals, torch.full((4, 1), 0.8)).mean()
    assert at_mode < far
    far.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_supervised_update_moves_deterministic_action_toward_label():
    learner = GCSLLearner(seed=3)
    for parameter in learner.policy.parameters():
        torch.nn.init.zeros_(parameter)
    batch = synthetic_batch(action=0.5)
    before = learner.deterministic_action(batch.states[:1], batch.goals[:1]).item()
    learner.update(batch)
    after = learner.deterministic_action(batch.states[:1], batch.goals[:1]).item()
    assert before == pytest.approx(0.0)
    assert after > before


def make_replay(reward_offset=0.0):
    replay = TrajectoryReplay(8)
    positions = (100.0, 200.0, 300.0, 400.0)
    previous = 0.0
    for index, position in enumerate(positions):
        source = physical_outcome(
            position_m=previous,
            previous_position_m=max(0.0, previous - 1.0),
            elapsed_time_s=float(index),
            max_speed_violation_m_s=0.0,
        )
        outcome = physical_outcome(
            position_m=position,
            previous_position_m=previous,
            elapsed_time_s=float(index + 1),
            max_speed_violation_m_s=0.0,
        )
        replay.append(
            state=np.full(12, index),
            action=np.asarray([index / 10]),
            source_outcome=source,
            outcome=outcome,
            historical_reward=reward_offset + index,
            terminated=False,
            truncated=index == len(positions) - 1,
            episode_id=0,
            episode_step=index,
        )
        previous = position
    return replay


def test_reward_perturbation_changes_neither_loss_nor_gradients():
    first = make_replay(-1000.0).sample(
        16, gamma=0.99, rng=np.random.default_rng(4), task=TASK
    )
    second = make_replay(1000.0).sample(
        16, gamma=0.99, rng=np.random.default_rng(4), task=TASK
    )
    assert not np.array_equal(first.historical_rewards, second.historical_rewards)
    np.testing.assert_array_equal(first.batch.states, second.batch.states)
    np.testing.assert_array_equal(first.batch.actions, second.batch.actions)
    np.testing.assert_array_equal(first.batch.goals, second.batch.goals)
    learner = GCSLLearner(seed=4)
    loss_a, gradients_a = learner.loss_and_gradients(first.batch)
    loss_b, gradients_b = learner.loss_and_gradients(second.batch)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    for left, right in zip(gradients_a, gradients_b, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_replay_sampling_rng_is_reproducible():
    replay = make_replay()
    first = replay.sample(32, gamma=0.99, rng=np.random.default_rng(44), task=TASK)
    second = replay.sample(32, gamma=0.99, rng=np.random.default_rng(44), task=TASK)
    np.testing.assert_array_equal(first.source_indices, second.source_indices)
    np.testing.assert_array_equal(first.future_indices, second.future_indices)


def test_diagnostics_do_not_perturb_external_training_rngs():
    replay = make_replay()
    sampled = replay.sample(8, gamma=0.99, rng=np.random.default_rng(4), task=TASK)
    learner = GCSLLearner(seed=5)
    policy_rng = torch.Generator().manual_seed(10)
    replay_rng = np.random.default_rng(11)
    policy_before = policy_rng.get_state().clone()
    replay_before = json.dumps(replay_rng.bit_generator.state, sort_keys=True)
    diagnostics = learning_diagnostics(learner, sampled)
    assert np.isfinite(diagnostics["action_nll"])
    assert torch.equal(policy_before, policy_rng.get_state())
    assert replay_before == json.dumps(replay_rng.bit_generator.state, sort_keys=True)


def test_checkpoint_round_trip_preserves_policy_optimizer_replay_and_output(tmp_path):
    learner = GCSLLearner(seed=6)
    learner.update(synthetic_batch())
    replay = make_replay()
    state = np.ones((1, 12), dtype=np.float32)
    goal = np.ones((1, 3), dtype=np.float32)
    expected = learner.deterministic_action(state, goal)
    path = tmp_path / "checkpoint.ckpt"
    save_checkpoint(
        path,
        learner_state=learner.state_dict(),
        replay=replay,
        environment={"position": 4},
        runtime=runtime_state(),
        configuration_sha256="1" * 64,
        scientific_source_sha256={"science": "2" * 64},
        execution_source_sha256={"execution": "3" * 64},
    )
    loaded = load_checkpoint(
        path,
        expected_configuration_sha256="1" * 64,
        expected_scientific_source_sha256={"science": "2" * 64},
        expected_execution_source_sha256={"execution": "3" * 64},
    )
    restored = GCSLLearner(seed=99)
    restored.load_state_dict(loaded.learner_state)
    np.testing.assert_array_equal(expected, restored.deterministic_action(state, goal))
    assert loaded.replay.size == replay.size
    assert loaded.environment == {"position": 4}


def test_checkpoint_rejects_provenance_change(tmp_path: Path):
    learner = GCSLLearner(seed=7)
    path = tmp_path / "checkpoint.ckpt"
    save_checkpoint(
        path,
        learner_state=learner.state_dict(),
        replay=TrajectoryReplay(4),
        environment=None,
        runtime=runtime_state(),
        configuration_sha256="1" * 64,
        scientific_source_sha256={},
        execution_source_sha256={},
    )
    with pytest.raises(RuntimeError, match="configuration"):
        load_checkpoint(
            path,
            expected_configuration_sha256="9" * 64,
            expected_scientific_source_sha256={},
            expected_execution_source_sha256={},
        )
