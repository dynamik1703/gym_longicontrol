from __future__ import annotations

import json
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from benchmarks.binary_reward.config import (
    DEFAULT_CONFIG_PATH,
    configuration_sha256,
    load_configuration,
)
from benchmarks.binary_reward.results import BinaryRunResult, load_result, save_result
from benchmarks.binary_reward.reward import BinarySuccessReward, binary_success
from benchmarks.scalar_sac.evaluation import EpisodeEvaluation, EvaluationSummary
from benchmarks.scalar_sb3.config import load_configuration as load_scalar_sb3
from gym_longicontrol.domain.metrics import EpisodeMetrics


class ScriptedEnvironment(gym.Env):
    observation_space = spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float64)
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)

    def __init__(self, transitions):
        self.transitions = list(transitions)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return np.zeros(8), {}

    def step(self, action):
        del action
        metrics, terminated, truncated = self.transitions.pop(0)
        return (
            np.zeros(8),
            123.0,
            terminated,
            truncated,
            {"episode_metrics": asdict(metrics)},
        )


def test_configuration_copies_frozen_scalar_sb3_sac_exactly():
    configuration = load_configuration()
    scalar = load_scalar_sb3()
    assert asdict(configuration.sac) == asdict(scalar.sac)
    assert configuration.training_seeds == scalar.training_seeds
    assert configuration.learning_curve_steps == scalar.learning_curve_steps
    assert configuration.track_splits == scalar.track_splits
    assert configuration.task == scalar.task
    assert len(configuration_sha256(configuration)) == 64


def test_binary_success_is_the_authoritative_feasibility_decision():
    configuration = load_configuration()
    assert binary_success(
        EpisodeMetrics(True, 140.0, 0.2, 0, 0.0, 0.0), configuration.task
    )
    assert not binary_success(
        EpisodeMetrics(True, 140.1, 0.2, 0, 0.0, 0.0), configuration.task
    )
    assert not binary_success(
        EpisodeMetrics(True, 100.0, 0.2, 1, 0.01, 0.001), configuration.task
    )
    assert not binary_success(
        EpisodeMetrics(False, 100.0, 0.0, 0, 0.0, 0.0), configuration.task
    )


def test_reward_is_zero_intermediately_and_one_only_on_terminal_success():
    incomplete = EpisodeMetrics(False, 0.1, 0.0, 0, 0.0, 0.0)
    success = EpisodeMetrics(True, 100.0, 0.2, 0, 0.0, 0.0)
    environment = BinarySuccessReward(
        ScriptedEnvironment(((incomplete, False, False), (success, True, False))),
        task=load_configuration().task,
    )
    try:
        environment.reset()
        _, reward, terminated, truncated, info = environment.step(np.zeros(1))
        assert reward == 0.0
        assert not terminated and not truncated
        assert not info["binary_reward_outcome_known"]
        assert not info["binary_reward_success"]
        assert info["historical_reward"] == 123.0
        _, reward, terminated, truncated, info = environment.step(np.zeros(1))
        assert reward == 1.0
        assert terminated and not truncated
        assert info["binary_reward_outcome_known"]
        assert info["binary_reward_success"]
    finally:
        environment.close()


@pytest.mark.parametrize(
    "metrics",
    (
        EpisodeMetrics(False, 180.0, 0.0, 0, 0.0, 0.0),
        EpisodeMetrics(True, 141.0, 0.2, 0, 0.0, 0.0),
        EpisodeMetrics(True, 100.0, 0.2, 1, 0.01, 0.001),
    ),
)
def test_every_terminal_failure_receives_zero(metrics):
    environment = BinarySuccessReward(
        ScriptedEnvironment(((metrics, True, False),)),
        task=load_configuration().task,
    )
    try:
        environment.reset()
        _, reward, terminated, _, info = environment.step(np.zeros(1))
        assert terminated
        assert reward == 0.0
        assert not info["binary_reward_success"]
    finally:
        environment.close()


def test_truncated_episode_is_a_known_zero_reward_failure():
    metrics = EpisodeMetrics(False, 180.0, 0.0, 0, 0.0, 0.0)
    environment = BinarySuccessReward(
        ScriptedEnvironment(((metrics, False, True),)),
        task=load_configuration().task,
    )
    try:
        environment.reset()
        _, reward, terminated, truncated, info = environment.step(np.zeros(1))
        assert not terminated and truncated
        assert reward == 0.0
        assert info["binary_reward_outcome_known"]
        assert not info["binary_reward_success"]
    finally:
        environment.close()


def test_binary_wrapper_keeps_public_observation_shape():
    from benchmarks.binary_reward.experiment import _training_environment

    environment = _training_environment(load_configuration())
    try:
        observation, _ = environment.reset(seed=2)
        assert observation.shape == (8,)
        assert environment.observation_space.shape == (8,)
    finally:
        environment.close()


def test_binary_result_roundtrip_recomputes_summary(tmp_path):
    configuration = load_configuration()
    episode = EpisodeEvaluation.from_metrics(
        3000,
        EpisodeMetrics(True, 100.0, 0.2, 0, 0.0, 0.0),
        configuration.task,
    )
    result = BinaryRunResult(
        benchmark_name=configuration.name,
        configuration_sha256=configuration_sha256(configuration),
        environment_id=configuration.environment_id,
        evaluation_split_id="validation-test",
        library_version="test",
        training_seed=11,
        training_steps=50_000,
        training_wall_time_s=1.0,
        policy_updates=49_900,
        task=configuration.task,
        reward=configuration.reward,
        episodes=(episode,),
        summary=EvaluationSummary.from_episodes((episode,), configuration.task),
        diagnostics={"train/actor_loss": -0.5},
    )
    assert load_result(save_result(tmp_path / "result.json", result)) == result


def test_configuration_rejects_reserved_validation_overlap(tmp_path):
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["track_splits"]["validation"][0] = 4000
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_configuration(path)
