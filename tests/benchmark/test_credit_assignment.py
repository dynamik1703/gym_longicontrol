import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.credit_assignment.action_repeat import ActionRepeat
from benchmarks.credit_assignment.config import (
    DEFAULT_CONFIG_PATH,
    configuration_sha256,
    load_configuration,
)
from benchmarks.credit_assignment.results import (
    CreditAssignmentRunResult,
    load_result,
    save_result,
)
from benchmarks.scalar_sac.evaluation import (
    EpisodeEvaluation,
    EvaluationSummary,
    evaluate_policy,
)
from benchmarks.scalar_sac.v2_config import load_v2_configuration
from benchmarks.scalar_sac.v2_reward import ScalarBenchmarkRewardV2
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.state import VehicleState


def _reward_environment(*, max_episode_steps=100):
    configuration = load_configuration()
    return ScalarBenchmarkRewardV2(
        gym.make("DeterministicTrack-v1", max_episode_steps=max_episode_steps),
        parameters=configuration.reward,
        max_time_s=configuration.task.max_time_s,
        max_speed_violation_m_s=configuration.task.max_speed_violation_m_s,
        energy_normalization_kwh=configuration.energy_normalization_kwh,
        speed_violation_normalization_m=(
            configuration.speed_violation_normalization_m
        ),
    )


def test_protocol_freezes_matrix_reward_splits_and_simulator_budget():
    configuration = load_configuration()
    v2b = load_v2_configuration(
        DEFAULT_CONFIG_PATH.parent.parent / "scalar_sac" / "canonical_v2b.json"
    )

    assert configuration.reward == v2b.reward_candidates[0]
    assert configuration.training_seeds == (11, 29, 47)
    assert configuration.simulator_step_checkpoints == tuple(
        range(50_000, 300_001, 50_000)
    )
    assert configuration.track_splits.development_calibration == tuple(
        range(2000, 2009)
    )
    assert configuration.track_splits.validation == tuple(range(3000, 3009))
    assert configuration.track_splits.paper_final_test_reserved == tuple(
        range(4000, 4018)
    )
    assert [
        (item.condition_id, item.action_repeat, item.gamma)
        for item in configuration.conditions
    ] == [
        ("a-baseline", 1, 0.99),
        ("b-discount-only", 1, 0.999),
        ("c-horizon-only", 5, 0.99),
        ("d-horizon-discount", 5, 0.999),
    ]
    assert len(configuration_sha256(configuration)) == 64


def test_protocol_rejects_changes_to_preregistered_matrix(tmp_path):
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["conditions"][2]["action_repeat"] = 4
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix"):
        load_configuration(path)


def test_action_repeat_matches_five_native_v2_steps():
    repeated = ActionRepeat(_reward_environment(), 5, capture_trace=True)
    native = _reward_environment()
    try:
        repeated_observation, _ = repeated.reset(seed=7)
        native_observation, _ = native.reset(seed=7)
        assert np.allclose(repeated_observation, native_observation)
        observation, reward, terminated, truncated, info = repeated.step([0.5])
        native_reward = 0.0
        for _ in range(5):
            (
                native_observation,
                step_reward,
                native_terminated,
                native_truncated,
                native_info,
            ) = native.step([0.5])
            native_reward += step_reward
        assert not terminated and not truncated
        assert not native_terminated and not native_truncated
        assert np.allclose(observation, native_observation)
        assert reward == pytest.approx(native_reward)
        assert reward == pytest.approx(
            sum(info["benchmark_reward_components"].values())
        )
        assert info["episode_metrics"] == native_info["episode_metrics"]
        assert info["simulator_steps_this_decision"] == 5
        assert info["episode_simulator_steps"] == 5
        assert len(info["action_repeat_trace"]) == 5
        assert info["elapsed_time_s"] == pytest.approx(0.5)
    finally:
        repeated.close()
        native.close()


def test_action_repeat_propagates_natural_termination_immediately():
    environment = ActionRepeat(_reward_environment(), 5)
    try:
        environment.reset(seed=1)
        environment.unwrapped.state = VehicleState(
            position_m=999.9, velocity_m_s=10
        )
        _, _, terminated, truncated, info = environment.step([0.0])
        assert terminated and not truncated
        assert info["simulator_steps_this_decision"] == 1
        assert info["episode_metrics"]["completed"]
    finally:
        environment.close()


def test_action_repeat_hits_exact_global_simulator_budget():
    environment = ActionRepeat(
        gym.make("DeterministicTrack-v1", max_episode_steps=100),
        5,
        max_simulator_steps=7,
    )
    try:
        environment.reset(seed=1)
        _, _, terminated, truncated, first = environment.step([0.0])
        assert not terminated and not truncated
        assert first["simulator_steps_total"] == 5
        _, _, terminated, truncated, final = environment.step([0.0])
        assert not terminated and truncated
        assert final["simulator_budget_truncated"]
        assert final["simulator_steps_this_decision"] == 2
        assert final["simulator_steps_total"] == 7
        assert final["episode_metrics"]["travel_time_s"] == pytest.approx(0.7)
        with pytest.raises(RuntimeError, match="exhausted"):
            environment.step([0.0])
    finally:
        environment.close()


def test_repeated_evaluation_records_native_simulator_steps():
    configuration = load_configuration()
    environment = ActionRepeat(
        gym.make("DeterministicTrack-v1", max_episode_steps=7), 5
    )
    try:
        episodes, _summary = evaluate_policy(
            lambda _observation: np.array([0.0]),
            environment,
            task=configuration.task,
            evaluation_seeds=(3000,),
        )
        assert episodes[0].step_count == 7
        assert episodes[0].travel_time_s == pytest.approx(0.7)
        assert episodes[0].traction_energy_kwh > 0
    finally:
        environment.close()


def test_credit_assignment_result_roundtrip(tmp_path):
    configuration = load_configuration()
    condition = configuration.conditions[3]
    metrics = EpisodeMetrics(True, 100.0, 0.2, 0, 0.0, 0.0)
    episodes = (
        EpisodeEvaluation.from_metrics(3000, metrics, configuration.task),
    )
    result = CreditAssignmentRunResult(
        benchmark_name=configuration.name,
        configuration_sha256=configuration_sha256(configuration),
        environment_id=configuration.environment_id,
        evaluation_split_id="validation-test",
        condition_id=condition.condition_id,
        condition_label=condition.label,
        gamma=condition.gamma,
        action_repeat=condition.action_repeat,
        library_version="test",
        training_seed=11,
        simulator_step_target=50_000,
        simulator_steps=50_002,
        agent_decisions=10_011,
        gradient_updates=9_910,
        training_wall_time_s=1.5,
        task=configuration.task,
        reward_parameters=configuration.reward,
        episodes=episodes,
        summary=EvaluationSummary.from_episodes(episodes, configuration.task),
        diagnostics={"train/actor_loss": -0.25},
    )
    path = save_result(tmp_path / "result.json", result)
    assert load_result(path) == result


def test_sb3_builder_applies_condition_gamma_and_repeat():
    pytest.importorskip("stable_baselines3")
    from benchmarks.credit_assignment.experiment import (
        _model,
        _training_environment,
    )

    configuration = load_configuration()
    condition = configuration.conditions[3]
    environment = _training_environment(configuration, condition)
    try:
        model = _model(configuration, condition, environment, 11, "cpu")
        assert model.gamma == pytest.approx(0.999)
        assert environment.repeat == 5
        assert environment.max_simulator_steps == 300_000
        assert model.batch_size == 256
        assert model.ent_coef == "auto"
    finally:
        environment.close()


def test_committed_result_summary_matches_preregistered_gate():
    payload = json.loads(
        (Path(__file__).parents[2] / "benchmarks/credit_assignment/results.json")
        .read_text(encoding="utf-8")
    )
    configuration = load_configuration()
    assert payload["configuration_sha256"] == configuration_sha256(configuration)
    assert payload["result_file_count"] == 144
    assert not payload["reserved_final_test_evaluated"]
    assert payload["decision_gate"]["case"] == "E"
    assert payload["decision_gate"]["selected_condition_id"] == "c-horizon-only"
    assert payload["interaction_accounting"][
        "simulator_step_budget_per_condition_and_seed"
    ] == 300_000
