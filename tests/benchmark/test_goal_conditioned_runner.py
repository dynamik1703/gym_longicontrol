from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import SAC

from benchmarks.goal_conditioned.config import load_configuration
from benchmarks.goal_conditioned.diagnostics import DiagnosticGoalReplayBuffer
from benchmarks.goal_conditioned.experiment import make_environment
from benchmarks.goal_conditioned.goal import (
    GoalScales,
    canonical_desired_goal,
    encode_achieved_goal,
)
from benchmarks.goal_conditioned.runner import (
    SCIENTIFIC_SHA256,
    DiagnosticWriter,
    GoalStudyCallback,
    _build_diagnostic_model,
    _capture_rng_state,
    _evaluate_without_rng_perturbation,
    _model_sha256,
    _new_manifest,
    _terminal_position_is_consistent,
    initialize_study_root,
    interrupted_attempt_provenance,
    validate_validation_gate,
)
from gym_longicontrol.domain.task import TaskSpecification


class _NoLiveRewardEnvironment:
    def env_method(self, *args, **kwargs):
        raise AssertionError("Diagnostics must not query a live environment")


def _achieved(position, previous, time_s, violation):
    return encode_achieved_goal(
        position_m=position,
        previous_position_m=previous,
        elapsed_time_s=time_s,
        max_speed_violation_m_s=violation,
        scales=GoalScales(),
    )


def _diagnostic_buffer(*, enabled=True, n_sampled_goal=4):
    observation_space = spaces.Dict(
        {
            "observation": spaces.Box(
                -1.0, 1.0, shape=(2,), dtype=np.float64
            ),
            "achieved_goal": spaces.Box(
                0.0, 1.0, shape=(4,), dtype=np.float64
            ),
            "desired_goal": spaces.Box(
                0.0, 1.0, shape=(4,), dtype=np.float64
            ),
        }
    )
    return DiagnosticGoalReplayBuffer(
        buffer_size=32,
        observation_space=observation_space,
        action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64),
        env=_NoLiveRewardEnvironment(),
        device="cpu",
        n_envs=1,
        n_sampled_goal=n_sampled_goal,
        handle_timeout_termination=False,
        diagnostics_enabled=enabled,
    )


def _add(buffer, achieved, next_achieved, *, done=False, reward=0.0):
    goal = canonical_desired_goal(TaskSpecification(140.0, 0.0), GoalScales())
    observation = {
        "observation": np.array([[0.0, 0.0]]),
        "achieved_goal": np.array([achieved]),
        "desired_goal": np.array([goal]),
    }
    next_observation = {
        "observation": np.array([[0.1, 0.0]]),
        "achieved_goal": np.array([next_achieved]),
        "desired_goal": np.array([goal]),
    }
    buffer.add(
        observation,
        next_observation,
        np.array([[0.0]]),
        np.array([reward]),
        np.array([done]),
        [{"TimeLimit.truncated": False}],
    )


def _populate(buffer):
    _add(buffer, _achieved(0, 0, 0, 0), _achieved(200, 0, 20, 0))
    _add(buffer, _achieved(200, 0, 20, 0), _achieved(400, 200, 40, 0))
    _add(
        buffer,
        _achieved(400, 200, 40, 0),
        _achieved(600, 400, 60, 0),
        done=True,
    )


def _sample_arrays(sample):
    return {
        **{
            f"observation:{key}": value.detach().cpu().numpy()
            for key, value in sample.observations.items()
        },
        **{
            f"next_observation:{key}": value.detach().cpu().numpy()
            for key, value in sample.next_observations.items()
        },
        "actions": sample.actions.detach().cpu().numpy(),
        "dones": sample.dones.detach().cpu().numpy(),
        "rewards": sample.rewards.detach().cpu().numpy(),
    }


def _assert_numpy_rng_equal(first, second):
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2:] == second[2:]


def test_diagnostics_do_not_change_samples_or_numpy_rng_state():
    observed = _diagnostic_buffer(enabled=True)
    silent = _diagnostic_buffer(enabled=False)
    _populate(observed)
    _populate(silent)

    np.random.seed(818)
    observed_sample = observed.sample(10)
    observed_state = np.random.get_state()
    np.random.seed(818)
    silent_sample = silent.sample(10)
    silent_state = np.random.get_state()

    for key, actual in _sample_arrays(observed_sample).items():
        np.testing.assert_array_equal(actual, _sample_arrays(silent_sample)[key])
    _assert_numpy_rng_equal(observed_state, silent_state)
    diagnostics = observed.replay_diagnostics.to_dict()
    assert diagnostics["total_sampled_rows"] == 10
    assert diagnostics["real_rows"] == 2
    assert diagnostics["virtual_rows"] == 8
    assert silent.replay_diagnostics.total_sampled_rows == 0


def test_no_her_has_identical_episode_replay_but_zero_virtual_rows():
    buffer = _diagnostic_buffer(n_sampled_goal=0)
    _populate(buffer)
    sample = buffer.sample(7)
    assert sample.rewards.shape == (7, 1)
    assert buffer.replay_diagnostics.real_rows == 7
    assert buffer.replay_diagnostics.virtual_rows == 0


class TinyGoalEnvironment(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    -1.0, 1.0, shape=(2,), dtype=np.float32
                ),
                "achieved_goal": spaces.Box(
                    0.0, 1.0, shape=(4,), dtype=np.float64
                ),
                "desired_goal": spaces.Box(
                    0.0, 1.0, shape=(4,), dtype=np.float64
                ),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.steps = 0
        self.goal = np.array([0.02, 0.02, 140 / 180, 0.0])

    def _observation(self, previous, current):
        return {
            "observation": np.array([current, previous], dtype=np.float32),
            "achieved_goal": np.array(
                [current, previous, self.steps / 1800, 0.0], dtype=np.float64
            ),
            "desired_goal": self.goal.copy(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._observation(0.0, 0.0), {}

    def step(self, action):
        del action
        previous = self.steps / 100
        self.steps += 1
        current = self.steps / 100
        terminated = self.steps == 2
        info = {
            "position_m": current * 1000,
            "elapsed_time_s": self.steps / 10,
            "max_speed_violation_m_s": 0.0,
            "goal_success": terminated,
        }
        if terminated:
            info["episode_metrics"] = {
                "completed": True,
                "travel_time_s": self.steps / 10,
                "energy_kwh": 0.01,
                "speed_violation_count": 0,
                "max_speed_violation_m_s": 0.0,
                "integrated_speed_violation_m": 0.0,
            }
        return (
            self._observation(previous, current),
            float(terminated),
            terminated,
            False,
            info,
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        del info
        achieved = np.asarray(achieved_goal)
        desired = np.asarray(desired_goal)
        return (
            (achieved[..., 1] < desired[..., 1])
            & (achieved[..., 0] >= desired[..., 0])
            & (achieved[..., 2] <= desired[..., 2])
            & (achieved[..., 3] <= desired[..., 3])
        ).astype(np.float32)


@pytest.mark.parametrize("n_sampled_goal", [0, 4])
def test_both_conditions_progress_after_completed_episode_warmup(n_sampled_goal):
    environment = TinyGoalEnvironment()
    model = SAC(
        "MultiInputPolicy",
        environment,
        seed=11,
        device="cpu",
        learning_starts=4,
        buffer_size=64,
        batch_size=4,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [8]},
        replay_buffer_class=DiagnosticGoalReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": n_sampled_goal,
            "handle_timeout_termination": False,
            "diagnostics_enabled": True,
        },
        verbose=0,
    )
    try:
        model.learn(total_timesteps=7)
        assert model.num_timesteps == 7
        assert model._n_updates == 3
        diagnostics = model.replay_buffer.replay_diagnostics
        assert diagnostics.real_rows > 0
        # Three optimizer batches of four rows; int(0.8 * 4) = three virtual.
        assert diagnostics.virtual_rows == (0 if n_sampled_goal == 0 else 9)
        done_indices = np.flatnonzero(model.replay_buffer.dones[:, 0])
        assert done_indices.size > 0
        for index in done_indices:
            assert model.replay_buffer.next_observations["achieved_goal"][
                index, 0, 0
            ] == pytest.approx(0.02)
    finally:
        environment.close()


def test_saved_model_preserves_deterministic_prediction(tmp_path):
    configuration = load_configuration()
    configuration = replace(
        configuration,
        sac=replace(configuration.sac, buffer_size=2048, batch_size=4),
    )
    environment = make_environment(configuration)
    try:
        observation, _ = environment.reset(seed=2000)
        model = _build_diagnostic_model(
            configuration, "sac-her", environment, 11, "cpu"
        )
        before, _ = model.predict(observation, deterministic=True)
        path = tmp_path / "model.zip"
        model.save(path)
        loaded = SAC.load(path, env=environment, device="cpu")
        after, _ = loaded.predict(observation, deterministic=True)
        np.testing.assert_array_equal(before, after)
        assert _model_sha256(path)
    finally:
        environment.close()


def test_checkpoint_callback_serializes_post_update_models_and_metrics(
    tmp_path, monkeypatch
):
    configuration = SimpleNamespace(
        checkpoint_steps=(5, 7),
        total_training_steps=7,
        track_splits=SimpleNamespace(development_calibration=(2000,)),
        goal_scales=GoalScales(),
        task=TaskSpecification(140.0, 0.0),
    )
    development = {
        "episodes": [{"evaluation_seed": 2000, "feasible": False}],
        "summary": {"evaluation_simulator_transitions": 2},
    }
    monkeypatch.setattr(
        "benchmarks.goal_conditioned.runner._evaluate_without_rng_perturbation",
        lambda *args, **kwargs: development,
    )
    environment = TinyGoalEnvironment()
    model = SAC(
        "MultiInputPolicy",
        environment,
        seed=11,
        device="cpu",
        learning_starts=2,
        buffer_size=64,
        batch_size=2,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [8]},
        replay_buffer_class=DiagnosticGoalReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 0,
            "handle_timeout_termination": False,
            "diagnostics_enabled": True,
        },
        verbose=0,
    )
    writer = DiagnosticWriter()
    model.set_logger(
        __import__(
            "stable_baselines3.common.logger", fromlist=["Logger"]
        ).Logger(folder=None, output_formats=[writer])
    )
    manifest = {
        "policies": {
            "sac-no-her:seed-11": {
                "status": "TRAINING",
                "development_checkpoints": [],
            }
        },
        "totals": {"development_evaluation_transitions": 0},
    }
    run_directory = tmp_path / "sac-no-her" / "training-seed-11"
    run_directory.mkdir(parents=True)
    callback = GoalStudyCallback(
        configuration=configuration,
        condition_id="sac-no-her",
        training_seed=11,
        model=model,
        run_directory=run_directory,
        manifest=manifest,
        study_root=tmp_path,
        diagnostics_writer=writer,
        started_at=time.perf_counter(),
    )
    try:
        model.learn(total_timesteps=7, callback=callback, log_interval=1)
    finally:
        environment.close()
    assert model.num_timesteps == 7
    assert model._n_updates == 5
    assert [
        row["training_transitions"]
        for row in manifest["policies"]["sac-no-her:seed-11"][
            "development_checkpoints"
        ]
    ] == [5, 7]
    first = json.loads(
        (run_directory / "step-000005/development-result.json").read_text()
    )
    final = json.loads(
        (run_directory / "step-000007/development-result.json").read_text()
    )
    assert first["gradient_updates"] == 3
    assert final["gradient_updates"] == 5
    assert (run_directory / "step-000007/model.zip").exists()
    outcomes = json.loads(
        (run_directory / "training-outcomes.json").read_text()
    )
    assert outcomes["completed_training_episode_count"] == 3
    assert outcomes["canonical_successful_episode_count"] == 3
    assert outcomes["partial_episode_at_training_boundary"][
        "transition_count"
    ] == 1


def test_evaluation_helper_restores_all_rng_states(monkeypatch):
    before = _capture_rng_state()

    def consuming_evaluation(*args, **kwargs):
        del args, kwargs
        random.random()
        np.random.random()
        torch.rand(1)
        return {"ok": True}

    monkeypatch.setattr(
        "benchmarks.goal_conditioned.runner.evaluate_model", consuming_evaluation
    )
    assert _evaluate_without_rng_perturbation(None, None, ()) == {"ok": True}
    after = _capture_rng_state()
    assert before["python"] == after["python"]
    _assert_numpy_rng_equal(before["numpy"], after["numpy"])
    assert torch.equal(before["torch_cpu"], after["torch_cpu"])


def _provenance():
    return {
        "scientific_source_sha256": dict(SCIENTIFIC_SHA256),
        "configuration_sha256": "test",
    }


def test_duplicate_study_root_is_refused_and_partial_state_is_preserved(tmp_path):
    configuration = load_configuration()
    root = tmp_path / "study"
    created, manifest = initialize_study_root(
        root, configuration, _provenance(), "cpu"
    )
    manifest["status"] = "TRAINING"
    (created / "partial-marker.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="duplicate/partial"):
        initialize_study_root(root, configuration, _provenance(), "cpu")
    assert (created / "partial-marker.txt").read_text(encoding="utf-8") == "keep"
    assert (created / "ACTIVE.lock").exists()


def test_authorized_restart_requires_preserved_interrupted_attempt(tmp_path):
    configuration = load_configuration()
    root, manifest = initialize_study_root(
        tmp_path / "attempt-1", configuration, _provenance(), "cpu"
    )
    with pytest.raises(RuntimeError, match="not marked INTERRUPTED"):
        interrupted_attempt_provenance(root)
    manifest["status"] = "INTERRUPTED"
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = interrupted_attempt_provenance(root)
    assert provenance["status"] == "INTERRUPTED"
    assert provenance["validation_opened"] is False
    (root / "validation-result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Validation results"):
        interrupted_attempt_provenance(root)


def test_terminal_position_check_respects_encoder_clipping():
    terminal = {"achieved_goal": np.array([1.0, 0.999, 0.5, 0.0])}
    assert _terminal_position_is_consistent(terminal, 1000.3, 1000.0)
    inconsistent = {"achieved_goal": np.array([0.99, 0.98, 0.5, 0.0])}
    assert not _terminal_position_is_consistent(inconsistent, 1000.3, 1000.0)


def test_validation_gate_requires_six_complete_hashed_policies(tmp_path):
    configuration = load_configuration()
    manifest = _new_manifest(configuration, _provenance(), "cpu")
    with pytest.raises(RuntimeError, match="all six"):
        validate_validation_gate(tmp_path, manifest, configuration)

    for index, item in enumerate(manifest["policies"].values()):
        model_path = tmp_path / f"model-{index}.zip"
        model_path.write_bytes(f"model-{index}".encode())
        item.update(
            {
                "status": "TRAINING_COMPLETE",
                "training_transitions": 300_000,
                "gradient_updates": 298_200,
                "final_model_path": model_path.name,
                "final_model_sha256": _model_sha256(model_path),
                "development_checkpoints": [{}] * 6,
            }
        )
    validate_validation_gate(tmp_path, manifest, configuration)
    manifest["validation_opened"] = True
    with pytest.raises(RuntimeError, match="already opened"):
        validate_validation_gate(tmp_path, manifest, configuration)


def test_scientific_hashes_still_match_preparation_commit():
    root = Path(__file__).resolve().parents[2]
    for relative, expected in SCIENTIFIC_SHA256.items():
        assert _model_sha256(root / relative) == expected


def test_replay_diagnostics_state_is_json_serializable():
    buffer = _diagnostic_buffer()
    _populate(buffer)
    buffer.sample(8)
    payload = buffer.replay_diagnostics.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["denominators"]["virtual_row_rates"] == "virtual_rows"
