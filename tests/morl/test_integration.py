"""Real optional-library smoke tests, including gradient updates and artifacts."""

import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("morl_baselines")
import mo_gymnasium as mo_gym  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from mo_gymnasium.wrappers import LinearReward, MORecordEpisodeStatistics  # noqa: E402

from examples import morl_baselines as example  # noqa: E402
from examples.morl.evaluation import (  # noqa: E402
    DEFAULT_REFERENCE,
    DEFAULT_WEIGHTS,
    make_env,
)


@pytest.fixture(autouse=True)
def no_external_logging(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Integration must not create or log to an external W&B run")

    for name in ("init", "log", "finish"):
        monkeypatch.setattr(wandb, name, forbidden)


@pytest.mark.parametrize("env_id", example.ENV_IDS)
def test_mo_gymnasium_wrappers(env_id):
    env = MORecordEpisodeStatistics(mo_gym.make(env_id, max_episode_steps=3))
    try:
        env.reset(seed=1)
        total = np.zeros(4)
        for _ in range(3):
            _, reward, _, truncated, info = env.step([0.5])
            total += reward
        assert truncated
        np.testing.assert_allclose(info["episode"]["r"], total)
    finally:
        env.close()
    scalar = LinearReward(make_env(env_id), weight=np.array(DEFAULT_WEIGHTS[0]))
    try:
        scalar.reset(seed=1)
        _, reward, _, _, info = scalar.step([0.5])
        assert np.isscalar(reward)
        assert reward == pytest.approx(
            np.dot(DEFAULT_WEIGHTS[0], list(info["reward_components"].values()))
        )
    finally:
        scalar.close()


def test_capql_updates_save_reload_and_deterministic_eval(tmp_path):
    CAPQL = example._dependencies()[2]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env, eval_env = make_env(max_episode_steps=4), make_env(max_episode_steps=4)
    try:
        env.reset(seed=5)
        env.action_space.seed(5)
        model = CAPQL(
            env,
            net_arch=[16, 16],
            learning_starts=8,
            batch_size=8,
            buffer_size=100,
            log=False,
            seed=5,
            device="cpu",
        )
        original = [p.detach().clone() for p in model.policy.parameters()]
        model.train(
            total_timesteps=24, eval_env=eval_env, ref_point=np.array(DEFAULT_REFERENCE)
        )
        assert model.global_step == 24
        assert any(
            not torch.equal(before, after)
            for before, after in zip(original, model.policy.parameters())
        )
        assert len(model.replay_buffer) == 24
        obs, _ = eval_env.reset(seed=9)
        weight = np.array(DEFAULT_WEIGHTS[0])
        action = model.eval(obs, weight)
        assert env.action_space.contains(action.astype(env.action_space.dtype))
        model.save(save_dir=str(tmp_path), filename="capql", save_replay_buffer=False)
        loaded = CAPQL(env, net_arch=[16, 16], log=False, seed=10, device="cpu")
        # Native upstream loading is only appropriate for these trusted local files.
        loaded.load(str(tmp_path / "capql.tar"), load_replay_buffer=False)
        np.testing.assert_array_equal(loaded.eval(obs, weight), action)
    finally:
        env.close()
        eval_env.close()
        torch.set_num_threads(previous_threads)


def test_complete_comparison_has_equal_budgets_and_repeatable_results(tmp_path):
    config = example.BenchmarkConfig(
        steps=125,
        train_seeds=(7, 8),
        eval_seeds=(1001, 1002),
        learning_starts=8,
        batch_size=8,
        max_episode_steps=4,
    )
    threads = torch.get_num_threads()
    output = tmp_path / "first"
    first = example.run(output, config)
    second = example.run(tmp_path / "second", config, plot=False)
    assert torch.get_num_threads() == threads
    assert len(first["runs"]) == 4
    assert (output / "return_projections.png").stat().st_size > 0
    assert (
        json.loads((output / "evaluation.json").read_text())["summary"]
        == first["summary"]
    )
    for a, b in zip(first["runs"], second["runs"]):
        assert a["policies"] == b["policies"]
        assert a["hypervolume"] == b["hypervolume"]
        assert a["training_steps"] == sum(a["steps_per_policy"]) == 125
        for artifact in a["artifacts"]:
            path = output / artifact["path"]
            assert example._artifact(path, output) == artifact
    for name, summary in first["summary"].items():
        values = [
            row["hypervolume"] for row in first["runs"] if row["algorithm"] == name
        ]
        assert summary["training_seeds"] == 2
        assert summary["hypervolume_mean"] == pytest.approx(np.mean(values))
    # CAPQL's seed argument does not seed all its global RNGs upstream. The
    # adapter must reproduce the same policy without a preceding SAC run.
    capql_only = example.run(
        tmp_path / "capql-only",
        replace(config, algorithms=("capql",), train_seeds=(7,)),
        plot=False,
    )
    assert capql_only["runs"][0]["policies"] == first["runs"][1]["policies"]
    with pytest.raises(FileExistsError):
        example.run(output, config)


def test_missing_optional_dependency_install_hint(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "morl_baselines.multi_policy.capql.capql", None)
    with pytest.raises(ImportError, match=r"\[morl\]"):
        example._dependencies()


def test_rollout_records_completion(monkeypatch):
    from examples.morl import evaluation
    from gym_longicontrol.domain.state import SimulationConfig

    def short_track(*args, **kwargs):
        return gym.make(
            "MODeterministicTrack-v1",
            config=SimulationConfig(track_length_m=1),
            speed_limit_positions=[0],
            speed_limits=[30],
        )

    monkeypatch.setattr(evaluation, "make_env", short_track)
    result = evaluation.rollout(lambda obs: np.array([1.0]))
    assert result["completed"] and not result["truncated"]
    assert result["distance_m"] >= 1
    assert Path(example.__file__).is_file()
