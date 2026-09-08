import copy
import json
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from training.cli import _build_agent, get_args, run  # noqa: E402
from training.sac import (  # noqa: E402
    InitPolicy,
    ReplayBuffer,
    numpy_to_torch,
    torch_to_numpy,
)


@pytest.fixture
def agent():
    args = get_args(
        [
            "--device",
            "cpu",
            "--optimization_batch",
            "4",
            "--replay_buffer_capacity",
            "16",
            "--hidden_layer_sizes",
            "8",
            "8",
        ]
    )
    env = gym.make("gym_longicontrol:DeterministicTrack-v1", max_episode_steps=3)
    evaluation = gym.make("DeterministicTrack-v1", max_episode_steps=3)
    result = _build_agent(args, env, evaluation)
    yield result
    env.close()
    evaluation.close()


def test_imports_do_not_parse_arguments_or_start_training(tmp_path):
    # CLI and parser imports also work without importing torch or gymnasium.
    code = (
        "import sys; sys.argv=['test','--unknown']; "
        "import training.cli; import training; "
        "assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_replay_first_sample_wrap_and_reproducibility():
    buffers = [ReplayBuffer(4, 1, 2, 1, seed=9) for _ in range(2)]
    for buffer in buffers:
        buffer.add_experience(([5, 6], [0.2], 7, [6, 7], False))
        assert buffer.sample_minibatch()[0].tolist() == [[5, 6]]
        for i in range(7):
            buffer.add_experience(([i, i + 1], [0.1], i, [i + 1, i + 2], True))
        assert len(buffer) == 4
        assert set(buffer.R[:, 0]) == {3, 4, 5, 6}
    for a, b in zip(buffers[0].sample_minibatch(), buffers[1].sample_minibatch()):
        np.testing.assert_array_equal(a, b)


def test_replay_rejects_unfilled_sampling():
    buffer = ReplayBuffer(4, 2, 2, 1)
    with pytest.raises(ValueError):
        buffer.sample_minibatch()


def test_initial_policy_covers_full_action_range():
    policy = InitPolicy(shape=(1,), seed=4)
    values = np.array([policy.get_action().item() for _ in range(1000)])
    assert values.min() < -0.9 and values.max() > 0.9
    assert (abs(values) <= 1).all()


def test_tensor_conversion_detaches():
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    np.testing.assert_array_equal(torch_to_numpy(value * 2), [2, 4])
    assert numpy_to_torch(np.array([1.0])).dtype == torch.float32


def test_training_updates_parameters_and_bootstraps_time_limits(agent):
    before = copy.deepcopy(agent.policy_function.state_dict())
    policy = InitPolicy.from_action_space(agent.env.action_space, seed=1)
    agent.init_replay_buffer(policy, 8, seed=1)
    agent.do_training(8, seed=2)
    assert not agent.replay_buffer.done.any()  # every short episode is truncated
    assert all(np.isfinite(value) for value in agent.losses.values())
    assert any(
        not torch.equal(value, before[key])
        for key, value in agent.policy_function.state_dict().items()
    )


def test_evaluation_does_not_change_training_state_or_rng(agent):
    agent.env.reset(seed=9)
    state = agent.env.unwrapped.state
    rng = copy.deepcopy(agent.env.unwrapped.np_random.bit_generator.state)
    result = agent.do_evaluation(2, seed=5)
    assert result[1] == 2
    assert agent.env.unwrapped.state == state
    assert agent.env.unwrapped.np_random.bit_generator.state == rng


def test_checkpoint_roundtrip_and_exact_next_update(agent, tmp_path):
    agent.init_replay_buffer(InitPolicy(shape=(1,), seed=5), 8, seed=5)
    agent.do_training(4, seed=6)
    path = save_checkpoint(tmp_path / "agent.tar", agent, epoch=1, history={})
    agent.do_training(4, seed=7)
    expected = copy.deepcopy(agent.policy_function.state_dict())
    losses = dict(agent.losses)
    load_checkpoint(path, agent)
    assert agent.alpha_optimizer.param_groups[0]["params"][0] is agent.log_alpha
    agent.do_training(4, seed=7)
    for key, value in agent.policy_function.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert agent.losses == losses


def test_checkpoint_rejects_unknown_schema_and_config(agent, tmp_path):
    path = save_checkpoint(
        tmp_path / "agent.tar",
        agent,
        epoch=1,
        history={},
        metadata={"config": {"env_id": "DeterministicTrack-v1"}},
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_checkpoint(path, agent, expected_config={"env_id": "StochasticTrack-v1"})
    torch.save({"agent": {}, "format_version": 999}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint"):
        load_checkpoint(path, agent)


def test_bundled_legacy_checkpoint_still_loads():
    root = Path(__file__).resolve().parents[2]
    checkpoint = root / "rl/pytorch/out/DeterministicTrack-v0/SAC_id9/seed2.tar"
    if not checkpoint.exists():
        pytest.skip("Historical demo checkpoint is only present in a git checkout")
    args = get_args(
        [
            "--device",
            "cpu",
            "--replay_buffer_capacity",
            "16",
            "--optimization_batch",
            "4",
        ]
    )
    env = gym.make("gym_longicontrol:DeterministicTrack-v1", max_episode_steps=3)
    try:
        agent = _build_agent(args, env, env)
        loaded = load_checkpoint(
            checkpoint, agent, load_optimizers=False, trust_legacy=True
        )
        assert loaded.metadata["legacy"]
        result = agent.do_evaluation(1)
        assert np.isfinite(result[0]) and result[1] == 1
    finally:
        env.close()


def test_cli_saves_last_epoch_and_resumes(tmp_path, monkeypatch):
    from training import cli

    original_make = cli.make_environment

    def short_environment(args, render_mode=None):
        return gym.wrappers.TimeLimit(original_make(args, render_mode), 5)

    monkeypatch.setattr(cli, "make_environment", short_environment)
    common = [
        "--device",
        "cpu",
        "--num_epochs",
        "2",
        "--num_steps_per_epoch",
        "8",
        "--optimization_batch",
        "4",
        "--replay_buffer_capacity",
        "16",
        "--num_evaluation_episodes",
        "1",
        "--hidden_layer_sizes",
        "8",
        "8",
        "--output_dir",
        str(tmp_path),
        "--evaluation_interval",
        "100",
    ]
    directory = run(get_args(common + ["--save_id", "1"]))
    checkpoint = torch.load(directory / "seed2.tar", weights_only=True)
    assert checkpoint["training"]["epoch"] == 2
    assert checkpoint["training"]["history"]["training_steps"] == [16]
    assert json.loads((directory / "seed2.config.json").read_text())["seed"] == 2
    with pytest.raises(FileExistsError):
        run(get_args(common + ["--save_id", "1"]))
    run(get_args(common + ["--load_id", "1"]))
    checkpoint = torch.load(directory / "seed2.tar", weights_only=True)
    assert checkpoint["training"]["epoch"] == 4
    assert checkpoint["training"]["history"]["training_steps"] == [16, 32]


def test_video_recording(agent, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("moviepy")
    env = gym.make(
        "DeterministicTrack-v1", render_mode="rgb_array", max_episode_steps=2
    )
    try:
        agent.do_visualization(record=True, save_dname=tmp_path, environment=env)
    finally:
        env.close()
    videos = list(tmp_path.rglob("*.mp4"))
    assert len(videos) == 1 and videos[0].stat().st_size > 0
