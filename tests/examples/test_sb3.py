import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
import torch  # noqa: E402
from stable_baselines3.common.env_checker import check_env  # noqa: E402

from examples import sb3_quickstart as example  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def short_env(monkeypatch):
    original = example.make_env

    def make(env_id=example.ENV_IDS[0], *, render_mode=None):
        return gym.wrappers.TimeLimit(original(env_id, render_mode=render_mode), 4)

    monkeypatch.setattr(example, "make_env", make)


def test_import_is_passive_and_help_works():
    code = (
        "import sys; sys.argv=['test', '--unknown']; "
        "import examples.sb3_quickstart; "
        "assert 'stable_baselines3' not in sys.modules; "
        "assert 'torch' not in sys.modules; assert 'matplotlib' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "examples.sb3_quickstart", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "train" in result.stdout


def test_missing_optional_dependency_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    with pytest.raises(ImportError, match=r"\[examples\]"):
        example._sb3()


@pytest.mark.parametrize("env_id", example.ENV_IDS)
def test_sb3_environment_contract(env_id):
    env = example.make_env(env_id)
    try:
        check_env(env.unwrapped)
    finally:
        env.close()


def test_train_save_reload_and_no_overwrite(tmp_path, short_env):
    settings = copy.deepcopy(example.SAC_SETTINGS)
    threads = torch.get_num_threads()
    path = example.train(tmp_path / "run", steps=24, learning_starts=8)
    assert torch.get_num_threads() == threads
    assert example.SAC_SETTINGS == settings  # SB3 must not mutate shared defaults.
    loaded, metadata = example.load_model(path / "model.zip")
    assert metadata["training_steps"] == loaded.num_timesteps == 24
    assert loaded._n_updates > 0
    assert metadata["model_sha256"] == example.sha256(path / "model.zip")
    assert metadata["training_script_sha256"] == example.sha256(example.__file__)
    report = json.loads((path / "evaluation.json").read_text())
    assert report["model_sha256"] == metadata["model_sha256"]
    assert report["sac"]["summary"]["completion_rate"] == 0
    assert all(row["truncated"] for row in report["sac"]["episodes"])
    first = example.rollout(loaded, seed=6)
    second, _ = example.load_model(path / "model.zip")
    assert first == example.rollout(second, seed=6)
    checksum = example.sha256(path / "model.zip")
    with pytest.raises(FileExistsError):
        example.train(path, steps=24)
    assert checksum == example.sha256(path / "model.zip")


def test_random_rollout_is_repeatable_and_accounts_for_timeout(short_env):
    first = example.rollout(seed=10)
    assert first == example.rollout(seed=10)
    metrics = first["metrics"]
    assert metrics["truncated"] and not metrics["completed"]
    assert metrics["steps"] == 4
    assert metrics["elapsed_time_s"] == pytest.approx(0.4)
    assert metrics["energy_kwh"] == first["trajectory"][-1]["total_energy_kwh"]


def test_rollout_records_finish(monkeypatch):
    from gym_longicontrol.domain.state import SimulationConfig

    def short_track(*args, **kwargs):
        return gym.make(
            "gym_longicontrol:DeterministicTrack-v1",
            config=SimulationConfig(track_length_m=5),
            speed_limit_positions=[0],
            speed_limits=[30],
        )

    class Accelerate:
        def predict(self, observation, deterministic):
            assert deterministic
            return np.array([1.0]), None

    monkeypatch.setattr(example, "make_env", short_track)
    metrics = example.rollout(Accelerate())["metrics"]
    assert metrics["completed"] and not metrics["truncated"]
    assert metrics["distance_m"] >= 5


def test_evaluation_uses_requested_seeds(short_env):
    result = example.compare(None, seeds=[7, 9])
    assert result["sac"] == result["random"]
    assert [row["seed"] for row in result["sac"]["episodes"]] == [7, 9]
    assert result["sac"]["summary"]["episodes"] == 2
    with pytest.raises(ValueError, match="seed"):
        example.evaluate(seeds=[])


def test_bundled_model_reproduces_documented_episode():
    model, metadata = example.load_model()
    report = json.loads((example.DEMO_DIR / "evaluation.json").read_text())
    assert report["model_sha256"] == metadata["model_sha256"]
    expected = report["sac"]["episodes"][0]
    actual = example.rollout(model, env_id=metadata["env_id"], seed=expected["seed"])
    assert actual["metrics"]["completed"] == expected["completed"]
    assert actual["metrics"]["truncated"] == expected["truncated"]
    for key in ("energy_kwh", "return", "distance_m", "elapsed_time_s"):
        assert actual["metrics"][key] == pytest.approx(
            expected[key], rel=1e-3, abs=1e-3
        )


def test_checksum_is_checked_before_deserialization(tmp_path, monkeypatch):
    shutil.copyfile(example.DEMO_DIR / "metadata.json", tmp_path / "metadata.json")
    (tmp_path / "model.zip").write_bytes(b"corrupted checkpoint")
    monkeypatch.setattr(example, "_sb3", lambda: pytest.fail("Must not deserialize"))
    with pytest.raises(ValueError, match="SHA-256"):
        example.load_model(tmp_path / "model.zip")


def test_json_writer_does_not_overwrite(tmp_path):
    path = tmp_path / "report.json"
    example.write_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        example.write_json(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--steps", "0"],
        ["train", "--seed", "-1"],
        ["train", "--learning-starts", "-1"],
        ["demo", "--episodes", "0"],
    ],
)
def test_cli_rejects_invalid_counts(args):
    with pytest.raises(SystemExit) as error:
        example.main(args)
    assert error.value.code == 2


def test_plot_has_explicit_units_and_works_headless(short_env):
    import matplotlib.pyplot as plt

    figure = example.plot_rollout(example.rollout())
    try:
        figure.canvas.draw()
        assert [axes.get_ylabel() for axes in figure.axes] == [
            "Speed [km/h]",
            "Acceleration [m/s²]",
            "Net energy [Wh]",
        ]
    finally:
        plt.close(figure)


def test_demo_cli_creates_report(tmp_path, short_env, capsys):
    path = tmp_path / "report.json"
    example.main(["demo", "--episodes", "1", "--report", str(path)])
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads(path.read_text())
    assert report["sac"]["summary"]["episodes"] == 1
    with pytest.raises(SystemExit) as error:
        example.main(["demo", "--report", str(path)])
    assert error.value.code == 1
