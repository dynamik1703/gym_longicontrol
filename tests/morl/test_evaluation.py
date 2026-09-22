import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from examples.morl.evaluation import (
    DEFAULT_WEIGHTS,
    evaluate,
    front_metrics,
    nondominated,
    rollout,
    validate_weights,
)
from examples.morl_baselines import BenchmarkConfig, main, split_budget

ROOT = Path(__file__).resolve().parents[2]


def test_import_and_help_do_not_import_training_dependencies():
    code = (
        "import sys; sys.argv=['test','--unknown']; import examples.morl_baselines; "
        "assert not {'torch','morl_baselines','mo_gymnasium','matplotlib','wandb'} "
        "& set(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "examples.morl_baselines", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and "--steps" in result.stdout


@pytest.mark.parametrize(
    "weights",
    [
        [],
        [[1, 0]],
        [[-1, 1, 0, 1]],
        [[1, 1, 1, 1]],
        [[np.nan, 0, 0, 1]],
        [DEFAULT_WEIGHTS[0]] * 2,
    ],
)
def test_invalid_preferences(weights):
    with pytest.raises(ValueError):
        validate_weights(weights)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 1},
        {"steps": -1},
        {"train_seeds": (2**32,)},
        {"batch_size": 50001, "learning_starts": 50001},
        {"train_seeds": ()},
        {"eval_seeds": (42,)},
        {"train_seeds": (42, 42)},
        {"eval_seeds": (-1,)},
        {"batch_size": 2000},
        {"reference": (0, 0)},
        {"scale": (0, 1, 1, 1)},
        {"algorithms": ("unknown",)},
        {"algorithms": ("capql", "capql")},
        {"max_episode_steps": 0},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


def test_exact_budget_split():
    assert split_budget(503, 5) == [101, 101, 101, 100, 100]
    assert sum(split_budget(503, 5)) == 503


def test_cli_rejects_invalid_config_before_creating_output(tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output), "--steps", "1"])
    assert error.value.code == 2 and not output.exists()


def test_pareto_maximization_and_duplicates():
    points = [[0, 1], [1, 0], [-1, -1], [0, 1], [0.5, 0.5]]
    np.testing.assert_array_equal(nondominated(points), [[0, 1], [0.5, 0.5], [1, 0]])
    assert nondominated(np.empty((0, 4))).shape == (0, 4)
    with pytest.raises(ValueError):
        nondominated([[np.nan, 0]])


def test_rollout_repeats_and_reports_timeout():
    first = rollout(lambda obs: np.array([0.5]), seed=2, max_episode_steps=4)
    second = rollout(lambda obs: np.array([0.5]), seed=2, max_episode_steps=4)
    assert first == second
    assert first["truncated"] and not first["completed"]
    assert first["steps"] == 4
    assert first["elapsed_time_s"] == pytest.approx(0.4)
    assert len(first["return_vector"]) == 4


def test_hypervolume_with_known_value_scaling_and_invalid_reference():
    pytest.importorskip("morl_baselines")
    result = front_metrics(
        [[1, 1, 1, 1], [0, 0, 0, 0]], reference=(0, 0, 0, 0), scale=(1, 1, 1, 1)
    )
    assert result["hypervolume"] == pytest.approx(1)
    result = front_metrics([[1, 1, 1, 1]], reference=(0, 0, 0, 0), scale=(2, 2, 2, 2))
    assert result["hypervolume"] == pytest.approx(1 / 16)
    with pytest.raises(ValueError, match="below"):
        front_metrics([[-1, 0, 0, 0]], reference=(0, 0, 0, 0))
    with pytest.raises(ValueError):
        front_metrics([[0, 0, 0, 0]], scale=(0, 1, 1, 1))


def test_evaluate_preserves_seeds_and_averages_vectors_before_front():
    pytest.importorskip("morl_baselines")
    weights = DEFAULT_WEIGHTS[:2]
    policies = [lambda obs: np.array([0.5]), lambda obs: np.array([0.1])]
    result = evaluate(policies, weights=weights, seeds=(2, 3), max_episode_steps=4)
    for row, weight in zip(result["policies"], weights):
        assert [e["seed"] for e in row["episodes"]] == [2, 3]
        expected = np.mean([e["return_vector"] for e in row["episodes"]], axis=0)
        np.testing.assert_array_equal(row["mean_return"], expected)
        assert row["mean_scalarized_return"] == pytest.approx(np.dot(weight, expected))
        assert row["completion_rate"] == 0
    assert len(result["pareto_front"]) <= 2
