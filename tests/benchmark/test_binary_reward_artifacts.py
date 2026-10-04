from __future__ import annotations

import json
from pathlib import Path

from benchmarks.binary_reward.config import configuration_sha256, load_configuration

ROOT = Path(__file__).parents[2]
BENCHMARK = ROOT / "benchmarks/binary_reward"


def _read(name):
    return json.loads((BENCHMARK / name).read_text(encoding="utf-8"))


def _evaluation_seeds(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "evaluation_seed":
                yield item
            else:
                yield from _evaluation_seeds(item)
    elif isinstance(value, list):
        for item in value:
            yield from _evaluation_seeds(item)


def test_frozen_binary_analysis_is_complete_and_internally_consistent():
    result = _read("results.json")
    final = result["final_validation"]
    training = result["training_outcomes"]
    assert result["schema_version"] == 1
    assert result["configuration_sha256"] == configuration_sha256(load_configuration())
    assert result["result_file_count"] == 36
    assert result["decision_gate"] == "D"
    assert not result["reserved_final_test_evaluated"]
    assert final["overall"]["episode_count"] == 27
    assert final["overall"]["feasible_count"] == 0
    assert final["overall"]["failure_mode_counts"] == {"incomplete+time": 27}
    assert final["overall"]["speed_compliance_rate"] == 1.0
    assert all(row["feasible_count"] == 0 for row in final["by_training_seed"].values())
    assert training["pooled_completed_training_episode_count"] == 604
    assert training["pooled_successful_training_episode_count"] == 0
    assert (
        sum(
            row["completed_training_episode_count"]
            for row in training["by_training_seed"].values()
        )
        == training["pooled_completed_training_episode_count"]
    )
    assert len(result["learning_curves"]) == 18
    assert all(row["rsr"] == 0.0 for row in result["learning_curves"])


def test_frozen_binary_trajectories_are_validation_only_and_reproducible():
    payload = _read("trajectories.json")
    assert payload["configuration_sha256"] == configuration_sha256(load_configuration())
    assert not payload["reserved_final_test_evaluated"]
    assert set(map(int, payload["track_roles"])) == {3000, 3001}
    assert len(payload["trajectories"]) == 6
    assert set(_evaluation_seeds(payload)) == {3000, 3001}
    assert all(not item["feasible"] for item in payload["trajectories"])
    assert all(
        item["failure_mode"] == "incomplete+time" for item in payload["trajectories"]
    )
    assert all(len(item["samples"]) == 1800 for item in payload["trajectories"])


def test_binary_artifacts_never_contain_reserved_evaluation_seeds():
    reserved = set(range(4000, 4018))
    for name in ("results.json", "trajectories.json"):
        assert not set(_evaluation_seeds(_read(name))) & reserved


def test_all_binary_plots_exist_and_are_nonempty():
    expected = {
        "final-failure-modes.png",
        "final-requirement-breakdown.png",
        "frozen-method-comparison.png",
        "optimizer-diagnostics.png",
        "representative-trajectories-track-3000.png",
        "representative-trajectories-track-3001.png",
        "training-success-onset.png",
        "validation-rsr-learning-curves.png",
    }
    plot_directory = BENCHMARK / "plots"
    actual = {path.name for path in plot_directory.glob("*.png")}
    assert actual == expected
    assert all((plot_directory / name).stat().st_size > 10_000 for name in expected)
