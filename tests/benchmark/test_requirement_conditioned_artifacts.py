from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BENCHMARK = ROOT / "benchmarks/requirement_conditioned"


def _read(name):
    return json.loads((BENCHMARK / name).read_text(encoding="utf-8"))


def _track_seeds(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "track_seed":
                yield item
            else:
                yield from _track_seeds(item)
    elif isinstance(value, list):
        for item in value:
            yield from _track_seeds(item)


def test_frozen_analysis_is_complete_and_internally_consistent():
    result = _read("results.json")
    validation = result["final_validation"]
    assert result["schema_version"] == 1
    assert result["result_file_count"] == 39
    assert result["primary_validation_episode_count"] == 135
    assert result["decision_gate"] == "D"
    assert not result["reserved_final_test_evaluated"]
    assert sum(row["episode_count"] for row in validation["by_margin"].values()) == 135
    assert (
        sum(row["feasible_count"] for row in validation["by_margin"].values())
        == validation["overall"]["feasible_count"]
    )
    assert validation["overall"]["requirement_satisfaction_rate"] == pytest.approx(
        validation["overall"]["feasible_count"] / 135
    )
    assert all(
        row["maximum_exposure_count_difference"] <= 1
        for row in result["training_diagnostics"]["by_training_seed"].values()
    )


def test_frozen_trajectories_cover_selected_tracks_and_requirements():
    payload = _read("trajectories.json")
    assert not payload["reserved_final_test_evaluated"]
    assert set(payload["selected_tracks"]) <= set(range(3000, 3009))
    grouped = {}
    for item in payload["trajectories"]:
        grouped.setdefault(item["track_seed"], set()).add(item["requirement_margin_s"])
    assert set(grouped) == set(payload["selected_tracks"])
    assert all(values == {20.0, 30.0, 40.0, 50.0, 60.0} for values in grouped.values())


def test_artifacts_never_contain_reserved_track_seeds():
    reserved = set(range(4000, 4018))
    for name in ("feasibility.json", "results.json", "trajectories.json"):
        assert not set(_track_seeds(_read(name))) & reserved


def test_all_declared_plots_exist_and_are_nonempty():
    expected = {
        "canonical-140-comparison.png",
        "representative-trajectory-track-3001.png",
        "representative-trajectory-track-3007.png",
        "requirement-response.png",
        "requirement-sensitivity.png",
        "training-costs-and-multipliers.png",
        "validation-learning-curves.png",
        "validation-rsr-by-requirement.png",
    }
    actual = {path.name for path in (BENCHMARK / "plots").glob("*.png")}
    assert actual == expected
    assert all(
        (BENCHMARK / "plots" / name).stat().st_size > 10_000 for name in expected
    )
