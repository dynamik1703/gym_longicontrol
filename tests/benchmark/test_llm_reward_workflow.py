from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from benchmarks.llm_reward.candidate_validation import source_sha256
from benchmarks.llm_reward.feedback import (
    assert_development_only,
    build_feedback,
    validate_feedback,
)
from benchmarks.llm_reward.final_evaluation import validate_final_evaluation_gate
from benchmarks.llm_reward.generation_package import (
    FORBIDDEN_PACKAGE_TERMS,
    PROTECTED_TRACK_IDS,
    assert_package_is_sanitized,
    build_generation_package,
)
from benchmarks.llm_reward.history import (
    load_history,
    new_history,
    next_candidate_id,
    save_history,
    start_search,
)
from benchmarks.llm_reward.protocol import load_protocol
from benchmarks.llm_reward.ranking import (
    freeze_winner,
    rank_candidates,
    ranking_key,
    select_parents,
)
from benchmarks.llm_reward.screening import RunningStatistics

# TEST FIXTURE - NOT AN LLM CANDIDATE. It never enters repository generations.
FIXTURE_SOURCE = '''\
"""TEST FIXTURE - NOT AN LLM CANDIDATE."""
from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    value = ctx.delta_position_m
    return RewardOutput(value, {"test_progress": value})
'''


def _episode(seed, *, feasible=False, completed=False, position=100.0):
    return {
        "evaluation_seed": seed,
        "completed": completed,
        "feasible": feasible,
        "travel_time_s": 100.0 if completed else 180.0,
        "energy_kwh": 0.2,
        "speed_violation_count": 0,
        "max_speed_violation_m_s": 0.0,
        "integrated_speed_violation_m": 0.0,
        "step_count": 1000 if completed else 1800,
        "final_position_m": 1000.0 if completed else position,
        "traction_energy_kwh": 0.2,
        "regenerative_energy_kwh": 0.0,
        "mean_abs_jerk_m_s3": 0.0,
        "max_abs_jerk_m_s3": 0.0,
        "mean_abs_acceleration_m_s2": 0.0,
        "mean_abs_action": 0.0,
        "action_total_variation": 0.0,
        "mean_abs_action_change": 0.0,
        "acceleration_sign_change_count": 0,
    }


def _screening(protocol):
    episodes = [
        _episode(seed, feasible=index < 3, completed=index < 5, position=400 + index)
        for index, seed in enumerate(protocol.screening.development_tracks)
    ]
    summary = {"requirement_satisfaction_rate": 3 / 9, "completion_rate": 5 / 9}
    return {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "candidate_id": "g0-c01",
        "candidate_source_sha256": "a" * 64,
        "generation": 0,
        "evaluation_scope": "development-only",
        "route_length_m": 1000.0,
        "checkpoints": [
            {
                "simulator_transitions": step,
                "evaluation_split_id": protocol.screening.development_split_id,
                "episodes": episodes,
                "summary": summary,
                "cumulative_training_successes": index,
            }
            for index, step in enumerate((10_000, 25_000, 50_000), start=1)
        ],
        "training_outcomes": [
            {"training_step": 8_000, "success": False},
            {"training_step": 20_000, "success": True},
        ],
        "episode_reward_statistics": {
            "count": 2,
            "mean": 1.0,
            "std": 0.5,
            "min": 0.5,
            "max": 1.5,
        },
        "episode_length_statistics": {
            "count": 2,
            "mean": 1400.0,
            "std": 400.0,
            "min": 1000.0,
            "max": 1800.0,
        },
        "component_statistics": {
            "test_progress": {
                "count": 100,
                "mean": 0.1,
                "std": 0.02,
                "min": 0.0,
                "max": 0.2,
            }
        },
    }


def _fields(
    *,
    rsr=0.0,
    completion=0.0,
    progress=0.0,
    deadline=0.0,
    speed=0.0,
    severity=None,
    energy=None,
):
    return {
        "development_rsr": rsr,
        "completion_rate": completion,
        "median_incomplete_route_progress": progress,
        "deadline_compliance_among_completed": deadline,
        "speed_compliance_among_relevant": speed,
        "speed_violation_severity_among_relevant": severity,
        "mean_feasible_energy_kwh": energy,
    }


def _add_candidate(history, tmp_path, candidate_id, generation, fields):
    source = tmp_path / f"{candidate_id}.py"
    source.write_text(FIXTURE_SOURCE, encoding="utf-8")
    reflection = tmp_path / f"{candidate_id}-reflection.json"
    reflection.write_text(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "candidate_source_sha256": source_sha256(FIXTURE_SOURCE),
                "information_scope": "development-only-aggregate",
                "ranking_fields": fields,
            }
        ),
        encoding="utf-8",
    )
    history["candidates"].append(
        {
            "candidate_id": candidate_id,
            "generation": generation,
            "parent_candidate_ids": [],
            "visible_rationale": "Temporary deterministic test fixture.",
            "source_sha256": source_sha256(FIXTURE_SOURCE),
            "source_path": str(source),
            "metadata_path": None,
            "validation_attempts": [],
            "repair_attempt_count": 0,
            "status": "reflected",
            "screening": {"simulator_transitions": 50_000},
            "reflection": {
                "json_path": str(reflection),
                "markdown_path": None,
            },
            "ranking": None,
            "engineering_effort": {},
        }
    )


def test_feedback_contains_required_aggregates_but_no_track_identity():
    protocol = load_protocol()
    screening = _screening(protocol)
    feedback = build_feedback(screening, protocol)
    validate_feedback(feedback)
    serialized = json.dumps(feedback, sort_keys=True)
    assert "evaluation_seed" not in serialized
    assert (
        feedback["training_success_trajectory"]["successful_training_episode_count"]
        == 1
    )
    assert feedback["development_metrics"]["episode_count"] == 9
    assert feedback["development_metrics"]["requirement_satisfaction_rate"] == 3 / 9
    assert feedback["development_metrics"]["completion_rate"] == 5 / 9
    assert feedback["development_metrics"]["feasible_energy_kwh"]["count"] == 3
    assert (
        feedback["development_metrics"]["normalized_route_progress_for_incomplete"][
            "count"
        ]
        == 4
    )


def test_feedback_rejects_non_development_or_protected_metadata():
    protocol = load_protocol()
    screening = _screening(protocol)
    screening["checkpoints"][0]["episodes"][0]["evaluation_seed"] = 3000
    with pytest.raises(ValueError, match="exact Development tracks"):
        assert_development_only(screening, protocol)
    feedback = build_feedback(_screening(protocol), protocol)
    feedback["forbidden_validation_note"] = "hidden"
    with pytest.raises(ValueError, match="Forbidden information"):
        validate_feedback(feedback)


def test_lexicographic_ranking_prevents_standstill_and_defers_energy(tmp_path):
    protocol = load_protocol()
    history = start_search(new_history(protocol))
    _add_candidate(
        history,
        tmp_path,
        "g0-c01",
        0,
        _fields(progress=0.0, speed=1.0, severity=0.0, energy=None),
    )
    _add_candidate(
        history,
        tmp_path,
        "g0-c02",
        0,
        _fields(progress=0.4, speed=0.0, severity=2.0, energy=None),
    )
    _add_candidate(
        history,
        tmp_path,
        "g0-c03",
        0,
        _fields(
            rsr=0.5,
            completion=1.0,
            progress=1.0,
            deadline=1.0,
            speed=1.0,
            severity=0.0,
            energy=0.3,
        ),
    )
    _add_candidate(
        history,
        tmp_path,
        "g0-c04",
        0,
        _fields(
            rsr=0.5,
            completion=1.0,
            progress=1.0,
            deadline=1.0,
            speed=1.0,
            severity=0.0,
            energy=0.2,
        ),
    )
    ranked = rank_candidates(history)
    assert [item["candidate_id"] for item in ranked] == [
        "g0-c04",
        "g0-c03",
        "g0-c02",
        "g0-c01",
    ]
    assert ranking_key("standing", _fields(speed=1.0, severity=0.0, energy=None)) > (
        ranking_key("moving", _fields(progress=0.1, speed=0.0, severity=10.0))
    )


def test_parent_selection_is_cumulative_deterministic_and_records_zero_shot(tmp_path):
    protocol = load_protocol()
    history = start_search(new_history(protocol))
    for index in range(1, 6):
        _add_candidate(
            history,
            tmp_path,
            f"g0-c{index:02d}",
            0,
            _fields(rsr=index / 10, completion=index / 10, progress=index / 10),
        )
    updated, parents = select_parents(history, protocol, 1)
    assert parents == ("g0-c05", "g0-c04")
    assert updated["best_generation_zero_candidate_id"] == "g0-c05"
    assert next_candidate_id(updated, protocol, 1) == "g1-c01"
    with pytest.raises(RuntimeError, match="already selected"):
        select_parents(updated, protocol, 1)


def test_generation_zero_package_is_sanitized_and_contains_no_candidate():
    protocol = load_protocol()
    package = build_generation_package(
        0,
        protocol=protocol,
        context_directory="benchmarks/llm_reward/context",
    )
    assert "Generate exactly 5 new candidate reward files" in package
    assert "At least three of the five" in package
    assert "Parent g" not in package
    assert_package_is_sanitized(package)


def test_later_package_contains_only_selected_parent_material(tmp_path):
    protocol = load_protocol()
    history = start_search(new_history(protocol))
    for index in range(1, 6):
        _add_candidate(
            history,
            tmp_path,
            f"g0-c{index:02d}",
            0,
            _fields(rsr=index / 10, completion=index / 10, progress=index / 10),
        )
    history, parents = select_parents(history, protocol, 1)
    package = build_generation_package(
        1,
        protocol=protocol,
        context_directory="benchmarks/llm_reward/context",
        history=history,
    )
    assert "Generate exactly 3 new candidate reward files" in package
    assert all(f"Parent {candidate_id}" in package for candidate_id in parents)
    assert "Parent g0-c01" not in package
    assert "evaluation_seed" not in package
    assert_package_is_sanitized(package)


@pytest.mark.parametrize(
    "leak",
    [
        *(str(track_id) for track_id in PROTECTED_TRACK_IDS),
        *FORBIDDEN_PACKAGE_TERMS,
        "Validation metrics",
        "paper-final metadata",
    ],
)
def test_generation_package_leakage_regression(leak):
    with pytest.raises(ValueError):
        assert_package_is_sanitized(f"permitted context plus {leak}")


def test_generation_package_allows_canonical_140_second_deadline():
    assert_package_is_sanitized("Complete the route within 140 seconds.")


def test_search_history_roundtrip_preserves_protocol_and_budget(tmp_path):
    protocol = load_protocol()
    history = start_search(new_history(protocol))
    path = tmp_path / "history.json"
    save_history(path, history, protocol)
    loaded = load_history(path, protocol)
    assert loaded == history
    assert loaded["candidate_slots_by_generation"] == {"0": 5, "1": 3, "2": 3}
    assert loaded["engineering_effort"]["human_reward_edits"] == 0
    assert (
        loaded["engineering_effort"]["human_selected_coefficients_during_search"] == 0
    )


def test_freeze_selects_one_winner_closes_search_and_gates_final_runner(tmp_path):
    protocol = load_protocol()
    history = start_search(new_history(protocol))
    for generation, count in enumerate((5, 3, 3)):
        for index in range(1, count + 1):
            _add_candidate(
                history,
                tmp_path,
                f"g{generation}-c{index:02d}",
                generation,
                _fields(
                    rsr=(generation * 3 + index) / 20,
                    completion=(generation * 3 + index) / 20,
                    progress=(generation * 3 + index) / 20,
                ),
            )
    history["parent_selections"] = [
        {
            "for_generation": 1,
            "candidate_ids": ["g0-c05", "g0-c04"],
            "eligible_candidate_count": 5,
        },
        {
            "for_generation": 2,
            "candidate_ids": ["g1-c03", "g0-c05"],
            "eligible_candidate_count": 8,
        },
    ]
    history["best_generation_zero_candidate_id"] = "g0-c05"
    final_path = tmp_path / "FINAL_REWARD.py"
    closed, frozen = freeze_winner(history, protocol, destination=final_path)
    assert frozen == final_path
    assert closed["status"] == "CLOSED"
    assert closed["winner"]["candidate_id"] == "g2-c03"
    assert closed["winner"]["source_sha256"] == source_sha256(FIXTURE_SOURCE)
    assert validate_final_evaluation_gate(closed, protocol, frozen) == closed["winner"]
    with pytest.raises(RuntimeError, match="OPEN"):
        next_candidate_id(closed, protocol, 2)
    final_path.write_text(FIXTURE_SOURCE + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_final_evaluation_gate(closed, protocol, frozen)


def test_final_evaluation_is_impossible_before_search_close(tmp_path):
    protocol = load_protocol()
    history = new_history(protocol)
    reward = tmp_path / "FINAL_REWARD.py"
    reward.write_text(FIXTURE_SOURCE, encoding="utf-8")
    with pytest.raises(RuntimeError, match="CLOSED"):
        validate_final_evaluation_gate(history, protocol, reward)


def test_running_statistics_are_deterministic_and_finite():
    statistics = RunningStatistics()
    for value in (1.0, 2.0, 3.0):
        statistics.observe(value)
    assert statistics.summary() == {
        "count": 3,
        "mean": 2.0,
        "std": pytest.approx((2 / 3) ** 0.5),
        "min": 1.0,
        "max": 3.0,
    }
    with pytest.raises(ValueError, match="finite"):
        statistics.observe(float("nan"))


def test_phase_one_contains_no_generated_or_final_reward_files():
    root = Path("benchmarks/llm_reward")
    generation_files = [
        path
        for path in (root / "generations").rglob("*")
        if path.is_file() and path.name != "README.md"
    ]
    assert generation_files == []
    assert not (root / "FINAL_REWARD.py").exists()
    assert not (root / "FINAL_REWARD.metadata.json").exists()
    history = json.loads((root / "search_history.json").read_text(encoding="utf-8"))
    assert history["status"] == "NOT_STARTED"
    assert history["candidates"] == []


def test_all_prefreeze_research_and_environment_files_match_frozen_hashes():
    root = Path(__file__).parents[2]
    manifest = root / "benchmarks/llm_reward/frozen_artifacts.sha256"
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split(maxsplit=1)
        path = root / relative_path
        assert path.is_file(), relative_path
        assert sha256(path.read_bytes()).hexdigest() == digest, relative_path
        entries.append(relative_path)
    assert len(entries) == 191
    assert any(path.startswith("benchmarks/binary_reward/") for path in entries)
    assert any(
        path.startswith("benchmarks/requirement_conditioned/") for path in entries
    )
    assert any(path.startswith("benchmarks/constrained_rl_v2/") for path in entries)
    assert "src/gym_longicontrol/domain/task.py" in entries
    assert "src/gym_longicontrol/domain/metrics.py" in entries
    assert "src/gym_longicontrol/envs/longicontrol.py" in entries


def test_development_scope_rejects_wrong_split_even_with_allowed_seed_set():
    protocol = load_protocol()
    screening = deepcopy(_screening(protocol))
    screening["checkpoints"][0]["evaluation_split_id"] = "other-split"
    with pytest.raises(ValueError, match="non-Development"):
        assert_development_only(screening, protocol)
