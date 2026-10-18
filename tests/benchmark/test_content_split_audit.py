from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from benchmarks.content_split_audit.fingerprints import (
    ReservedTaskError,
    assert_task_not_reserved,
    canonical_json,
    default_environment_parameters,
    fingerprint_track,
)
from benchmarks.content_split_audit.generate_results import generate
from benchmarks.content_split_audit.streams import (
    SealedSplitError,
    collisions,
    continuous_tracks,
    duplicate_statistics,
    explicit_seed_stream,
    fingerprint_rows,
    track_from_seed,
)
from gym_longicontrol.domain.track import Track


def _semantics(**requirement_overrides):
    requirements = {"max_time_s": 140.0, "max_speed_violation_m_s": 0.0}
    requirements.update(requirement_overrides)
    return {
        "observation": {"name": "core-eight-dimensional-v1"},
        "dynamics": {"name": "longicontrol-native-0.1s-v1"},
        "objective": {"name": "negative-net-energy", "scale_kwh": 0.25},
        "costs": ["speed_integral_m", "deadline_deficit_integral_s"],
        "feasibility": {"all_cost_limits": 0.0},
        "termination": {"route_complete": True, "time_limit_steps": 1800},
        "requirements": requirements,
    }


def test_different_numeric_seeds_with_identical_content_hash_identically():
    historical_collision = continuous_tracks(11, [316])[0]
    development_2006 = track_from_seed(2006)
    left = fingerprint_track(historical_collision, _semantics(), numeric_seed=11)
    right = fingerprint_track(development_2006, _semantics(), numeric_seed=2006)
    assert left.route_sha256 == right.route_sha256
    assert left.full_task_sha256 == right.full_task_sha256


def test_speed_change_changes_route_and_full_task_hashes():
    track = track_from_seed(2000)
    limits = np.array(track.limits_m_s, copy=True)
    limits[0] += 1.0
    changed = Track(track.positions_m, limits)
    baseline = fingerprint_track(track, _semantics())
    modified = fingerprint_track(changed, _semantics())
    assert baseline.route_sha256 != modified.route_sha256
    assert baseline.full_task_sha256 != modified.full_task_sha256


def test_energy_or_task_only_change_preserves_route_hash():
    track = track_from_seed(2000)
    baseline = fingerprint_track(track, _semantics())
    task_change = fingerprint_track(track, _semantics(max_time_s=141.0))
    parameters = default_environment_parameters(energy_factor=1.5)
    energy_change = fingerprint_track(
        track, _semantics(), environment_parameters=parameters
    )
    assert baseline.route_sha256 == task_change.route_sha256
    assert baseline.route_sha256 == energy_change.route_sha256
    assert baseline.full_task_sha256 != task_change.full_task_sha256
    assert baseline.full_task_sha256 != energy_change.full_task_sha256


def test_fingerprint_is_stable_across_repeated_generation():
    first = fingerprint_track(track_from_seed(3000), _semantics())
    second = fingerprint_track(track_from_seed(3000), _semantics())
    assert first == second


def test_canonical_serialization_ignores_dict_order_and_identity():
    first = {"a": [1, 2.0], "b": {"c": "value"}}
    second = OrderedDict(
        [("b", OrderedDict([("c", "".join(["val", "ue"]))])), ("a", [1, 2.0])]
    )
    assert first is not second
    assert canonical_json(first) == canonical_json(second)
    with pytest.raises(ValueError, match="reserved"):
        canonical_json({"__float_hex__": "not-a-float"})


def test_paper_track_construction_is_blocked_before_generation():
    with pytest.raises(SealedSplitError, match="Paper-final"):
        track_from_seed(4000)


def test_known_current_stream_development_2006_collisions():
    seeds = explicit_seed_stream(11, domain_tag=0x4D564531, count=500)
    rows = fingerprint_rows(
        (track_from_seed(seed) for seed in seeds),
        _semantics(),
        numeric_seeds=seeds,
    )
    reference = fingerprint_rows(
        [track_from_seed(2006)], _semantics(), numeric_seeds=[2006]
    )
    found = collisions(rows, reference, split="Development", stream="T_CURRENT")
    assert [row["training_episode_index"] for row in found] == [163, 242]
    assert all(row["route_sha256_match"] for row in found)
    assert all(row["full_task_sha256_match"] for row in found)


def test_duplicate_statistics_regression_for_critical_streams():
    historical = continuous_tracks(11, list(range(0, 1000, 2)))
    historical_rows = fingerprint_rows(
        historical,
        _semantics(),
        numeric_seeds=[11, *([None] * 499)],
        draw_indices=list(range(0, 1000, 2)),
    )
    current_seeds = explicit_seed_stream(11, domain_tag=0x4D564531, count=500)
    current_rows = fingerprint_rows(
        (track_from_seed(seed) for seed in current_seeds),
        _semantics(),
        numeric_seeds=current_seeds,
    )
    historical_stats = duplicate_statistics(historical_rows)
    current_stats = duplicate_statistics(current_rows)
    assert historical_stats["route"]["unique"] == 482
    assert historical_stats["route"]["maximum_multiplicity"] == 3
    assert current_stats["route"]["unique"] == 490
    assert current_stats["route"]["maximum_multiplicity"] == 4


def test_future_split_gate_supports_all_three_policies():
    fingerprint = fingerprint_track(
        track_from_seed(12345), _semantics(), numeric_seed=12345
    )
    with pytest.raises(ReservedTaskError, match="numeric seed"):
        assert_task_not_reserved(fingerprint, {12345}, set(), set(), policy="S")
    with pytest.raises(ReservedTaskError, match="full-task"):
        assert_task_not_reserved(
            fingerprint, set(), set(), {fingerprint.full_task_sha256}, policy="F"
        )
    with pytest.raises(ReservedTaskError, match="route"):
        assert_task_not_reserved(
            fingerprint, set(), {fingerprint.route_sha256}, set(), policy="R"
        )
    assert_task_not_reserved(
        fingerprint, set(), {fingerprint.route_sha256}, set(), policy="F"
    )
    assert_task_not_reserved(fingerprint, set(), set(), set(), policy="R")


def test_static_audit_regressions_and_paper_seal():
    result = generate()
    assert result["known_2006_regression_verified"] is True
    assert result["longicontrol_simulator_transitions"] == 0
    assert result["policy_evaluation_episodes"] == 0
    assert result["paper"]["numeric_exclusion_verified"] is True
    assert result["paper"]["instantiated"] is False
    assert result["paper"]["fingerprinted"] is False
    assert result["paper"]["content_inspected"] is False
    historical = result["critical_streams"]["T_HIST"]
    current = result["critical_streams"]["T_CURRENT"]
    assert historical["validation_collisions"] == []
    assert current["validation_collisions"] == []
    assert len(result["proposed_diagnostic_set"]["candidates"]) == 18
    assert all(
        not row["evaluated"] for row in result["proposed_diagnostic_set"]["candidates"]
    )
