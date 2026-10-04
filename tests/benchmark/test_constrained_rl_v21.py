from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.constrained_rl_v21.analysis import (
    distribution,
    dominant_failure_classification,
    event_mechanism,
    pearson_correlation,
)
from benchmarks.constrained_rl_v21.diagnosis import (
    comparison_track,
    extract_violation_events,
    speed_limit_transitions,
    trajectory_smoothness,
    transition_crossing_time_s,
)

ROOT = Path(__file__).parents[2]


def sample(
    time_s: float,
    position_m: float,
    velocity_m_s: float,
    speed_limit_m_s: float = 10.0,
) -> dict[str, float]:
    return {
        "time_s": time_s,
        "position_m": position_m,
        "velocity_m_s": velocity_m_s,
        "speed_limit_m_s": speed_limit_m_s,
        "acceleration_m_s2": 0.25,
        "jerk_m_s3": 0.5,
        "action": 0.1,
        "deadline_slack_s": 20.0,
        "deadline_deficit_s": 0.0,
        "deadline_cost_s": 0.0,
        "lambda_speed": 2.0,
        "lambda_deadline": 1.0,
    }


def test_no_speed_violation_produces_no_event():
    samples = [sample(0.1, 1.0, 9.0), sample(0.2, 2.0, 10.0)]
    assert extract_violation_events(samples) == ()


def test_contiguous_violation_has_exact_duration_integral_and_distance():
    samples = [
        sample(0.1, 1.0, 9.0),
        sample(0.2, 2.0, 10.1),
        sample(0.3, 3.0, 10.2),
        sample(0.4, 4.0, 10.0),
    ]
    (event,) = extract_violation_events(samples)
    assert event["duration_s"] == pytest.approx(0.2)
    assert event["integrated_overspeed_m"] == pytest.approx(0.03)
    assert event["speed_cost_m"] == pytest.approx(0.03)
    assert event["distance_while_violating_m"] == pytest.approx(2.0)
    assert event["peak_overspeed_m_s"] == pytest.approx(0.2)
    assert event["first_violation_time_s"] == pytest.approx(0.2)


def test_separated_violations_produce_two_events():
    samples = [
        sample(0.1, 1.0, 10.1),
        sample(0.2, 2.0, 10.0),
        sample(0.3, 3.0, 10.2),
    ]
    events = extract_violation_events(samples)
    assert len(events) == 2
    assert [event["duration_s"] for event in events] == pytest.approx([0.1, 0.1])


def test_speed_limit_transition_and_alignment_are_signed():
    transitions = speed_limit_transitions([0.0, 2.5], [12.0, 10.0])
    assert len(transitions) == 1
    assert transitions[0].kind == "reduction"
    assert transitions[0].change_m_s == -2.0

    samples = [
        sample(0.1, 2.0, 11.0, 12.0),
        sample(0.2, 3.0, 11.0, 10.0),
        sample(0.3, 4.0, 9.0, 10.0),
    ]
    assert transition_crossing_time_s(samples, 2.5) == pytest.approx(0.15)
    (event,) = extract_violation_events(samples, transitions)
    nearest = event["nearest_speed_limit_transition"]
    assert nearest["signed_distance_m"] == pytest.approx(0.5)
    assert nearest["signed_time_s"] == pytest.approx(0.05)


def test_invalid_transition_profile_is_rejected():
    with pytest.raises(ValueError):
        speed_limit_transitions([1.0, 2.0], [10.0, 8.0])


def test_mechanism_and_formal_classification():
    boundary = {
        "peak_overspeed_m_s": 0.05,
        "nearest_speed_limit_transition": {
            "kind": "increase",
            "signed_time_s": 5.0,
        },
    }
    braking = {
        "peak_overspeed_m_s": 1.0,
        "nearest_speed_limit_transition": {
            "kind": "reduction",
            "signed_time_s": 0.05,
        },
    }
    assert event_mechanism(boundary) == "small_constant_section_boundary_overshoot"
    assert event_mechanism(braking) == "late_braking_after_speed_limit_reduction"
    assert dominant_failure_classification([event_mechanism(boundary)]) == "A"
    assert dominant_failure_classification([event_mechanism(braking)]) == "B"
    assert (
        dominant_failure_classification(
            [event_mechanism(boundary), event_mechanism(braking)]
        )
        == "F"
    )


def test_summary_math_and_constant_correlation():
    assert distribution([1.0, 3.0]) == {
        "count": 2,
        "minimum": 1.0,
        "median": 2.0,
        "mean": 2.0,
        "maximum": 3.0,
    }
    assert pearson_correlation([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0)
    assert pearson_correlation([1.0, 2.0], [0.0, 0.0]) is None
    with pytest.raises(ValueError):
        pearson_correlation([1.0], [1.0, 2.0])


def test_smoothness_uses_actions_acceleration_and_jerk():
    samples = [sample(0.1, 1.0, 9.0), sample(0.2, 2.0, 9.0)]
    samples[0]["action"] = -0.5
    samples[1]["action"] = 0.5
    samples[0]["acceleration_m_s2"] = -2.0
    samples[1]["acceleration_m_s2"] = 3.0
    samples[1]["jerk_m_s3"] = 50.0
    result = trajectory_smoothness(samples)
    assert result["max_acceleration_m_s2"] == 3.0
    assert result["max_deceleration_m_s2"] == -2.0
    assert result["max_abs_jerk_m_s3"] == 50.0
    assert result["action_sign_change_count"] == 1
    assert result["action_total_variation"] == 1.0


def test_comparison_track_is_same_feature_nearest_with_seed_tiebreak():
    failed = {"track": {"positions_m": [0.0, 100.0], "limits_m_s": [20.0, 10.0]}}
    successful = [
        {
            "track_seed": 8,
            "track": {
                "positions_m": [0.0, 100.0],
                "limits_m_s": [20.0, 15.0],
            },
        },
        {
            "track_seed": 7,
            "track": {
                "positions_m": [0.0, 100.0],
                "limits_m_s": [20.0, 10.0],
            },
        },
    ]
    assert comparison_track(failed, successful) == 7


def test_published_diagnosis_has_exact_failures_and_sealed_tracks_absent():
    diagnosis = json.loads(
        (ROOT / "benchmarks/constrained_rl_v21/diagnosis.json").read_text()
    )
    pairs = {
        (item["training_seed"], item["track_seed"])
        for item in diagnosis["failed_episodes"]
    }
    assert pairs == {
        (29, 3000),
        (29, 3001),
        (29, 3004),
        (29, 3005),
        (47, 3003),
        (47, 3007),
    }
    assert diagnosis["formal_diagnosis"]["classification"] == "F"
    assert not diagnosis["formal_diagnosis"]["v21_training_justified"]
    assert not diagnosis["reserved_final_test_evaluated"]
    assert not any(
        4000 <= value <= 4017
        for item in diagnosis["failed_episodes"]
        for value in (item["training_seed"], item["track_seed"])
    )


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        (
            "benchmarks/constrained_rl_v2/PROTOCOL.md",
            "4e5c3886b8f4c415d67eb73a8b3f8b54d7ea38d74affa2a9f98c52a2dd420277",
        ),
        (
            "benchmarks/constrained_rl_v2/RESULTS.md",
            "0c6ca0a02328c91866148d3e9e2e1bee735f39e2fd5e9f3f46cb882e77616059",
        ),
        (
            "benchmarks/constrained_rl_v2/results.json",
            "a26f8ff8f385beb528a867197628e928aff7e268a56c9ef2b89dc064a5ec7be5",
        ),
        (
            "benchmarks/constrained_rl_v2/trajectories.json",
            "5af1d8179bf8056dfa8e6a346af87f6145204ff34aa1d902c26dbcbb000a5ccf",
        ),
        (
            "benchmarks/constrained_rl/canonical.json",
            "4f9609ec2903870af0d726dedfa6028823a0a0364288e9c03a074f4e8b544a2d",
        ),
        (
            "benchmarks/scalar_sb3/results.json",
            "df2bb8b560f395a40f91082034c854474936ab59db4870fbaae2b99220164a87",
        ),
        (
            "benchmarks/credit_assignment/results.json",
            "a77cb2015178f196e8f18d3812c3bd932fa25f7af7a274e1cc517ef626d5127b",
        ),
        (
            "src/gym_longicontrol/domain/task.py",
            "56489eb5aa128608e2fa9cb37b755c0eca9efaf288965f80c5bc543ee00abdee",
        ),
        (
            "src/gym_longicontrol/domain/metrics.py",
            "1957806bd7a3d16f9b9c982fcb790e43863d2327bae06f09aea8bc5b34c86d9c",
        ),
        (
            "src/gym_longicontrol/envs/longicontrol.py",
            "b473e326d348a1a2de9e2f289555f0fe0c6464e32e856c3684d0a6b07862327a",
        ),
    ],
)
def test_frozen_artifact_hash(relative_path: str, expected: str):
    assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected


def test_diagnosis_plots_are_nonempty_images():
    plot_directory = ROOT / "benchmarks/constrained_rl_v21/plots"
    expected = {
        "failure-severity.png",
        "speed-limit-transition-alignment.png",
        "multiplier-and-validation-dynamics.png",
        "event-windows-seed-29-track-3000.png",
        "event-windows-seed-29-track-3001.png",
        "event-windows-seed-29-track-3004.png",
        "event-windows-seed-29-track-3005.png",
        "event-windows-seed-47-track-3003.png",
        "event-windows-seed-47-track-3007.png",
    }
    assert {path.name for path in plot_directory.glob("*.png")} == expected
    for path in plot_directory.glob("*.png"):
        image = plt_imread(path)
        assert image.ndim == 3
        assert image.shape[0] > 100
        assert image.shape[1] > 100


def plt_imread(path: Path) -> np.ndarray:
    from matplotlib import image

    return image.imread(path)
