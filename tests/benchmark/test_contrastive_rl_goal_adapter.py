import numpy as np
import pytest

from benchmarks.contrastive_rl.goal_adapter import (
    OutcomeScales,
    augment_policy_state,
    canonical_goal_set_membership,
    normalize_outcome,
    physical_outcome,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

TASK = TaskSpecification(max_time_s=140.0, max_speed_violation_m_s=0.0)


def outcome(position, previous, time, violation=0.0):
    return physical_outcome(
        position_m=position,
        previous_position_m=previous,
        elapsed_time_s=time,
        max_speed_violation_m_s=violation,
    )


def test_deadline_is_an_inclusive_inequality_not_a_point_goal():
    assert canonical_goal_set_membership(outcome(1000, 999, 100), task=TASK)
    assert canonical_goal_set_membership(outcome(1000, 999, 140), task=TASK)
    assert not canonical_goal_set_membership(outcome(1000, 999, 140.001), task=TASK)


def test_first_arrival_crossing_not_endpoint_occupancy():
    assert canonical_goal_set_membership(outcome(1002, 999, 120), task=TASK)
    assert not canonical_goal_set_membership(outcome(1002, 1000, 120), task=TASK)
    assert not canonical_goal_set_membership(outcome(999, 998, 120), task=TASK)


def test_previous_violation_remains_invalidating_but_safe_prefix_is_valid_state():
    safe_prefix = outcome(500, 499, 60, 0.0)
    later_unsafe_arrival = outcome(1000, 999, 120, 0.01)
    assert safe_prefix[3] == 0.0
    assert not canonical_goal_set_membership(safe_prefix, task=TASK)
    assert not canonical_goal_set_membership(later_unsafe_arrival, task=TASK)


def test_timeout_without_first_arrival_is_not_success():
    assert not canonical_goal_set_membership(outcome(900, 899, 180), task=TASK)


def test_normalization_does_not_clip_and_cannot_hide_overshoot_or_violation():
    scales = OutcomeScales()
    raw = outcome(1005, 999, 181, 38)
    normalized = normalize_outcome(raw, scales)
    assert normalized[0] > 1.0
    assert normalized[2] > 1.0
    assert normalized[3] > 1.0
    assert not canonical_goal_set_membership(raw, task=TASK)


def test_goal_arrays_are_values_not_views_of_mutable_evaluator_state():
    scales = OutcomeScales()
    raw = outcome(10, 5, 1, 0)
    augmented = augment_policy_state(np.arange(8), raw, scales)
    raw[:] = 99
    np.testing.assert_allclose(augmented[-4:], [0.01, 0.005, 1 / 180, 0])


def test_vectorized_membership_and_validation():
    result = canonical_goal_set_membership(
        np.stack([outcome(1000, 999, 140), outcome(1000, 999, 141)]),
        task=TASK,
    )
    np.testing.assert_array_equal(result, [True, False])
    with pytest.raises(ValueError):
        physical_outcome(
            position_m=5,
            previous_position_m=6,
            elapsed_time_s=1,
            max_speed_violation_m_s=0,
        )


def test_first_arrival_membership_matches_authoritative_episode_evaluator():
    achieved = outcome(1000, 999, 140, 0)
    metrics = EpisodeMetrics(
        completed=True,
        travel_time_s=140,
        energy_kwh=1.2,
        speed_violation_count=0,
        max_speed_violation_m_s=0,
        integrated_speed_violation_m=0,
    )
    assert canonical_goal_set_membership(achieved, task=TASK)
    assert is_feasible(metrics, TASK)
