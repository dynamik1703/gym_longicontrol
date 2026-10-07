import numpy as np
import pytest

from benchmarks.contrastive_rl.goal_adapter import (
    OutcomeScales,
    augment_policy_state,
    canonical_command,
    canonical_goal_set_membership,
    normalize_outcome,
    physical_outcome,
    project_outcome,
    projected_goal_is_canonical,
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


def projected(position, previous, time, violation=0.0, *, terminated=False):
    return project_outcome(
        outcome(position, previous, time, violation),
        terminated=terminated,
        task=TASK,
    )


def test_safe_arrival_at_any_allowed_time_maps_to_same_command():
    expected = canonical_command()
    for arrival_time in (100.0, 130.0, 140.0):
        np.testing.assert_array_equal(
            projected(1000, 999, arrival_time, terminated=True), expected
        )


def test_late_and_unsafe_arrivals_preserve_their_own_requirements():
    np.testing.assert_array_equal(
        projected(1000, 999, 140.0001, terminated=True), [1.0, 0.0, 1.0]
    )
    np.testing.assert_array_equal(
        projected(1000, 999, 130, 0.01, terminated=True), [1.0, 1.0, 0.0]
    )
    assert not projected_goal_is_canonical(
        projected(1000, 999, 150, terminated=True)
    )


def test_smallest_positive_violation_is_not_rounded_to_compliance():
    violation = np.nextafter(0.0, 1.0)
    goal = projected(1000, 999, 120, violation, terminated=True)
    assert violation > 0.0
    np.testing.assert_array_equal(goal, [1.0, 1.0, 0.0])


def test_safe_timely_incomplete_prefix_has_unsaturated_progress():
    goal = projected(500, 499, 100)
    np.testing.assert_array_equal(goal, [0.5, 1.0, 1.0])
    assert not projected_goal_is_canonical(goal)


def test_route_overshoot_saturates_only_projection_not_raw_outcome():
    raw = outcome(1007.25, 999.0, 130.0)
    raw_before = raw.copy()
    goal = project_outcome(raw, terminated=True, task=TASK)
    np.testing.assert_array_equal(goal, canonical_command())
    np.testing.assert_array_equal(raw, raw_before)
    assert raw[0] > 1000.0


def test_post_terminal_source_transition_is_rejected():
    with pytest.raises(ValueError, match="post-terminal"):
        project_outcome(
            outcome(1001.0, 1000.0, 131.0), terminated=True, task=TASK
        )


def test_termination_flag_must_match_route_crossing():
    with pytest.raises(ValueError, match="first route crossing"):
        projected(1000, 999, 130, terminated=False)
    with pytest.raises(ValueError, match="first route crossing"):
        projected(900, 899, 130, terminated=True)


def test_scalar_and_batched_projection_agree():
    raw = np.stack(
        [outcome(500, 499, 100), outcome(1002, 999, 141), outcome(1000, 999, 120, 0.1)]
    )
    terminals = np.array([False, True, True])
    batched = project_outcome(raw, terminated=terminals, task=TASK)
    scalar = np.stack(
        [
            project_outcome(row, terminated=terminal, task=TASK)
            for row, terminal in zip(raw, terminals, strict=True)
        ]
    )
    np.testing.assert_array_equal(batched, scalar)


def test_projection_reads_stored_future_values_not_later_live_state():
    stored = outcome(500, 499, 100)
    projected_stored = project_outcome(stored, terminated=False, task=TASK)
    live_state = stored.copy()
    live_state[:] = outcome(1000, 999, 120)
    np.testing.assert_array_equal(projected_stored, [0.5, 1.0, 1.0])
    projected_live = project_outcome(live_state, terminated=True, task=TASK)
    assert not np.array_equal(projected_stored, projected_live)


def test_canonical_command_exists_before_any_observed_success():
    failed_raw = np.stack(
        [outcome(400, 399, 80), outcome(1000, 999, 141), outcome(1000, 999, 130, 0.1)]
    )
    failed_goals = project_outcome(
        failed_raw,
        terminated=np.array([False, True, True]),
        task=TASK,
    )
    assert not projected_goal_is_canonical(failed_goals).any()
    np.testing.assert_array_equal(canonical_command(), [1.0, 1.0, 1.0])


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
    np.testing.assert_array_equal(
        project_outcome(safe_prefix, terminated=False, task=TASK),
        [0.5, 1.0, 1.0],
    )
    np.testing.assert_array_equal(
        project_outcome(later_unsafe_arrival, terminated=True, task=TASK),
        [1.0, 1.0, 0.0],
    )


def test_historical_violation_survives_later_braking():
    slowed_after_violation = outcome(700, 699, 110, 0.01)
    np.testing.assert_array_equal(
        project_outcome(slowed_after_violation, terminated=False, task=TASK),
        [0.7, 1.0, 0.0],
    )


def test_projection_rejects_noncanonical_task_variants():
    with pytest.raises(ValueError, match="only for the canonical task"):
        project_outcome(
            outcome(500, 499, 100),
            terminated=False,
            task=TaskSpecification(141.0, 0.0),
        )


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


def test_policy_state_keeps_continuous_position_and_absolute_time_information():
    scales = OutcomeScales()
    first = augment_policy_state(np.zeros(8), outcome(500, 499, 100), scales)
    second = augment_policy_state(np.zeros(8), outcome(501, 500, 101), scales)
    assert first.shape == second.shape == (12,)
    assert first[8] != second[8]
    assert first[10] != second[10]
    np.testing.assert_allclose(
        first[-4:]
        * np.array(
            [
                scales.route_length_m,
                scales.route_length_m,
                scales.horizon_s,
                scales.speed_scale_m_s,
            ]
        ),
        outcome(500, 499, 100),
    )


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
    assert projected_goal_is_canonical(
        project_outcome(achieved, terminated=True, task=TASK)
    )
    assert is_feasible(metrics, TASK)


@pytest.mark.parametrize(
    ("raw", "terminated"),
    [
        (outcome(1000, 999, 100, 0), True),
        (outcome(1007, 999, 140, 0), True),
        (outcome(999, 998, 100, 0), False),
        (outcome(1000, 999, 140.001, 0), True),
        (outcome(1000, 999, 130, 0.001), True),
    ],
)
def test_projected_command_equivalent_to_raw_first_arrival_predicate(
    raw, terminated
):
    projected_success = projected_goal_is_canonical(
        project_outcome(raw, terminated=terminated, task=TASK)
    )
    raw_success = canonical_goal_set_membership(raw, task=TASK)
    assert projected_success == raw_success
