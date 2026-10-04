from dataclasses import FrozenInstanceError, replace
from math import inf, nextafter

import pytest

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible


def measurements(**changes):
    return replace(EpisodeMetrics(True, 60.0, 0.2, 0, 0.0, 0.0), **changes)


def test_task_is_minimal_immutable_and_accepts_integer_or_float_budgets():
    assert TaskSpecification(60) == TaskSpecification(max_time_s=60.0)
    assert TaskSpecification(60).max_speed_violation_m_s == 0.0
    assert TaskSpecification(60, 0.1).max_speed_violation_m_s == 0.1
    with pytest.raises(FrozenInstanceError):
        TaskSpecification(60).max_time_s = 61


@pytest.mark.parametrize("value", [0, -1, inf, -inf, float("nan"), True, "60", None])
def test_invalid_time_budgets(value):
    with pytest.raises(ValueError, match="max_time_s"):
        TaskSpecification(value)


@pytest.mark.parametrize("value", [-0.1, inf, -inf, float("nan"), True, "0", None])
def test_invalid_speed_tolerances(value):
    with pytest.raises(ValueError, match="max_speed_violation_m_s"):
        TaskSpecification(60, value)


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({}, True),
        ({"completed": False}, False),
        ({"travel_time_s": 61}, False),
        ({"max_speed_violation_m_s": 0.01}, False),
        ({"energy_kwh": -0.1}, True),
        ({"energy_kwh": 100}, True),
    ],
)
def test_feasibility_uses_completion_time_and_max_excess_only(changes, expected):
    assert is_feasible(measurements(**changes), TaskSpecification(60)) is expected


def test_inclusive_boundaries_without_hidden_epsilon():
    task = TaskSpecification(60, 0.1)
    metrics = measurements(
        max_speed_violation_m_s=0.1,
        speed_violation_count=1,
        integrated_speed_violation_m=1.0,
    )
    assert is_feasible(metrics, task)
    assert not is_feasible(replace(metrics, travel_time_s=nextafter(60.0, inf)), task)
    assert not is_feasible(
        replace(metrics, max_speed_violation_m_s=nextafter(0.1, inf)), task
    )
    assert is_feasible(replace(metrics, travel_time_s=nextafter(60.0, 0)), task)
    assert not is_feasible(
        measurements(max_speed_violation_m_s=1e-15), TaskSpecification(60)
    )


def test_tasks_can_rescore_the_same_immutable_physical_measurements():
    metrics = measurements(
        travel_time_s=65,
        max_speed_violation_m_s=0.1,
        speed_violation_count=2,
        integrated_speed_violation_m=0.3,
    )
    assert not is_feasible(metrics, TaskSpecification(60, 0.1))
    assert not is_feasible(metrics, TaskSpecification(65, 0))
    assert is_feasible(metrics, TaskSpecification(65, 0.1))
    assert metrics.speed_violation_count == 2
