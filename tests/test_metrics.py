from dataclasses import FrozenInstanceError, replace

import pytest

from gym_longicontrol.domain.metrics import EpisodeMetrics, _EpisodeMetricsAccumulator
from gym_longicontrol.domain.state import VehicleState


@pytest.mark.parametrize(
    "velocities, count, maximum, integral",
    [
        ([0, 9, 10], 0, 0, 0),
        ([10, 12, 10], 1, 2, 0.5),
        ([11, 12, 13], 1, 3, 1.5),
        ([11, 12, 10, 13, 10], 2, 3, 1.5),
        ([12, 9, 11], 2, 2, 0.75),
    ],
)
def test_violation_events_maximum_and_integral(velocities, count, maximum, integral):
    accumulator = _EpisodeMetricsAccumulator()
    for velocity in velocities:
        accumulator.update(velocity_m_s=velocity, speed_limit_m_s=10, dt_s=0.25)
    metrics = accumulator.snapshot(VehicleState(), completed=False)
    assert metrics.speed_violation_count == count
    assert metrics.max_speed_violation_m_s == maximum
    assert metrics.integrated_speed_violation_m == integral
    assert accumulator.speed_excess_m_s == max(0, velocities[-1] - 10)


def test_changing_limit_and_step_duration():
    accumulator = _EpisodeMetricsAccumulator()
    for limit, dt in [(12, 0.2), (8, 0.1), (7, 0.5), (10, 0.1), (9, 0.2)]:
        accumulator.update(velocity_m_s=10, speed_limit_m_s=limit, dt_s=dt)
    metrics = accumulator.snapshot(VehicleState(), completed=False)
    assert metrics.speed_violation_count == 2
    assert metrics.max_speed_violation_m_s == 3
    assert metrics.integrated_speed_violation_m == pytest.approx(1.9)


def test_zero_initial_snapshot_and_existing_state_accounting():
    accumulator = _EpisodeMetricsAccumulator()
    assert accumulator.snapshot(VehicleState(), completed=False) == EpisodeMetrics(
        False, 0.0, 0.0, 0, 0.0, 0.0
    )
    state = VehicleState(elapsed_time_s=0.1 + 0.2, total_energy_kwh=-0.12)
    saved = accumulator.snapshot(state, completed=True)
    assert saved.travel_time_s == state.elapsed_time_s  # No rounding/reintegration.
    assert saved.energy_kwh == -0.12  # Signed net energy includes regeneration.
    assert saved.completed
    accumulator.update(velocity_m_s=11, speed_limit_m_s=10, dt_s=0.1)
    assert saved.speed_violation_count == 0
    with pytest.raises(FrozenInstanceError):
        saved.completed = False


@pytest.mark.parametrize(
    "field",
    [
        "travel_time_s",
        "energy_kwh",
        "max_speed_violation_m_s",
        "integrated_speed_violation_m",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_metrics_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        replace(EpisodeMetrics(False, 0, 0, 0, 0, 0), **{field: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"travel_time_s": -1},
        {"max_speed_violation_m_s": -1},
        {"integrated_speed_violation_m": -1},
        {"speed_violation_count": -1},
        {"speed_violation_count": 1.5},
        {"speed_violation_count": True},
        {"completed": "yes"},
        {"energy_kwh": "1.0"},
    ],
)
def test_invalid_metrics_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(EpisodeMetrics(False, 0, 0, 0, 0, 0), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"dt_s": 0},
        {"dt_s": -1},
        {"dt_s": float("inf")},
        {"velocity_m_s": float("nan")},
        {"speed_limit_m_s": float("inf")},
        {"velocity_m_s": "11"},
    ],
)
def test_invalid_sample_does_not_partially_update_statistics(changes):
    accumulator = _EpisodeMetricsAccumulator()
    sample = dict(velocity_m_s=11, speed_limit_m_s=10, dt_s=0.1)
    accumulator.update(**sample)
    before = replace(accumulator)
    with pytest.raises(ValueError):
        accumulator.update(**{**sample, **changes})
    assert accumulator == before
