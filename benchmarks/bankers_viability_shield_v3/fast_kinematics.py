"""The production kinematic expressions in their original operation order."""

from dataclasses import dataclass
from fractions import Fraction
from math import inf, isfinite, nextafter

import numpy as np

from benchmarks.bankers_viability_shield_v1.backup_policy import (
    CONFIG,
    SafetyIntegrityError,
)
from gym_longicontrol.domain.vehicle import NumpyPowerModel, VehicleModel, VehicleSpecs


@dataclass(frozen=True)
class KinematicState:
    position_m: float
    velocity_m_s: float
    acceleration_m_s2: float


def fast_kinematic_successor(state, action, vehicle):
    dt = CONFIG.dt_s
    velocity = state.velocity_m_s
    acceleration = vehicle.acceleration_from_action(velocity, action)
    low, high = vehicle.specs.velocity_limits
    acceleration = min(max(acceleration, (low - velocity) / dt), (high - velocity) / dt)
    new_velocity = velocity + acceleration * dt
    result = KinematicState(
        position_m=state.position_m + velocity * dt + 0.5 * acceleration * dt**2,
        velocity_m_s=new_velocity,
        acceleration_m_s2=acceleration,
    )
    if not all(isfinite(v) for v in (result.position_m, new_velocity, acceleration)):
        raise SafetyIntegrityError("nonfinite production dynamics")
    if not 0 <= new_velocity <= 37 or result.position_m < state.position_m:
        raise SafetyIntegrityError("production dynamics violated model bounds")
    return result


def model_finiteness_audit(vehicle):
    """Rational absolute-sum bounds, upward rounded with generous error budget."""
    if (
        type(vehicle) is not VehicleModel
        or type(vehicle.power_model) is not NumpyPowerModel
    ):
        return None
    if vehicle.specs != VehicleSpecs():
        return None
    if any(name in vars(vehicle) for name in ("power_kw", "acceleration_from_action")):
        return None
    reference = VehicleModel()
    names = (
        "velocity_grid",
        "negative_acceleration",
        "positive_acceleration",
        "coasting_acceleration",
    )
    if any(
        not np.array_equal(getattr(vehicle, n), getattr(reference, n)) for n in names
    ):
        return None
    if len(vehicle.power_model.layers) != len(reference.power_model.layers):
        return None
    for pair, expected in zip(vehicle.power_model.layers, reference.power_model.layers):
        if any(not np.array_equal(a, b) for a, b in zip(pair, expected)):
            return None
    bounds = [38.0, 4.0]
    maxima = []
    for w, b in vehicle.power_model.layers:
        assert max(w.shape) <= 10
        out = []
        for j in range(w.shape[1]):
            exact = 2 * (
                sum(
                    Fraction(bounds[i]) * abs(Fraction(float(w[i, j])))
                    for i in range(len(bounds))
                )
                + abs(Fraction(float(b[j])))
                + 1
            )
            out.append(nextafter(float(exact), inf))
        maxima.append(max(out))
        bounds = out
    assert maxima[-1] < 1e6
    assert vehicle.negative_acceleration[5:].max() < -1.4
    for n in names:
        getattr(vehicle, n).setflags(write=False)
    return dict(
        layer_absolute_bounds=maxima,
        power_absolute_bound_kw=1e6,
        assumptions=(
            "frozen arrays/model/methods immutable; "
            "deterministic audited binary64 runtime"
        ),
    )


def reduction_domain(state):
    return (
        all(isfinite(v) and abs(v) <= 1e100 for v in vars(state).values())
        and 0 <= state.velocity_m_s <= 37
    )
