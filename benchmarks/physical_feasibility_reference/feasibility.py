"""Physical envelopes and conservative feasibility classifications.

Only an executed simulator trajectory establishes a feasible witness.  A failed
controller remains unresolved unless an independent lower bound proves that the
deadline is impossible.  The route-speed integral is deliberately reported as
an optimistic heuristic, not promoted to a certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.domain.track import Track
from gym_longicontrol.domain.vehicle import VehicleModel

CANONICAL_TASK = TaskSpecification(max_time_s=140.0, max_speed_violation_m_s=0.0)
MAX_EPISODE_STEPS = 1_800


class FeasibilityClassification(str, Enum):
    """The three evidence classes permitted by the protocol."""

    FEASIBLE_WITNESS = "FEASIBLE_WITNESS"
    PROVEN_INFEASIBLE = "PROVEN_INFEASIBLE"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class BrakingResult:
    initial_velocity_m_s: float
    target_velocity_m_s: float
    elapsed_time_s: float
    distance_m: float
    final_velocity_m_s: float
    steps: int


def maximum_braking_result(
    initial_velocity_m_s: float,
    target_velocity_m_s: float,
    *,
    vehicle: VehicleModel | None = None,
    config: SimulationConfig = SimulationConfig(),
) -> BrakingResult:
    """Execute exact production steps at action -1 until the target is met."""

    if initial_velocity_m_s < 0 or target_velocity_m_s < 0:
        raise ValueError("velocities must be nonnegative")
    model = vehicle if vehicle is not None else VehicleModel()
    state = VehicleState(velocity_m_s=float(initial_velocity_m_s))
    # The historical interpolation fixes braking acceleration to zero at
    # exactly zero speed.  Its low-speed branch therefore approaches zero
    # asymptotically.  Use a declared 1 micrometre/s practical standstill for
    # target zero while preserving the requested target in the result.
    integration_target = (
        1e-6 if target_velocity_m_s == 0.0 else target_velocity_m_s
    )
    steps = 0
    while state.velocity_m_s > integration_target:
        state, _, _ = advance(state, -1.0, model, config)
        steps += 1
        if steps > 100_000:
            raise RuntimeError("maximum-braking integration did not converge")
    return BrakingResult(
        initial_velocity_m_s=float(initial_velocity_m_s),
        target_velocity_m_s=float(target_velocity_m_s),
        elapsed_time_s=state.elapsed_time_s,
        distance_m=state.position_m,
        final_velocity_m_s=state.velocity_m_s,
        steps=steps,
    )


def braking_distance_m(
    initial_velocity_m_s: float,
    target_velocity_m_s: float,
    *,
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> float:
    """Return the exact discrete maximum-braking distance."""

    if initial_velocity_m_s <= target_velocity_m_s:
        return 0.0
    return maximum_braking_result(
        initial_velocity_m_s,
        target_velocity_m_s,
        vehicle=vehicle,
        config=config,
    ).distance_m


def acceleration_envelope(
    velocities_m_s: np.ndarray | None = None,
    *,
    vehicle: VehicleModel | None = None,
) -> list[dict[str, float]]:
    """Sample the production action-to-acceleration envelope."""

    model = vehicle if vehicle is not None else VehicleModel()
    velocities = (
        np.linspace(0.0, model.specs.velocity_limits[1], 38)
        if velocities_m_s is None
        else np.asarray(velocities_m_s, dtype=np.float64)
    )
    return [
        {
            "velocity_m_s": float(velocity),
            "velocity_km_h": float(velocity * 3.6),
            "maximum_acceleration_m_s2": model.acceleration_from_action(
                float(velocity), 1.0
            ),
            "coasting_acceleration_m_s2": model.acceleration_from_action(
                float(velocity), 0.0
            ),
            "maximum_braking_m_s2": model.acceleration_from_action(
                float(velocity), -1.0
            ),
        }
        for velocity in velocities
    ]


def optimistic_route_time_s(
    track: Track, *, config: SimulationConfig = SimulationConfig()
) -> float:
    """Integrate distance/limit, ignoring acceleration and transition timing."""

    ends = np.r_[track.positions_m[1:], config.track_length_m]
    return float(np.sum((ends - track.positions_m) / track.limits_m_s))


def unconstrained_minimum_time_lower_bound_s(
    *,
    config: SimulationConfig = SimulationConfig(),
    vehicle: VehicleModel | None = None,
) -> float:
    """Fastest production rollout after removing every route speed constraint.

    Maximum action is pointwise acceleration-maximal in the production vehicle
    model.  Removing route limits can only make the original task easier, so the
    resulting completion time is a valid lower bound for deadline proofs.
    """

    model = vehicle if vehicle is not None else VehicleModel()
    state = VehicleState()
    steps = 0
    while state.position_m < config.track_length_m:
        state, _, _ = advance(state, 1.0, model, config)
        steps += 1
        if steps > MAX_EPISODE_STEPS:
            raise RuntimeError("unconstrained lower-bound rollout did not finish")
    return state.elapsed_time_s


def classify_outcome(
    metrics: EpisodeMetrics | None,
    task: TaskSpecification = CANONICAL_TASK,
    *,
    unconstrained_time_lower_bound_s: float | None = None,
) -> tuple[FeasibilityClassification, str]:
    """Classify evidence without interpreting optimizer failure as impossibility."""

    if metrics is not None and is_feasible(metrics, task):
        return (
            FeasibilityClassification.FEASIBLE_WITNESS,
            "executed original-simulator trajectory satisfies all hard requirements",
        )
    if (
        unconstrained_time_lower_bound_s is not None
        and unconstrained_time_lower_bound_s > task.max_time_s
    ):
        return (
            FeasibilityClassification.PROVEN_INFEASIBLE,
            "constraint-free exact-dynamics minimum-time lower bound exceeds deadline",
        )
    return (
        FeasibilityClassification.UNRESOLVED,
        "no feasible witness and no rigorous infeasibility certificate",
    )


def physical_audit() -> dict[str, Any]:
    """Return deterministic physical findings used by reports and tests."""

    vehicle = VehicleModel()
    config = SimulationConfig()
    braking_100_to_60 = maximum_braking_result(
        100.0 / 3.6, 60.0 / 3.6, vehicle=vehicle, config=config
    )
    stopping = {
        str(speed): asdict(
            maximum_braking_result(
                speed / 3.6, 0.0, vehicle=vehicle, config=config
            )
        )
        for speed in (20, 40, 60, 80, 100)
    }
    return {
        "production_advance": "gym_longicontrol.domain.dynamics.advance",
        "vehicle": "BMW_electric_i3_2014",
        "simulation": asdict(config),
        "vehicle_specs": asdict(vehicle.specs),
        "acceleration_envelope": acceleration_envelope(vehicle=vehicle),
        "braking_100_to_60": asdict(braking_100_to_60),
        "minimum_stopping_profiles": stopping,
        "standstill_semantics": (
            "maximum braking approaches exact zero asymptotically because the "
            "production interpolation has zero braking acceleration at zero; "
            "reported stopping profiles use 1e-6 m/s practical standstill"
        ),
        "unconstrained_minimum_time_lower_bound_s": (
            unconstrained_minimum_time_lower_bound_s(
                config=config, vehicle=vehicle
            )
        ),
        "t_min_status": "optimistic heuristic; not a certified exact minimum",
    }
