"""Causal deadline viability estimate used by the Hybrid safety layer."""

from __future__ import annotations

from dataclasses import dataclass

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

DEADLINE_S = 140.0
DEADLINE_RESERVE_S = 2.0
MINIMUM_LIMIT_M_S = 20.0 / 3.6
VELOCITY_MARGIN_M_S = 1e-6


@dataclass(frozen=True)
class DeadlineViability:
    """A falsifiable causal recoverability estimate, not a global proof."""

    completion_bound_s: float
    deadline_s: float
    reserve_s: float
    viable: bool
    assumption: str
    physics_prediction_calls: int


def pessimistic_completion_bound(
    state: VehicleState,
    *,
    vehicle: VehicleModel | None = None,
    config: SimulationConfig = SimulationConfig(),
    deadline_s: float = DEADLINE_S,
    reserve_s: float = DEADLINE_RESERVE_S,
) -> DeadlineViability:
    """Bound completion using a causal all-20-km/h unknown-tail envelope.

    If the vehicle is below the generator's minimum limit, exact maximum-
    traction dynamics are integrated until that floor is reached. Remaining
    distance is then charged at ``20 km/h - frozen velocity margin``. A state is
    admitted only when that conservative estimate plus the frozen reserve fits.

    The construction is deliberately testable. It is not promoted to a formal
    global guarantee because no proof covers every discrete fallback/transition
    interaction in the stochastic generator.
    """

    model = vehicle if vehicle is not None else VehicleModel()
    floor = MINIMUM_LIMIT_M_S - VELOCITY_MARGIN_M_S
    cursor = state
    calls = 0
    while cursor.velocity_m_s < floor and cursor.position_m < config.track_length_m:
        cursor, _, _ = advance(cursor, 1.0, model, config)
        calls += 1
        if calls > 10_000:
            raise RuntimeError("deadline viability acceleration failed to converge")
    remaining = max(0.0, config.track_length_m - cursor.position_m)
    completion_bound_s = cursor.elapsed_time_s + remaining / floor
    return DeadlineViability(
        completion_bound_s=completion_bound_s,
        deadline_s=float(deadline_s),
        reserve_s=float(reserve_s),
        viable=completion_bound_s <= deadline_s - reserve_s,
        assumption=(
            "exact acceleration to the generator minimum limit, then the entire "
            "unknown remainder at 20 km/h minus the frozen velocity margin"
        ),
        physics_prediction_calls=calls,
    )
