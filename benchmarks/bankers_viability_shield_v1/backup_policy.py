"""Exact causal state propagation and persistent observation lower bounds."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from math import isfinite

import numpy as np

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

MIN_LIMIT = 20.0 / 3.6
VELOCITY_MARGIN = 1e-6
POSITION_MARGIN = 1e-7
CONFIG = SimulationConfig()
LEGAL_LIMITS = tuple(k / 3.6 for k in range(20, 101, 10))
BACKUP_ID = "MAX_BRAKE_TO_MIN_THEN_MAX_BRAKE"


class SafetyIntegrityError(RuntimeError):
    """Stop execution; never substitute an uncertified fallback."""


def legal_limit(value: float) -> float:
    closest = min(LEGAL_LIMITS, key=lambda v: abs(v - value))
    if not isfinite(value) or abs(value - closest) > 1e-10:
        raise SafetyIntegrityError("unsupported legal-limit observation")
    return closest


def finite_state(state: VehicleState) -> bool:
    return all(isfinite(v) for v in asdict(state).values())


def checked_advance(state, action, vehicle):
    result, power, energy = advance(state, action, vehicle, CONFIG)
    if not finite_state(result) or not isfinite(power) or not isfinite(energy):
        raise SafetyIntegrityError("nonfinite production dynamics")
    if not 0 <= result.velocity_m_s <= 37 or result.position_m < state.position_m:
        raise SafetyIntegrityError("production dynamics violated model bounds")
    return result, power, energy


@dataclass
class CausalMemory:
    state: VehicleState = field(default_factory=VehicleState)
    # Right-continuous piecewise constant lower envelope, floor before first knot.
    envelope: list[tuple[float, float]] = field(default_factory=list)
    remembered_signs: list[tuple[float, float, float]] = field(default_factory=list)
    frontier_m: float = 0.0
    last_observation: list[float] | None = None
    current_limit_m_s: float = MIN_LIMIT
    visible_signs: list[tuple[float, float, float]] = field(default_factory=list)

    def bound(self, position: float) -> float:
        index = bisect_right(self.envelope, (position, float("inf"))) - 1
        return MIN_LIMIT if index < 0 else self.envelope[index][1]

    def ingest(self, observation) -> None:
        ordinary = np.asarray(observation, dtype=np.float64)
        if ordinary.shape != (8,) or not np.isfinite(ordinary).all():
            raise SafetyIntegrityError("ordinary observation must be finite 8-vector")
        if not finite_state(self.state):
            raise SafetyIntegrityError("nonfinite causal memory")
        if abs(ordinary[0] * 37 - self.state.velocity_m_s) > 1e-10:
            raise SafetyIntegrityError("causal velocity propagation mismatch")
        if abs(ordinary[1] * 6 - 3 - self.state.acceleration_m_s2) > 1e-10:
            raise SafetyIntegrityError("causal acceleration propagation mismatch")
        current = legal_limit(float(ordinary[2] * 37))
        future = [legal_limit(float(v * 37)) for v in ordinary[3:5]]
        distances = [float(v * 150) for v in ordinary[5:7]]
        if any(not 0 <= d <= 150 + 1e-10 for d in distances):
            raise SafetyIntegrityError("invalid preview distances")
        x = self.state.position_m
        end = x + 150 - POSITION_MARGIN
        visible = []
        previous = current
        for limit, distance in zip(future, distances):
            if distance < 150 - 1e-9 or limit != previous:
                if distance <= 0:
                    raise SafetyIntegrityError("nonfuture visible sign")
                p = x + distance
                visible.append((p - POSITION_MARGIN, p + POSITION_MARGIN, limit))
            previous = limit
        if any(a[0] >= b[0] for a, b in zip(visible, visible[1:])):
            raise SafetyIntegrityError("unordered visible signs")

        # Both slots exhaust the frozen generator's <=2 transitions / 150 m.
        def observed_bound(p):
            if p < x or p >= end:
                return MIN_LIMIT
            limit = current
            for low, high, following in visible:
                if p < low:
                    break
                if p <= high:
                    return min(limit, following)
                limit = following
            return limit

        knots = {x, end}
        for low, high, _ in visible:
            # Treat the closed uncertainty interval conservatively. At high the
            # old lower value still applies; move the new segment one ulp right.
            knots.add(low)
            knots.add(float(np.nextafter(high, float("inf"))))
        knots.update(p for p, _ in self.envelope if p >= x)
        merged = []
        for p in sorted(knots):
            value = max(self.bound(p), observed_bound(p))
            if not merged or merged[-1][1] != value:
                merged.append((p, value))
        self.envelope = merged
        self.frontier_m = max(self.frontier_m, end)
        self.visible_signs = visible
        # Intersect overlapping enclosures of the same remembered sign.
        for low, high, limit in visible:
            for i, (old_low, old_high, old_limit) in enumerate(self.remembered_signs):
                if old_limit == limit and max(low, old_low) <= min(high, old_high):
                    self.remembered_signs[i] = (
                        max(low, old_low),
                        min(high, old_high),
                        limit,
                    )
                    break
            else:
                self.remembered_signs.append((low, high, limit))
        self.remembered_signs.sort()
        self.current_limit_m_s = current
        self.last_observation = ordinary.tolist()

    def execute(self, action: float, vehicle: VehicleModel) -> None:
        self.state, _, _ = checked_advance(self.state, action, vehicle)
        self.last_observation = None

    def snapshot(self) -> dict:
        return asdict(self)

    @classmethod
    def restore(cls, data: dict) -> CausalMemory:
        data = dict(data)
        data["state"] = VehicleState(**data["state"])
        for key in ("envelope", "remembered_signs", "visible_signs"):
            data[key] = [tuple(row) for row in data[key]]
        return cls(**data)


def backup_action(_observation=None, _memory=None) -> float:
    """One policy for every unseen continuation. No seed/route argument."""
    return -1.0
