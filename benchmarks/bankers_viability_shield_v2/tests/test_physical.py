from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v1.safe_action_set import safe_action_set
from benchmarks.bankers_viability_shield_v2.independent_verification import (
    certificate_from_physics,
    verify_envelope,
)
from benchmarks.bankers_viability_shield_v2.inherited_shield_adapter import (
    ACTION_GRID,
    CONFIG,
    MIN_LIMIT,
    VELOCITY_MARGIN,
    BankersShield,
    CausalMemory,
    SafetyIntegrityError,
)
from benchmarks.bankers_viability_shield_v2.serialization import canonical_bytes
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel


def context(state=None, positions=(0,), limits=(100,)):
    vehicle = VehicleModel()
    track = make_track(positions, limits)
    state = state or VehicleState()
    ordinary = observation(
        state, track.sense(state.position_m, 150), vehicle.specs, CONFIG, 1.0
    )
    memory = CausalMemory(state=state)
    memory.ingest(ordinary)
    return vehicle, track, ordinary, memory


@pytest.mark.parametrize("limit", [20, 100])
def test_v2_initial_certified(limit):
    vehicle, _, obs, m = context(limits=(limit,))
    menu = safe_action_set(BankersShield(vehicle), obs, m)
    assert menu.mask[0] and menu.mask[-1]


@pytest.mark.parametrize("x", [149.999999999, 150.0, 150.000000001])
def test_v2_exact_preview_boundary(x):
    v, t, o, m = context(positions=(0, x), limits=(60, 20))
    assert len(m.visible_signs) == int(x <= 150)
    verify_envelope(m, CausalMemory(), t)
    for action in ACTION_GRID:
        shield = BankersShield(v)
        c = shield.certify(o, m, action)
        assert canonical_bytes(c.as_dict()) == canonical_bytes(
            certificate_from_physics(o, m, action, v, shield.identity)
        )


def test_v2_within_step_transition_and_unsafe_acceleration():
    v, t, o, m = context(
        VehicleState(position_m=99.9, velocity_m_s=10.0), (0, 100), (60, 20)
    )
    c = BankersShield(v).certify(o, m, 1.0)
    assert c.predicted_next_state["position_m"] > 100
    assert not c.accepted
    assert (
        c.predicted_next_state["velocity_m_s"]
        > t.sense(c.predicted_next_state["position_m"], 150).current_limit_m_s
    )


def test_v2_multiple_signs_and_retention():
    v, t, o, m = context(
        VehicleState(position_m=60.0, velocity_m_s=10.0), (0, 100, 200), (60, 40, 20)
    )
    assert len(m.visible_signs) == 2
    old = copy.deepcopy(m)
    shield = BankersShield(v)
    for _ in range(60):
        assert shield.certify(o, m, -1.0).accepted
        m.execute(-1.0, v)
        o = observation(m.state, t.sense(m.state.position_m, 150), v.specs, CONFIG, 1.0)
        m.ingest(o)
        verify_envelope(m, old, t)
        old = copy.deepcopy(m)
    assert len(m.remembered_signs) == 2


@pytest.mark.parametrize(
    "speed",
    [
        100 / 3.6,
        np.nextafter(MIN_LIMIT, -np.inf),
        MIN_LIMIT,
        np.nextafter(MIN_LIMIT, np.inf),
        MIN_LIMIT - VELOCITY_MARGIN,
        np.nextafter(MIN_LIMIT - VELOCITY_MARGIN, np.inf),
    ],
)
def test_v2_rounding_and_maximum_legal_speed(speed):
    v, t, o, m = context(VehicleState(velocity_m_s=float(speed)))
    shield = BankersShield(v)
    for a in ACTION_GRID:
        c = shield.certify(o, m, a)
        expected = certificate_from_physics(o, m, a, v, shield.identity)
        assert canonical_bytes(c.as_dict()) == canonical_bytes(expected)
        if c.accepted:
            successor, _, _ = advance(m.state, a, v, CONFIG)
            assert (
                successor.velocity_m_s
                <= t.sense(successor.position_m, 150).current_limit_m_s
            )
    # A speed exactly at a legal limit is not automatically in the inner set:
    # current-state certification includes the full pre-frontier brake witness.


def test_v2_full_backup_after_admitted_action():
    v, t, o, m = context(VehicleState(position_m=100.0, velocity_m_s=20.0))
    shield = BankersShield(v)
    c = shield.certify(o, m, 1.0)
    assert c.accepted
    m.execute(1.0, v)
    for _ in range(c.backup_steps + 30):
        o = observation(m.state, t.sense(m.state.position_m, 150), v.specs, CONFIG, 1.0)
        m.ingest(o)
        assert shield.certify(o, m, -1.0).accepted
        m.execute(-1.0, v)
        assert (
            m.state.velocity_m_s <= t.sense(m.state.position_m, 150).current_limit_m_s
        )


def test_v2_empty_mask_and_stale_observation_fail_closed(monkeypatch):
    v, _, o, m = context()
    shield = BankersShield(v)
    rejected = replace(shield.certify(o, m, 0.0), accepted=False)
    monkeypatch.setattr(shield, "certify", lambda *_: rejected)
    with pytest.raises(SafetyIntegrityError, match="EMPTY_MASK"):
        safe_action_set(shield, o, m)
    assert not BankersShield(v).certify(o + 1e-3, m, 1.0).accepted


def test_v2_zero_is_coasting():
    v, _, _, m = context(VehicleState(velocity_m_s=10.0))
    brake, _, _ = advance(m.state, -1.0, v, CONFIG)
    coast, _, _ = advance(m.state, 0.0, v, CONFIG)
    assert 0 < brake.velocity_m_s < coast.velocity_m_s < m.state.velocity_m_s
    assert coast.acceleration_m_s2 != 0
