from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.bankers_viability_shield_v1.backup_policy import (
    CONFIG,
    MIN_LIMIT,
    VELOCITY_MARGIN,
    CausalMemory,
    SafetyIntegrityError,
    backup_action,
)
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v1.integrity import (
    PairLedger,
    exclusive_writer,
    verify_sources,
    verify_task,
)
from benchmarks.bankers_viability_shield_v1.pre_shield import PreShield
from benchmarks.bankers_viability_shield_v1.safe_action_set import (
    ACTION_GRID,
    masked_distribution,
    safe_action_set,
)
from benchmarks.bankers_viability_shield_v1.verification import verify_certificate
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.track import StochasticTrackGenerator
from gym_longicontrol.domain.vehicle import VehicleModel
from gym_longicontrol.envs.longicontrol import LongiControlEnv


def context(state=None, positions=(0, 300, 400), limits=(100, 60, 20)):
    vehicle = VehicleModel()
    track = make_track(positions, limits)
    state = state or VehicleState()
    obs = observation(
        state, track.sense(state.position_m, 150), vehicle.specs, CONFIG, 1
    )
    memory = CausalMemory(state=state)
    memory.ingest(obs)
    return vehicle, track, obs, memory


@pytest.mark.parametrize("speed", [0, 1e-15, 0.5, MIN_LIMIT, 60 / 3.6, 100 / 3.6, 37])
@pytest.mark.parametrize("action", [-1.0, 0.0, 1.0])
def test_production_dynamics_and_native_sign(speed, action):
    v, _, obs, m = context(VehicleState(velocity_m_s=speed))
    c = BankersShield(v).certify(obs, m, action)
    successor, _, _ = advance(m.state, action, v, CONFIG)
    assert c.predicted_next_state == asdict(successor)
    if speed > 0 and action == -1:
        assert successor.velocity_m_s < speed
    if speed > 0 and action == 0:
        assert successor.velocity_m_s < speed  # zero means real coasting
    if speed == 0 and action == 1:
        assert successor.velocity_m_s > 0


@pytest.mark.parametrize("p", [149.999999999, 150, 150.000000001])
def test_exact_preview_boundary(p):
    v, t, obs, m = context(positions=(0, p), limits=(60, 20))
    assert len(m.visible_signs) == int(p <= 150)
    assert m.bound(p + 1e-6) <= t.sense(p + 1e-6, 150).current_limit_m_s
    verify_certificate(BankersShield(v).certify(obs, m, -1), obs, m, v)


def test_endpoint_crossed_inside_step_and_exact_speed():
    v, t, obs, m = context(
        VehicleState(position_m=99.9, velocity_m_s=10),
        positions=(0, 100),
        limits=(60, 20),
    )
    c = BankersShield(v).certify(obs, m, 1)
    assert c.predicted_next_state["position_m"] > 100
    assert not c.accepted and not c.right_endpoint_compliant
    _, _, obs, m = context(
        VehicleState(velocity_m_s=np.nextafter(MIN_LIMIT, np.inf)),
        positions=(0,),
        limits=(20,),
    )
    assert not BankersShield(v).certify(obs, m, 1).accepted


def test_known_transition_uses_successor_and_brake_suffix():
    v, _, obs, m = context(
        VehicleState(position_m=260, velocity_m_s=16),
        positions=(0, 300),
        limits=(60, 20),
    )
    c = BankersShield(v).certify(obs, m, 1)
    assert c.right_endpoint_compliant
    assert not c.accepted and not c.known_transition_braking_feasible
    assert c.backup_steps > 0


def test_unknown_frontier_consumes_proposed_step():
    v, _, obs, m = context(VehicleState(velocity_m_s=27), positions=(0,), limits=(100,))
    c = BankersShield(v).certify(obs, m, 1)
    assert c.preview_available_after_action_m < 150
    assert not c.accepted and not c.unknown_preview_recoverable


def test_base_backup_nonempty_and_mask_exact_grid():
    v, _, obs, m = context()
    shield = BankersShield(v)
    menu = safe_action_set(shield, obs, m)
    assert ACTION_GRID == tuple(i / 10 for i in range(-10, 11))
    assert menu.mask[0] and len(menu.mask) == 21
    for c in menu.certificates:
        verify_certificate(c, obs, m, v)


def test_empty_mask_fail_closed(monkeypatch):
    v, _, obs, m = context()
    shield = BankersShield(v)
    rejected = replace(shield.certify(obs, m, 0), accepted=False)
    monkeypatch.setattr(shield, "certify", lambda *_: rejected)
    with pytest.raises(SafetyIntegrityError, match="EMPTY_MASK"):
        safe_action_set(shield, obs, m)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.001, 1.001, None])
def test_nonfinite_and_outside_actions_fail_closed(bad):
    v, _, obs, m = context()
    c = BankersShield(v).certify(obs, m, bad)
    assert not c.accepted


def test_nonfinite_physics_fail_closed(monkeypatch):
    v, _, obs, m = context()
    monkeypatch.setattr(v, "power_kw", lambda *_: float("nan"))
    assert not BankersShield(v).certify(obs, m, 1).accepted


def test_discrepant_observation_fail_closed():
    _, _, obs, m = context()
    with pytest.raises(SafetyIntegrityError):
        m.ingest(obs + 0.01)


def test_two_signs_remembered_and_envelope_monotone():
    v, t, obs, m = context(
        VehicleState(position_m=60, velocity_m_s=10),
        positions=(0, 100, 200),
        limits=(60, 40, 20),
    )
    assert len(m.visible_signs) == 2
    old = copy.deepcopy(m)
    m.execute(-1, v)
    obs = observation(m.state, t.sense(m.state.position_m, 150), v.specs, CONFIG, 1)
    m.ingest(obs)
    for x in np.linspace(m.state.position_m, 500, 1001):
        assert m.bound(x) >= old.bound(x)
        assert m.bound(x) <= t.sense(x, 150).current_limit_m_s
    assert len(m.remembered_signs) == 2


def test_explicit_backup_and_induction_after_actual_execution():
    v, t, obs, m = context(
        VehicleState(position_m=200, velocity_m_s=15),
        positions=(0, 300, 400),
        limits=(60, 40, 20),
    )
    shield = BankersShield(v)
    assert shield.certify(obs, m, -1).accepted
    for _ in range(70):
        before = shield.certify(obs, m, backup_action())
        assert before.accepted
        actual, _, _ = advance(m.state, -1, v, CONFIG)
        m.execute(-1, v)
        assert m.state == actual
        assert actual.velocity_m_s <= t.sense(actual.position_m, 150).current_limit_m_s
        obs = observation(m.state, t.sense(m.state.position_m, 150), v.specs, CONFIG, 1)
        m.ingest(obs)
        assert shield.certify(obs, m, -1).accepted
    assert m.state.velocity_m_s <= MIN_LIMIT - VELOCITY_MARGIN


def test_native_environment_observation_parity_and_final_selection():
    track = make_track([0, 150, 250], [60, 20, 60])
    env = gym.wrappers.TimeLimit(LongiControlEnv(track_generator=lambda _: track), 1800)
    obs, _ = env.reset()
    controller = PreShield()
    for _ in range(30):
        menu = controller.prepare(obs)
        index = max(i for i, b in enumerate(menu.mask) if b)
        actor = controller.actor_input(obs)
        assert set(actor) == {
            "ordinary_observation",
            "physical_memory",
            "remaining_distance_m",
            "remaining_deadline_s",
            "safe_action_mask",
        }
        action, _ = controller.select(index, obs)
        obs, _, _, _, info = env.step([action])
        controller.observe_executed_action(action)
        assert controller.memory.state == env.unwrapped.state
        assert info["speed_excess_m_s"] == 0
    env.close()


def test_final_discrepancy_stale_mask_invalid_index():
    _, _, obs, _ = context()
    p = PreShield()
    p.prepare(obs)
    with pytest.raises(SafetyIntegrityError):
        p.select(100, obs)
    p.memory.state = replace(p.memory.state, position_m=0.01)
    with pytest.raises(SafetyIntegrityError, match="DISCREPANCY"):
        p.select(0, obs)


def test_invalid_probabilities_entropy_and_targets_excluded():
    mask = [False] * 21
    mask[0] = mask[20] = True
    logits = np.full(21, 1e300)
    logits[0] = logits[20] = 0
    p, entropy = masked_distribution(logits, mask)
    assert p[0] == p[20] == 0.5 and sum(p) == 1
    assert entropy == pytest.approx(np.log(2))
    assert np.dot(p, np.arange(21)) == 10
    with pytest.raises(SafetyIntegrityError):
        masked_distribution(logits, [False] * 21)


def test_no_hidden_route_or_deadline_claim_in_certificate():
    _, _, obs, m = context()
    c = BankersShield().certify(obs, m, 0).as_dict()
    assert not any(
        k in json.dumps(m.snapshot()) for k in ("reset_seed", "full_route", "oracle")
    )
    assert not any("deadline" in key or "completion" in key for key in c)
    assert c["accepted"]  # safe at rest even with impossible conservative deadline


def test_exact_checkpoint_memory_and_repeatability():
    v, _, obs, m = context(VehicleState(position_m=170, velocity_m_s=10))
    restored = CausalMemory.restore(json.loads(json.dumps(m.snapshot())))
    shield = BankersShield(v)
    a = safe_action_set(shield, obs, m)
    b = safe_action_set(shield, obs, restored)
    assert a == b and a.sha256 == b.sha256
    assert a.distinct_successors == 21


def test_generator_assumptions_and_braking_tail():
    generator = StochasticTrackGenerator()
    v = VehicleModel()
    assert np.max(v.negative_acceleration[1:]) < -1.4
    assert np.all(v.negative_acceleration <= 0)
    assert v.negative_acceleration[0] == 0
    for seed in range(100):  # synthetic generator probes, not performance episodes
        t = generator(np.random.default_rng(seed + 98700000))
        assert np.diff(t.positions_m).min() >= 100
        assert set(t.limits_m_s) <= set(k / 3.6 for k in range(20, 101, 10))
        assert np.all(np.abs(np.diff(t.limits_m_s)) * 3.6 <= 40 + 1e-12)


def test_source_immutability():
    assert verify_sources()["checked_files"] > 100


def test_exactly_once_ledger_and_ambiguous_resume(tmp_path):
    tasks = [{"numeric_seed": 11500000}]
    path = tmp_path / "ledger.json"
    ledger = PairLedger(path, tasks, {"id": "test"})
    key = "BANKERS_PRE__11500000"
    ledger.start(key)
    with pytest.raises(SafetyIntegrityError):
        ledger.start(key)
    with pytest.raises(SafetyIntegrityError, match="ambiguous"):
        PairLedger(path, tasks, {"id": "test"})
    ledger.complete(key, {"steps": 1})
    resumed = PairLedger(path, tasks, {"id": "test"})
    with pytest.raises(SafetyIntegrityError):
        resumed.start(key)
    with pytest.raises(SafetyIntegrityError, match="mismatch"):
        PairLedger(path, tasks, {"id": "different"})


def test_exclusive_writer(tmp_path):
    with exclusive_writer(tmp_path):
        with pytest.raises(FileExistsError):
            with exclusive_writer(tmp_path):
                pass


def test_paper_guard_before_task_generation(monkeypatch):
    def forbidden(_seed):
        raise AssertionError("must never construct Paper")

    monkeypatch.setattr(
        "benchmarks.bankers_viability_shield_v1.integrity.track_from_seed", forbidden
    )
    for seed in range(4000, 4018):
        with pytest.raises(SafetyIntegrityError, match="PAPER"):
            verify_task({"numeric_seed": seed})


def test_tampered_witness_rejected():
    v, _, obs, m = context(VehicleState(velocity_m_s=12))
    c = BankersShield(v).certify(obs, m, -1).as_dict()
    c["backup_steps"] += 1
    with pytest.raises(SafetyIntegrityError, match="step count"):
        verify_certificate(c, obs, m, v)
