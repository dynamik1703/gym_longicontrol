import copy
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.bankers_viability_shield_v1.backup_policy import (
    CONFIG,
    MIN_LIMIT,
    CausalMemory,
    SafetyIntegrityError,
)
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v3.evaluation import run
from benchmarks.bankers_viability_shield_v3.fast_certificate import (
    FastBankersShield,
    safety_view,
)
from benchmarks.bankers_viability_shield_v3.fast_kinematics import (
    fast_kinematic_successor,
)
from benchmarks.bankers_viability_shield_v3.independent_verification import (
    replay,
    verify_compact,
)
from benchmarks.bankers_viability_shield_v3.safe_action_mask import (
    closest_certified,
    masked_log_probabilities,
    safe_action_mask,
)
from benchmarks.bankers_viability_shield_v3.serialization import (
    EvidenceError,
    canonical_bytes,
    normalize,
    read_evidence,
    strict_load,
    write_evidence,
)
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel
from gym_longicontrol.envs.longicontrol import LongiControlEnv


def context(state=None, positions=(0,), limits=(100,), vehicle=None):
    v = vehicle or VehicleModel()
    state = state or VehicleState()
    track = make_track(positions, limits)
    o = observation(state, track.sense(state.position_m, 150), v.specs, CONFIG, 1.0)
    m = CausalMemory(state=state)
    m.ingest(o)
    return v, track, o, m


@pytest.mark.parametrize(
    "speed", [0.0, 1e-15, 0.5, MIN_LIMIT, 60 / 3.6, 100 / 3.6, 37.0]
)
@pytest.mark.parametrize("action", [-1.0, 0.0, 1.0])
def test_bitwise_native_parity(speed, action):
    v, _, o, m = context(VehicleState(velocity_m_s=speed))
    full, _, _ = advance(m.state, action, v, CONFIG)
    fast = fast_kinematic_successor(m.state, action, v)
    assert full.position_m.hex() == fast.position_m.hex()
    assert full.velocity_m_s.hex() == fast.velocity_m_s.hex()
    assert safety_view(BankersShield(v).certify(o, m, action).as_dict()) == safety_view(
        FastBankersShield(v).certify(o, m, action)
    )


@pytest.mark.parametrize("boundary", [149.999999999, 150.0, 150.000000001])
def test_frontier(boundary):
    v, t, o, m = context(positions=(0, boundary), limits=(60, 20))
    assert len(m.visible_signs) == int(boundary <= 150)
    s = FastBankersShield(v)
    for c in safe_action_mask(s, o, m):
        verify_compact(c, o, m, v, s.frozen.identity)


@pytest.mark.parametrize(
    "state,positions,limits",
    [
        (VehicleState(position_m=99.9, velocity_m_s=10.0), (0, 100), (60, 20)),
        (VehicleState(position_m=260.0, velocity_m_s=16.0), (0, 300), (60, 20)),
        (VehicleState(velocity_m_s=27.0), (0,), (100,)),
    ],
)
def test_inherited_rejections(state, positions, limits):
    v, t, o, m = context(state, positions, limits)
    c = FastBankersShield(v).certify(o, m, 1.0)
    assert not c["accepted"]
    verify_compact(c, o, m, v, BankersShield(v).identity)


def test_inductive_backup_and_memory():
    v, t, o, m = context(
        VehicleState(position_m=60.0, velocity_m_s=10.0), (0, 100, 200), (60, 40, 20)
    )
    s = FastBankersShield(v)
    assert len(m.visible_signs) == 2
    for _ in range(90):
        old = copy.deepcopy(m)
        assert s.certify(o, m, -1.0)["accepted"]
        m.execute(-1.0, v)
        o = observation(m.state, t.sense(m.state.position_m, 150), v.specs, CONFIG, 1.0)
        m.ingest(o)
        for x in np.linspace(m.state.position_m, 500, 101):
            assert m.bound(x) >= old.bound(x)
    assert len(m.remembered_signs) == 2 and m.state.velocity_m_s < MIN_LIMIT


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.001, 1.001, None])
def test_invalid(bad):
    v, _, o, m = context()
    f = BankersShield(v).certify(o, m, bad).as_dict()
    c = FastBankersShield(v).certify(o, m, bad)
    assert safety_view(f) == safety_view(c) and not c["accepted"]


def test_custom_nonfinite_model_falls_back():
    v, _, o, m = context()
    v.power_kw = lambda *_: float("nan")
    s = FastBankersShield(v)
    assert s.finiteness is None
    assert not s.certify(o, m, 1.0)["accepted"]


def test_stale_and_mask_identity():
    v, _, o, m = context()
    s = FastBankersShield(v)
    mask = safe_action_mask(s, o, m)
    m.state = replace(m.state, position_m=0.01)
    assert mask[0]["input_sha256"] != safe_action_mask(s, o, m)[0]["input_sha256"]
    m.last_observation = None
    assert not s.certify(o, m, 0.0)["accepted"]


def test_distribution_and_tie():
    mask = [False] * 21
    mask[9] = mask[11] = True
    assert closest_certified(mask, 0.0) == 9
    logits = np.full(21, np.nan)
    logits[9] = 0.0
    logits[11] = -1000.0
    p, lp, h = masked_log_probabilities(logits, mask)
    assert p[9] == 1 and p[11] == 0 and lp[11] == -1000 and np.isfinite(h)
    assert np.all(p[~np.array(mask)] == 0) and np.all(np.isneginf(lp[~np.array(mask)]))
    with pytest.raises(SafetyIntegrityError):
        closest_certified([False] * 21, 0.0)


@pytest.mark.parametrize(
    "method",
    [
        "CAUSAL60",
        "LEGACY_NATIVE_POST_GREEDY",
        "BANKERS_GREEDY",
        "BANKERS_MASKED_CAUSAL60",
        "V3_SYNTHETIC_MINIMAL",
    ],
)
@pytest.mark.parametrize("speed", [0.0, 10.0, 25.0])
def test_method_native_disk_replay(tmp_path, method, speed):
    track = make_track([0], [100])
    state = VehicleState(velocity_m_s=speed)
    env = gym.wrappers.TimeLimit(LongiControlEnv(track_generator=lambda _: track), 1800)
    rows, result = run(env, method, max_steps=4, initial_state=state)
    env.close()
    path = tmp_path / "e.gz"
    meta = write_evidence(path, rows, {}, "V3_TEST_NATIVE")
    disk = read_evidence(path, meta, {}, "V3_TEST_NATIVE")
    replay(disk, track, method, result, complete=False, initial_state=state)
    with pytest.raises(FileExistsError):
        write_evidence(path, rows, {}, "V3_TEST_NATIVE")
    with pytest.raises(EvidenceError):
        read_evidence(path, dict(meta, sha256="0" * 64), {}, "V3_TEST_NATIVE")
    if method in ("CAUSAL60", "LEGACY_NATIVE_POST_GREEDY"):
        assert all(not r["certificates"] for r in rows)
    forged = copy.deepcopy(disk)
    forged[0]["actual_next_state"]["total_energy_kwh"] = np.nextafter(
        forged[0]["actual_next_state"]["total_energy_kwh"], np.inf
    )
    with pytest.raises(EvidenceError):
        replay(forged, track, method, complete=False, initial_state=state)


def test_bad_transcript_rejected():
    v, _, o, m = context(VehicleState(velocity_m_s=10.0))
    s = FastBankersShield(v)
    c = s.certify(o, m, -1.0)
    c["backup_transcript_b64"] = ""
    with pytest.raises(EvidenceError):
        verify_compact(c, o, m, v, s.frozen.identity)


def test_serialization_bits_and_keys():
    value = {"tuple": (np.int64(4), np.float64(np.nextafter(1.0, 2.0)), -0.0)}
    recovered = strict_load(canonical_bytes(value))
    assert normalize(value) == recovered
    assert recovered["tuple"][-1].hex() == (-0.0).hex()
    for data in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(EvidenceError):
            strict_load(data)


def test_exactly_once_immutable_start_result_and_lock(tmp_path):
    from benchmarks.bankers_viability_shield_v3.integrity import (
        PairLedger,
        exclusive_writer,
    )

    ledger = PairLedger(
        tmp_path / "ledger.json", ["V3_TEST_LEDGER"], {"source": "test"}
    )
    ledger.start("V3_TEST_LEDGER")
    started = (tmp_path / "started/V3_TEST_LEDGER.json").read_bytes()
    with pytest.raises(EvidenceError):
        PairLedger(tmp_path / "ledger.json", ["V3_TEST_LEDGER"], {"source": "test"})
    ledger.complete("V3_TEST_LEDGER", {"steps": 1})
    assert (tmp_path / "started/V3_TEST_LEDGER.json").read_bytes() == started
    assert (tmp_path / "sealed_results/V3_TEST_LEDGER.json").exists()
    with pytest.raises(EvidenceError):
        ledger.start("V3_TEST_LEDGER")
    with exclusive_writer(tmp_path):
        with pytest.raises(FileExistsError):
            with exclusive_writer(tmp_path):
                pass


def test_numeric_paper_guard_before_construction(monkeypatch):
    from benchmarks.bankers_viability_shield_v3.integrity import guarded_track

    def forbidden(_):
        raise AssertionError("Paper must not be generated")

    monkeypatch.setattr(
        "benchmarks.content_split_audit.streams.track_from_seed", forbidden
    )
    for seed in range(4000, 4018):
        with pytest.raises(EvidenceError):
            guarded_track(seed)
