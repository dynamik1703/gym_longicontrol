from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v2.evaluation import run
from benchmarks.bankers_viability_shield_v2.independent_verification import replay
from benchmarks.bankers_viability_shield_v2.inherited_shield_adapter import (
    METHODS,
    SafetyIntegrityError,
)
from benchmarks.bankers_viability_shield_v2.integrity import (
    PairLedger,
    exclusive_writer,
)
from benchmarks.bankers_viability_shield_v2.masked_causal_controller import (
    closest_certified,
)
from benchmarks.bankers_viability_shield_v2.serialization import (
    EvidenceError,
    atomic_json,
    canonical_bytes,
    digest,
    normalize,
    read_evidence,
    strict_load,
    write_evidence,
)
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.envs.longicontrol import LongiControlEnv


@pytest.fixture
def synthetic():
    track = make_track([0, 150, 250], [60, 20, 60])
    env = gym.wrappers.TimeLimit(LongiControlEnv(track_generator=lambda _: track), 1800)
    rows, result = run(env, "BANKERS_MASKED_CAUSAL60", max_steps=3)
    env.close()
    return track, rows, result


def test_nested_normalization_and_float_bits():
    values = {
        "z": (np.int64(4), [np.float64(np.nextafter(1.0, 2.0)), -0.0]),
        "a": {"certificate": (True, None)},
    }
    recovered = strict_load(canonical_bytes(values))
    assert normalize(values) == recovered
    assert canonical_bytes(values) == canonical_bytes(recovered)
    assert canonical_bytes(values).startswith(b'{"a":')
    assert recovered["z"][1][0].hex() == np.nextafter(1.0, 2.0).hex()
    assert recovered["z"][1][1].hex() == (-0.0).hex()
    assert digest({"x": 1.0}) != digest({"x": np.nextafter(1.0, 2.0)})


@pytest.mark.parametrize(
    "value", [np.nan, np.inf, -np.inf, {"a": [np.nan]}, {1: "bad"}, set([1]), b"data"]
)
def test_reject_unsupported_and_nonfinite(value):
    with pytest.raises((EvidenceError, TypeError)):
        canonical_bytes(value)


def test_duplicate_keys_and_nonfinite_json():
    for value in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
        with pytest.raises(EvidenceError):
            strict_load(value)


def test_disk_contract_and_independent_semantics(tmp_path, synthetic):
    track, rows, result = synthetic
    path = tmp_path / "evidence.gz"
    identity = {"source": "synthetic", "config": "synthetic"}
    metadata = write_evidence(path, rows, identity, "V2_TEST_001")
    recovered = read_evidence(path, metadata, identity, "V2_TEST_001")
    assert (
        replay(recovered, track, "BANKERS_MASKED_CAUSAL60", result, complete=False)[
            "steps"
        ]
        == 3
    )
    with pytest.raises(FileExistsError):
        write_evidence(path, rows, identity, "V2_TEST_001")
    for field in ("sha256", "canonical_sha256"):
        bad = dict(metadata, **{field: "0" * 64})
        with pytest.raises(EvidenceError):
            read_evidence(path, bad, identity, "V2_TEST_001")
    with pytest.raises(EvidenceError, match="stale"):
        read_evidence(path, metadata, {"source": "changed"}, "V2_TEST_001")
    with pytest.raises(EvidenceError, match="attempt"):
        read_evidence(path, metadata, identity, "OTHER")
    bad = dict(metadata, record_count=4)
    with pytest.raises(EvidenceError):
        read_evidence(path, bad, identity, "V2_TEST_001")
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 1
    path.write_bytes(content)
    with pytest.raises(EvidenceError, match="SHA-256"):
        read_evidence(path, metadata, identity, "V2_TEST_001")


@pytest.mark.parametrize(
    "field",
    [
        "certificates",
        "memory",
        "actual_speed_limit_m_s",
        "speed_excess_m_s",
        "deadline_diagnostic",
    ],
)
def test_missing_safety_fields(tmp_path, synthetic, field):
    _, rows, _ = synthetic
    del rows[0][field]
    with pytest.raises(EvidenceError, match="fields"):
        write_evidence(tmp_path / "missing.gz", rows, {}, "V2_MISSING")


def test_duplicate_and_missing_steps(tmp_path, synthetic):
    _, rows, _ = synthetic
    rows[1]["step"] = 0
    with pytest.raises(EvidenceError, match="duplicate"):
        write_evidence(tmp_path / "duplicate.gz", rows, {}, "V2_DUPLICATE")
    rows[1]["step"] = 2
    with pytest.raises(EvidenceError):
        write_evidence(tmp_path / "missing.gz", rows, {}, "V2_MISSING")


@pytest.mark.parametrize("field", ["velocity_m_s", "total_energy_kwh", "position_m"])
def test_forged_valid_hash_does_not_replace_physical_replay(tmp_path, synthetic, field):
    track, rows, _ = synthetic
    rows[0]["actual_next_state"][field] = np.nextafter(
        rows[0]["actual_next_state"][field], np.inf
    )
    path = tmp_path / "forged.gz"
    metadata = write_evidence(path, rows, {}, "V2_FORGED")
    disk = read_evidence(path, metadata, {}, "V2_FORGED")
    with pytest.raises(EvidenceError, match="actual successor"):
        replay(disk, track, "BANKERS_MASKED_CAUSAL60", complete=False)


def test_changed_certificate_field_is_detected(synthetic):
    track, rows, _ = synthetic
    rows[0]["certificates"][0]["minimum_velocity_slack_m_s"] += 1e-12
    with pytest.raises(EvidenceError, match="certificates"):
        replay(rows, track, "BANKERS_MASKED_CAUSAL60", complete=False)


def test_missing_deadline_value(synthetic, tmp_path):
    _, rows, _ = synthetic
    rows[0]["deadline_diagnostic"]["viable"] = None
    with pytest.raises(EvidenceError, match="deadline"):
        write_evidence(tmp_path / "bad.gz", rows, {}, "V2_BAD_DEADLINE")


def test_interrupted_atomic_publication(tmp_path, monkeypatch, synthetic):
    _, rows, _ = synthetic
    path = tmp_path / "evidence.gz"

    def interrupted(*_args):
        raise OSError("simulated interrupted publication")

    monkeypatch.setattr(
        "benchmarks.bankers_viability_shield_v2.serialization.os.link", interrupted
    )
    with pytest.raises(OSError, match="interrupted"):
        write_evidence(path, rows, {}, "V2_INTERRUPTED")
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_interrupted_atomic_update_preserves_prior(tmp_path, monkeypatch):
    path = tmp_path / "ledger.json"
    atomic_json(path, {"state": "STARTED"})

    def interrupted(*_args):
        raise OSError("interrupted")

    monkeypatch.setattr(
        "benchmarks.bankers_viability_shield_v2.serialization.os.replace", interrupted
    )
    with pytest.raises(OSError):
        atomic_json(path, {"state": "COMPLETED"})
    assert json.loads(path.read_text()) == {"state": "STARTED"}


def test_exactly_once_and_exclusive_writer(tmp_path):
    ledger = PairLedger(tmp_path / "ledger.json", ["V2_TEST"], {"source": "one"})
    with exclusive_writer(tmp_path):
        with pytest.raises(FileExistsError):
            with exclusive_writer(tmp_path):
                pass
    ledger.start("V2_TEST")
    with pytest.raises(EvidenceError, match="ambiguous"):
        PairLedger(tmp_path / "ledger.json", ["V2_TEST"], {"source": "one"})
    with pytest.raises(EvidenceError):
        ledger.start("V2_TEST")
    ledger.complete("V2_TEST", {"synthetic": True})
    with pytest.raises(EvidenceError):
        ledger.start("V2_TEST")
    with pytest.raises(EvidenceError, match="mismatch"):
        PairLedger(tmp_path / "ledger.json", ["V2_TEST"], {"source": "two"})


def test_masked_causal_tie_and_empty():
    mask = [False] * 21
    mask[0] = mask[9] = mask[11] = True
    assert closest_certified(mask, 0.0) == 9
    assert closest_certified(mask, 0.099) == 11
    with pytest.raises(SafetyIntegrityError):
        closest_certified([False] * 21, 0.0)


@pytest.mark.parametrize("method", [*METHODS, "V2_SYNTHETIC_MINIMAL"])
@pytest.mark.parametrize("speed", [0.0, 10.0, 25.0])
def test_controller_disk_replay_at_synthetic_speeds(tmp_path, method, speed):
    track = make_track([0], [100])
    state = VehicleState(velocity_m_s=speed)
    env = gym.wrappers.TimeLimit(LongiControlEnv(track_generator=lambda _: track), 1800)
    rows, _ = run(env, method, max_steps=4, initial_state=state)
    meta = write_evidence(tmp_path / "rows.gz", rows, {}, "V2_CONTROLLER_TEST")
    disk = read_evidence(tmp_path / "rows.gz", meta, {}, "V2_CONTROLLER_TEST")
    replay(disk, track, method, complete=False, initial_state=state)
    env.close()


def test_full_synthetic_lifecycle_and_metrics():
    track = make_track([0], [20])
    env = gym.wrappers.TimeLimit(LongiControlEnv(track_generator=lambda _: track), 1800)
    rows, result = run(env, "BANKERS_GREEDY")
    assert len(rows) == 1800 and rows[-1]["truncated"]
    assert not result["feasible"]
    assert result["metrics"]["max_speed_violation_m_s"] == 0
    replay(rows, track, "BANKERS_GREEDY", result)
    env.close()
