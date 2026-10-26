import copy
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG, CausalMemory
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v3.fast_certificate import (
    FastBankersShield,
    safety_view,
)
from benchmarks.bankers_viability_shield_v4.compact_certificate import CompactShield
from benchmarks.bankers_viability_shield_v4.evaluation import run
from benchmarks.bankers_viability_shield_v4.independent_verification import (
    replay,
    verify_compact,
)
from benchmarks.bankers_viability_shield_v4.integrity import (
    PairLedger,
    exclusive_writer,
    guarded_track,
    source_identity,
)
from benchmarks.bankers_viability_shield_v4.serialization import (
    EvidenceError,
    atomic_json,
    canonical_bytes,
    native_journal,
    read_evidence,
    strict_load,
    write_evidence,
)
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel
from gym_longicontrol.envs.longicontrol import LongiControlEnv


@pytest.fixture(scope="module")
def identity():
    return source_identity()


def context(state=None):
    vehicle = VehicleModel()
    track = make_track([0], [100])
    state = state or VehicleState()
    o = observation(
        state, track.sense(state.position_m, 150), vehicle.specs, CONFIG, 1.0
    )
    m = CausalMemory(state=state)
    m.ingest(o)
    return vehicle, track, o, m


@pytest.mark.parametrize(
    "kind",
    [
        "nan_action",
        "inf_action",
        "range",
        "none",
        "stale",
        "nan_state",
        "inf_energy",
        "huge_energy",
    ],
)
def test_fail_closed(kind, identity):
    v, _, o, m = context()
    action = 0.0
    if kind == "nan_action":
        action = float("nan")
    if kind == "inf_action":
        action = float("inf")
    if kind == "range":
        action = 1.1
    if kind == "none":
        action = None
    if kind == "stale":
        m.last_observation = None
    if kind == "nan_state":
        m.state = replace(m.state, velocity_m_s=float("nan"))
    if kind == "inf_energy":
        m.state = replace(m.state, total_energy_kwh=float("inf"))
    if kind == "huge_energy":
        m.state = replace(m.state, total_energy_kwh=1e308)
    c = CompactShield(identity).certify(o, m, action)
    frozen = FastBankersShield().certify(o, m, action)
    assert canonical_bytes(safety_view(c)) == canonical_bytes(safety_view(frozen))
    verify_compact(c, o, m, v, BankersShield(v).identity, identity, raw_action=action)


@pytest.mark.parametrize(
    "method",
    [
        "CAUSAL60",
        "LEGACY_NATIVE_POST_GREEDY",
        "BANKERS_GREEDY",
        "BANKERS_MASKED_CAUSAL60",
    ],
)
@pytest.mark.parametrize("speed", [0.0, 10.0, 25.0])
def test_native_disk_replay(method, speed, tmp_path, identity):
    _, track, _, _ = context()
    state = VehicleState(velocity_m_s=speed)
    env = gym.wrappers.TimeLimit(LongiControlEnv(track_generator=lambda _: track), 1800)
    attempt = "V4_UNIT_" + method
    with native_journal(tmp_path / "native.jsonl", identity, attempt) as sink:
        rows, result = run(
            env,
            method,
            max_steps=3,
            initial_state=state,
            evidence_identity=identity,
            attempt=attempt,
            record_sink=sink,
        )
    meta = write_evidence(tmp_path / "native.gz", rows, identity, attempt)
    disk = read_evidence(tmp_path / "native.gz", meta, identity, attempt)
    replay(
        disk,
        track,
        method,
        result,
        complete=False,
        initial_state=state,
        evidence_identity=identity,
        attempt=attempt,
    )
    assert b"backup_transcript_b64" not in (tmp_path / "native.jsonl").read_bytes()
    if method in ("CAUSAL60", "LEGACY_NATIVE_POST_GREEDY"):
        assert all(not r["certificates"] for r in disk)
    with pytest.raises(FileExistsError):
        write_evidence(tmp_path / "native.gz", rows, identity, attempt)
    with pytest.raises(EvidenceError):
        read_evidence(
            tmp_path / "native.gz",
            dict(meta, record_count=meta["record_count"] + 1),
            identity,
            attempt,
        )
    env.close()


@pytest.fixture(scope="module")
def native(identity):
    _, track, _, _ = context()
    state = VehicleState(velocity_m_s=10.0)
    env = gym.wrappers.TimeLimit(LongiControlEnv(track_generator=lambda _: track), 1800)
    rows, result = run(
        env,
        "BANKERS_MASKED_CAUSAL60",
        max_steps=2,
        initial_state=state,
        evidence_identity=identity,
    )
    env.close()
    return rows, result, track, state


@pytest.mark.parametrize(
    "corruption",
    [
        "mask_bit",
        "rejection_reason",
        "backup_hash",
        "successor_position",
        "margin",
        "native_energy",
        "executed_action",
        "source_identity",
        "count",
        "order",
        "timestamp",
        "lifecycle",
        "deadline",
        "valid_prefix",
        "selected_certificate",
        "missing_field",
    ],
)
def test_corruption_semantics(corruption, native, identity):
    rows, result, track, state = native
    changed = copy.deepcopy(rows)
    r = changed[0]
    c = r["certificates"][0]
    if corruption == "mask_bit":
        r["safe_action_mask"][0] = not r["safe_action_mask"][0]
    if corruption == "rejection_reason":
        c["rejection_reason"] = "UNKNOWN_PREVIEW_BACKUP"
    if corruption == "backup_hash":
        c["backup_trajectory_sha256"] = "0" * 64
    if corruption == "successor_position":
        c["predicted_next_state"]["position_m"] = float(
            np.nextafter(c["predicted_next_state"]["position_m"], np.inf)
        )
    if corruption == "margin":
        c["numerical_margins"]["velocity_margin_m_s"] *= 2
    if corruption == "native_energy":
        r["signed_energy_increment_kwh"] += 1e-8
    if corruption == "executed_action":
        r["executed_action"] = -0.123
    if corruption == "source_identity":
        r["evidence_identity"]["configuration_sha256"] = "0" * 64
    if corruption == "count":
        changed.pop()
    if corruption == "order":
        changed.reverse()
    if corruption == "timestamp":
        r["native_time_s"] += 0.1
    if corruption == "lifecycle":
        r["terminated"] = True
    if corruption == "deadline":
        r["deadline_diagnostic"]["viable"] = not r["deadline_diagnostic"]["viable"]
    if corruption == "valid_prefix":
        c["backup_valid_prefix_steps"] += 1
    if corruption == "selected_certificate":
        r["selected_certificate"]["accepted"] = False
    if corruption == "missing_field":
        del r["actual_power_kw"]
    with pytest.raises(EvidenceError):
        replay(
            changed,
            track,
            "BANKERS_MASKED_CAUSAL60",
            result,
            complete=False,
            initial_state=state,
            evidence_identity=identity,
        )


def test_atomic_interrupted_publication(tmp_path, monkeypatch, native, identity):
    import benchmarks.bankers_viability_shield_v4.serialization as serialization

    def interrupted(*_):
        raise OSError("injected interrupted atomic publication")

    rows, _, _, _ = native
    with monkeypatch.context() as m:
        m.setattr(serialization.os, "link", interrupted)
        with pytest.raises(OSError):
            write_evidence(tmp_path / "e.gz", rows, identity, rows[0]["attempt"])
    assert not list(tmp_path.iterdir())
    # Existing immutable evidence survives an interrupted duplicate publication.
    meta = write_evidence(tmp_path / "e.gz", rows, identity, rows[0]["attempt"])
    with monkeypatch.context() as m:
        m.setattr(serialization.os, "link", interrupted)
        with pytest.raises(OSError):
            write_evidence(tmp_path / "e.gz", rows, identity, rows[0]["attempt"])
    assert canonical_bytes(
        read_evidence(tmp_path / "e.gz", meta, identity, rows[0]["attempt"])
    ) == canonical_bytes(rows)


def test_replay_does_not_call_online_classifier(monkeypatch, native, identity):
    def forbidden(*_, **__):
        raise AssertionError("online classifier called by independent replay")

    monkeypatch.setattr(FastBankersShield, "certify", forbidden)
    monkeypatch.setattr(CompactShield, "certify", forbidden)
    rows, result, track, state = native
    replay(
        rows,
        track,
        "BANKERS_MASKED_CAUSAL60",
        result,
        complete=False,
        initial_state=state,
        evidence_identity=identity,
    )


def test_atomic_process_crash(tmp_path):
    import subprocess
    import sys

    target = tmp_path / "never-published.json"
    code = """
import os, sys
from benchmarks.bankers_viability_shield_v4.serialization import atomic_json
os.link = lambda *args: os._exit(73)
atomic_json(sys.argv[1], {"crash": "before publication"}, immutable=True)
"""
    result = subprocess.run([sys.executable, "-c", code, str(target)])
    assert result.returncode == 73 and not target.exists()
    # An orphan temporary file is retained after a real crash and counted on disk.
    assert len(list(tmp_path.iterdir())) == 1


def test_immutable_lifecycle_and_timestamp(tmp_path):
    key = "V4_UNIT_LIFECYCLE"
    ledger = PairLedger(tmp_path / "ledger.json", [key], {})
    ledger.start(key)
    with pytest.raises(EvidenceError):
        PairLedger(tmp_path / "ledger.json", [key], {})
    ledger.complete(key, {"passed": True})
    with pytest.raises(EvidenceError):
        ledger.start(key)
    PairLedger(tmp_path / "ledger.json", [key], {})
    marker = tmp_path / "started" / (key + ".json")
    value = strict_load(marker.read_bytes())
    value["started_unix"] += 1
    atomic_json(marker, value)
    with pytest.raises(EvidenceError):
        PairLedger(tmp_path / "ledger.json", [key], {})


def test_lock_and_paper_guard(tmp_path, monkeypatch):
    with exclusive_writer(tmp_path):
        with pytest.raises(FileExistsError):
            with exclusive_writer(tmp_path):
                pass

    def forbidden(_):
        raise AssertionError("Paper construction forbidden")

    monkeypatch.setattr(
        "benchmarks.content_split_audit.streams.track_from_seed", forbidden
    )
    for seed in range(4000, 4018):
        with pytest.raises(EvidenceError):
            guarded_track(seed)


def test_canonical_missing_data():
    assert strict_load(canonical_bytes({"x": -0.0}))["x"].hex() == (-0.0).hex()
    for invalid in (b'{"x":NaN}', b'{"x":1,"x":2}'):
        with pytest.raises(EvidenceError):
            strict_load(invalid)
