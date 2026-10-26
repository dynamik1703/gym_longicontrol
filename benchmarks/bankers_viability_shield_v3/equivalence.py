"""Standalone frozen-V2 versus V3 comparison, synthetic identities only."""

import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG, CausalMemory
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v1.comparison import MeteredVehicle
from benchmarks.bankers_viability_shield_v1.fixtures import fixtures, make_track
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState

from .fast_certificate import FastBankersShield, safety_view
from .independent_verification import verify_compact
from .serialization import atomic_json, require

HERE = Path(__file__).resolve().parent


def test_states():
    rng = np.random.default_rng(2026102603)
    for name, track in fixtures().items():
        for x in (
            0.0,
            99.99999999,
            100.0,
            149.9999999,
            150.0,
            299.9999999,
            300.0,
            998.9,
            999.0,
            1000.0,
            1003.0,
        ):
            limit = track.sense(x, 150).current_limit_m_s
            for speed in (
                0.0,
                1e-14,
                limit - 1e-6,
                np.nextafter(limit - 1e-6, np.inf),
                min(37.0, limit + 0.01),
            ):
                yield (
                    "V3_DIFF__" + name,
                    track,
                    VehicleState(position_m=x, velocity_m_s=float(speed)),
                    False,
                )
        for _ in range(20):
            x = float(rng.uniform(0, 1000))
            limit = track.sense(x, 150).current_limit_m_s
            yield (
                "V3_DIFF_RANDOM__" + name,
                track,
                VehicleState(position_m=x, velocity_m_s=float(rng.uniform(0, limit))),
                False,
            )
    track = make_track([0], [100])
    for speed in (
        5.555554555555555,
        5.555555555555555,
        16.666666666666668,
        25.0,
        25.5,
        25.6625,
        27.77777777777778,
        37.0,
    ):
        yield "V3_DIFF_BOUNDARY", track, VehicleState(velocity_m_s=speed), False
    yield (
        "V3_DIFF_HISTORY",
        fixtures()["repeated_adversarial"],
        VehicleState(position_m=850.0, velocity_m_s=3.0),
        True,
    )


def differential():
    require(
        not (HERE / "equivalence_final.json").exists(),
        "differential run already sealed",
    )
    old_vehicle = MeteredVehicle()
    old = BankersShield(old_vehicle)
    fast = FastBankersShield()
    start = time.process_time()
    states = actions = admitted = 0
    old_calls = 0
    new_calls = 0
    rows = []
    counts = []
    for fixture, track, state, history in test_states():
        memory = CausalMemory(state=state)
        if history:
            for x in range(0, 850, 10):
                memory.state = replace(state, position_m=float(x))
                obs = observation(
                    memory.state, track.sense(x, 150), fast.vehicle.specs, CONFIG, 1.0
                )
                memory.ingest(obs)
            memory.state = state
        obs = observation(
            state, track.sense(state.position_m, 150), fast.vehicle.specs, CONFIG, 1.0
        )
        memory.ingest(obs)
        accepted = 0
        for action in ACTION_GRID:
            before = old_vehicle.predictions
            t = time.perf_counter()
            frozen = old.certify(obs, memory, action).as_dict()
            old_s = time.perf_counter() - t
            old_calls += old_vehicle.predictions - before
            t = time.perf_counter()
            current = fast.certify(obs, memory, action)
            new_s = time.perf_counter() - t
            if safety_view(frozen) != safety_view(current):
                atomic_json(
                    HERE / "first_equivalence_counterexample.json",
                    dict(
                        fixture=fixture,
                        state=asdict(state),
                        action=action,
                        observation=obs.tolist(),
                        memory=memory.snapshot(),
                        frozen=frozen,
                        optimized=current,
                    ),
                    immutable=True,
                )
                raise RuntimeError("BANKERS_V3_EQUIVALENCE_STOP")
            verify_compact(current, obs, memory, old_vehicle, old.identity)
            actions += 1
            accepted += int(current["accepted"])
            admitted += int(current["accepted"])
            new_calls += current["physics_prediction_count"]
            rows.append(
                dict(
                    state_index=states,
                    fixture=fixture,
                    speed_m_s=state.velocity_m_s,
                    action=action,
                    old_s=old_s,
                    fast_s=new_s,
                    accepted=current["accepted"],
                    rejection_reason=current["rejection_reason"],
                )
            )
        counts.append(accepted)
        states += 1
    invalid = []
    track = make_track([0], [100])
    state = VehicleState()
    obs = observation(state, track.sense(0, 150), fast.vehicle.specs, CONFIG, 1.0)
    memory = CausalMemory()
    memory.ingest(obs)
    for kind in (
        "nan_action",
        "inf_action",
        "range_action",
        "stale",
        "nan_state",
        "inf_energy",
        "huge_energy",
    ):
        m = CausalMemory.restore(memory.snapshot())
        a = 0.0
        o = obs.copy()
        if kind == "nan_action":
            a = float("nan")
        if kind == "inf_action":
            a = float("inf")
        if kind == "range_action":
            a = 1.1
        if kind == "stale":
            m.last_observation = None
        if kind == "nan_state":
            m.state = replace(state, velocity_m_s=float("nan"))
        if kind == "inf_energy":
            m.state = replace(state, total_energy_kwh=float("inf"))
        if kind == "huge_energy":
            m.state = replace(state, total_energy_kwh=1e308)
        f = old.certify(o, m, a).as_dict()
        c = fast.certify(o, m, a)
        require(safety_view(f) == safety_view(c), "invalid differential " + kind)
        invalid.append(
            dict(
                case=kind,
                accepted=c["accepted"],
                rejection_reason=c["rejection_reason"],
            )
        )
    # Original refreshed-preview counterexample, with a fresh V3 identity.
    state = VehicleState(velocity_m_s=25.6625)
    track = make_track([0, 150.000001], [100, 20])
    obs = observation(state, track.sense(0, 150), fast.vehicle.specs, CONFIG, 1.0)
    memory = CausalMemory(state=state)
    memory.ingest(obs)
    cert = fast.certify(obs, memory, 1.0)
    cursor, _, _ = advance(state, 1.0, fast.vehicle, CONFIG)
    successor = cursor
    first_bad = None
    from benchmarks.bankers_viability_shield_v1.backup_policy import (
        MIN_LIMIT,
        VELOCITY_MARGIN,
    )

    while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
        cursor, _, _ = advance(cursor, -1.0, fast.vehicle, CONFIG)
        if (
            cursor.velocity_m_s > track.sense(cursor.position_m, 150).current_limit_m_s
            and first_bad is None
        ):
            first_bad = asdict(cursor)
    require(
        first_bad is not None
        and not cert["accepted"]
        and cursor.position_m - successor.position_m + 0.5 <= 150,
        "negative control",
    )
    result = dict(
        identity="V3_DIFFERENTIAL_02",
        states=states,
        grid_actions=actions,
        mismatches=0,
        admitted=admitted,
        invalid_cases=invalid,
        accepted_action_counts=counts,
        frozen_full_model_calls=old_calls,
        fast_kinematic_calls=new_calls,
        fast_power_calls=0,
        all_transcripts_independently_verified=True,
        cpu_s=time.process_time() - start,
        measurements=rows,
        finiteness_audit=fast.finiteness,
        naive_preview_negative_control=dict(
            correctly_rejected=True, first_physical_violation=first_bad
        ),
    )
    atomic_json(HERE / "equivalence_final.json", result, immutable=True)
    print(
        "DIFFERENTIAL PASS",
        states,
        actions,
        "accepted",
        admitted,
        "cpu",
        result["cpu_s"],
        flush=True,
    )
    return result


if __name__ == "__main__":
    differential()
