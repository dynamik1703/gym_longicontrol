"""Deterministic workloads; measured time is not deterministic."""

import cProfile
import gzip
import hashlib
import json
import pstats
import time
from pathlib import Path

import numpy as np

from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG, CausalMemory
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v1.comparison import (
    MeteredVehicle,
    inherited_test,
)
from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v1.safe_action_set import (
    ACTION_GRID,
    SafeActionSet,
)
from benchmarks.bankers_viability_shield_v2.independent_verification import (
    certificate_from_physics,
)
from benchmarks.bankers_viability_shield_v2.serialization import canonical_bytes
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState

OUT = Path("/private/tmp/bankers_v3_baseline_profile.json")
assert not OUT.exists()
start_cpu = time.process_time()
vehicle = MeteredVehicle()
shield = BankersShield(vehicle)
track = make_track([0], [100])
measurements = []
profile = cProfile.Profile()
for speed in (0.0, 20 / 3.6, 10.0, 60 / 3.6, 20.0, 25.0, 25.5, 100 / 3.6):
    state = VehicleState(velocity_m_s=speed)
    obs = observation(state, track.sense(0, 150), vehicle.specs, CONFIG, 1.0)
    memory = CausalMemory(state=state)
    start = time.perf_counter()
    memory.ingest(obs)
    ingest = time.perf_counter() - start
    samples = []
    for repeat in range(5):
        before = vehicle.predictions
        profile.enable()
        start = time.perf_counter()
        menu = SafeActionSet(tuple(shield.certify(obs, memory, a) for a in ACTION_GRID))
        mask = time.perf_counter() - start
        start = time.perf_counter()
        final = shield.certify(obs, memory, max(menu.actions) if menu.actions else -1.0)
        final_s = time.perf_counter() - start
        profile.disable()
        online_calls = vehicle.predictions - before
        start = time.perf_counter()
        rebuilt = [
            certificate_from_physics(obs, memory, a, vehicle, shield.identity)
            for a in ACTION_GRID
        ]
        replay = time.perf_counter() - start
        assert [c.as_dict() for c in menu.certificates] == rebuilt
        start = time.perf_counter()
        payload = canonical_bytes(rebuilt)
        serialization = time.perf_counter() - start
        start = time.perf_counter()
        hashlib.sha256(payload).hexdigest()
        hashing = time.perf_counter() - start
        start = time.perf_counter()
        compressed = gzip.compress(payload, mtime=0)
        compression = time.perf_counter() - start
        start = time.perf_counter()
        inherited_test(state, 1.0, 100 / 3.6, (), -1.0, vehicle)
        legacy_proposal = time.perf_counter() - start
        samples.append(
            dict(
                mask_s=mask,
                selected_s=final_s,
                replay_s=replay,
                serialization_s=serialization,
                hashing_s=hashing,
                compression_s=compression,
                legacy_proposal_s=legacy_proposal,
                mask_physics_calls=menu.physics_prediction_count,
                mask_power_calls=menu.physics_prediction_count,
                online_physics_calls=online_calls,
                compressed_bytes=len(compressed),
            )
        )
    measurements.append(dict(speed_m_s=speed, causal_ingest_s=ingest, samples=samples))
stats = pstats.Stats(profile)
costs = []
for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
    if any(
        x in file
        for x in ("bankers", "dynamics.py", "vehicle.py", "dataclasses.py", "json/")
    ):
        costs.append(
            dict(
                file=file,
                line=line,
                function=name,
                primitive_calls=cc,
                calls=nc,
                self_s=tt,
                cumulative_s=ct,
            )
        )
result = dict(
    identity="V3_PROFILE_FROZEN_V2_01",
    source_commit="5030be8f9163b6909808626382f4a0f84c9b5d1e",
    workload="8 fixed speeds, constant 100 km/h, 5 repeats, full 21 actions",
    measurements=measurements,
    cprofile=sorted(costs, key=lambda x: -x["cumulative_s"]),
    cpu_s=time.process_time() - start_cpu,
    numpy=np.__version__,
    development_tasks_constructed=0,
)
OUT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
print("BASELINE PROFILE SEALED", OUT, "cpu_s", result["cpu_s"])
for row in measurements:
    print(
        row["speed_m_s"],
        "mask_ms",
        np.mean([s["mask_s"] for s in row["samples"]]) * 1000,
        "calls",
        row["samples"][0]["mask_physics_calls"],
    )
