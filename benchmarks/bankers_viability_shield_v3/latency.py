"""Matched, uninstrumented old/new workloads; allocation probes kept separate."""

import time
import tracemalloc

import numpy as np

from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG, CausalMemory
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState

from .fast_certificate import FastBankersShield, fast_backup_witness, safety_view
from .fast_kinematics import fast_kinematic_successor
from .integrity import HERE
from .safe_action_mask import safe_action_mask
from .serialization import atomic_json, require


def summary(values):
    return dict(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        maximum=max(values),
        samples=len(values),
    )


def benchmark():
    start = time.process_time()
    old = BankersShield()
    fast = FastBankersShield()
    track = make_track([0], [100])
    rows = []
    for speed in (0.0, 20 / 3.6, 10.0, 60 / 3.6, 20.0, 25.0, 25.5):
        state = VehicleState(velocity_m_s=speed)
        memory = CausalMemory(state=state)
        obs = observation(state, track.sense(0, 150), old.vehicle.specs, CONFIG, 1.0)
        memory.ingest(obs)
        values = {
            k: []
            for k in (
                "frozen_mask_s",
                "fast_mask_s",
                "full_selected_s",
                "fast_backup_s",
                "mask_plus_final_s",
            )
        }
        for repeat in range(40):
            t = time.perf_counter()
            original = [old.certify(obs, memory, a).as_dict() for a in ACTION_GRID]
            old_s = time.perf_counter() - t
            t = time.perf_counter()
            new = safe_action_mask(fast, obs, memory)
            fast_s = time.perf_counter() - t
            require(
                [safety_view(c) for c in original] == [safety_view(c) for c in new],
                "latency equivalence",
            )
            action = max(c["proposed_action"] for c in new if c["accepted"])
            t = time.perf_counter()
            old.certify(obs, memory, action)
            selected = time.perf_counter() - t
            successor = fast_kinematic_successor(state, action, fast.vehicle)
            t = time.perf_counter()
            fast_backup_witness(successor, memory, fast.vehicle)
            backup = time.perf_counter() - t
            for k, v in zip(
                values, (old_s, fast_s, selected, backup, fast_s + selected)
            ):
                values[k].append(v)
        allocations = {}
        for label, fn in [
            (
                "frozen",
                lambda: [old.certify(obs, memory, a).as_dict() for a in ACTION_GRID],
            ),
            ("fast", lambda: safe_action_mask(fast, obs, memory)),
        ]:
            tracemalloc.start()
            before = tracemalloc.take_snapshot()
            retained = fn()
            after = tracemalloc.take_snapshot()
            allocations[label] = dict(
                peak_traced_bytes=tracemalloc.get_traced_memory()[1],
                retained_positive_allocations=sum(
                    max(0, s.count_diff) for s in after.compare_to(before, "lineno")
                ),
            )
            tracemalloc.stop()
            del retained
        rows.append(
            dict(
                speed_m_s=speed,
                position_m=0.0,
                repeats=40,
                measurements=values,
                summary={k: summary(v) for k, v in values.items()},
                allocations=allocations,
                kinematic_calls_per_mask=sum(
                    c["physics_prediction_count"] for c in new
                ),
                frozen_power_calls_per_mask=sum(
                    c["physics_prediction_count"] for c in new
                ),
                fast_power_calls_per_mask=0,
            )
        )
    result = dict(
        identity="V3_MATCHED_LATENCY_01",
        distribution="uniform 7 constant-100 synthetic states; 40 repetitions each",
        rows=rows,
        cpu_s=time.process_time() - start,
        hard_realtime_proven=False,
        overall={
            k: summary([v for r in rows for v in r["measurements"][k]]) for k in values
        },
    )
    atomic_json(HERE / "latency_results.json", result, immutable=True)
    print("LATENCY", result["overall"], flush=True)


if __name__ == "__main__":
    benchmark()
