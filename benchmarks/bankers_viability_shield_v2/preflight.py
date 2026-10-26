"""V2 synthetic-only adversarial fixtures and complete runtime projection."""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np

from benchmarks.bankers_viability_shield_v1.fixtures import fixtures, make_track
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from .evaluation import run
from .independent_verification import certificate_from_physics, replay, verify_envelope
from .inherited_shield_adapter import (
    ACTION_GRID,
    CONFIG,
    MIN_LIMIT,
    VELOCITY_MARGIN,
    BankersShield,
    CausalMemory,
)
from .integrity import (
    HERE,
    PairLedger,
    exclusive_writer,
    source_identity,
    verify_sources,
)
from .serialization import (
    atomic_json,
    canonical_bytes,
    read_evidence,
    require,
    write_evidence,
)


def search():
    vehicle = VehicleModel()
    shield = BankersShield(vehicle)
    rng = np.random.default_rng(2026102602)
    admitted = proposals = 0
    fixture_counts = {}
    for name, track in fixtures().items():
        checked = 0
        for _ in range(40):
            x = float(rng.uniform(0, 999))
            limit = track.sense(x, 150).current_limit_m_s
            state = VehicleState(
                position_m=x, velocity_m_s=float(rng.uniform(0, limit - 2e-6))
            )
            ordinary = observation(
                state, track.sense(x, 150), vehicle.specs, CONFIG, 1.0
            )
            memory = CausalMemory(state=state)
            memory.ingest(ordinary)
            for action in ACTION_GRID:
                proposals += 1
                cert = shield.certify(ordinary, memory, action)
                rebuilt = certificate_from_physics(
                    ordinary, memory, action, vehicle, shield.identity
                )
                require(
                    canonical_bytes(cert.as_dict()) == canonical_bytes(rebuilt),
                    "search independent certificate",
                )
                if not cert.accepted:
                    continue
                admitted += 1
                cursor, _, _ = advance(state, action, vehicle, CONFIG)
                path = [asdict(cursor)]
                while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
                    cursor, _, _ = advance(cursor, -1.0, vehicle, CONFIG)
                    path.append(asdict(cursor))
                bad = [
                    p
                    for p in path
                    if p["velocity_m_s"]
                    > track.sense(p["position_m"], 150).current_limit_m_s
                ]
                if bad:
                    atomic_json(
                        HERE / "counterexample.json",
                        {
                            "state": asdict(state),
                            "action": action,
                            "observation": ordinary,
                            "memory": memory.snapshot(),
                            "certificate": cert.as_dict(),
                            "path": path,
                            "first_violation": bad[0],
                        },
                        immutable=True,
                    )
                    raise RuntimeError("BANKERS_V2_COUNTEREXAMPLE_FOUND")
                following = CausalMemory.restore(memory.snapshot())
                following.execute(action, vehicle)
                new_obs = observation(
                    following.state,
                    track.sense(following.state.position_m, 150),
                    vehicle.specs,
                    CONFIG,
                    1.0,
                )
                following.ingest(new_obs)
                verify_envelope(following, memory, track)
                require(
                    shield.certify(new_obs, following, -1.0).accepted,
                    "inductive backup",
                )
                checked += 1
        fixture_counts["V2_SEARCH__" + name] = checked
    # Reproduce the retained negative control without re-running a V1 pair.
    state = VehicleState(velocity_m_s=25.6625)
    track = make_track([0, 150.000001], [100, 20])
    ordinary = observation(state, track.sense(0, 150), vehicle.specs, CONFIG, 1.0)
    memory = CausalMemory(state=state)
    memory.ingest(ordinary)
    cert = shield.certify(ordinary, memory, 1.0)
    cursor, _, _ = advance(state, 1.0, vehicle, CONFIG)
    successor = cursor
    first_bad = None
    while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
        cursor, _, _ = advance(cursor, -1.0, vehicle, CONFIG)
        if (
            cursor.velocity_m_s > track.sense(cursor.position_m, 150).current_limit_m_s
            and first_bad is None
        ):
            first_bad = asdict(cursor)
    require(first_bad is not None and not cert.accepted, "negative control reproduced")
    require(
        cursor.position_m - successor.position_m + 0.5 <= 150,
        "naive shortcut acceptance",
    )
    return {
        "identity": "V2_ADVERSARIAL_SEARCH_2026102602",
        "states": 320,
        "proposals": proposals,
        "admitted_backup_and_induction_checks": admitted,
        "core_counterexamples": 0,
        "fixture_counts": fixture_counts,
        "naive_refreshed_preview_negative_control": {
            "pre_state": asdict(state),
            "proposed_action": 1.0,
            "first_violation": first_bad,
            "violation_m_s": first_bad["velocity_m_s"] - MIN_LIMIT,
            "correct_certificate_accepted": cert.accepted,
            "naive_backup_distance_plus_margin_m": cursor.position_m
            - successor.position_m
            + 0.5,
        },
    }


def microbenchmark():
    measurements = []
    for speed in (0.0, MIN_LIMIT, 10.0, 60 / 3.6, 20.0, 25.0, 25.5):
        # High-speed causal contexts on a constant-100 synthetic road.
        track = make_track([0], [100])
        state = VehicleState(velocity_m_s=speed)
        env = gym.wrappers.TimeLimit(
            LongiControlEnv(track_generator=lambda _: track), 1800
        )
        start_cpu = time.process_time()
        start = time.perf_counter()
        rows, _ = run(env, "BANKERS_GREEDY", max_steps=5, initial_state=state)
        online = time.perf_counter() - start
        replay_start = time.perf_counter()
        replay(rows, track, "BANKERS_GREEDY", complete=False, initial_state=state)
        replay_s = time.perf_counter() - replay_start
        # A complete disk write/read + semantic replay is measured on synthetic records.
        attempt = f"V2_RESOURCE__{speed.hex()}"
        path = HERE / "evidence" / (attempt + ".jsonl.gz")
        start = time.perf_counter()
        metadata = write_evidence(path, rows, source_identity(), attempt)
        recovered = read_evidence(path, metadata, source_identity(), attempt)
        require(
            canonical_bytes(recovered) == canonical_bytes(rows),
            "resource evidence readback",
        )
        io_s = time.perf_counter() - start
        measurements.append(
            {
                "speed_m_s": speed,
                "sampled_steps": len(rows),
                "mask_s_mean": sum(r["timing"]["mask_s"] for r in rows) / len(rows),
                "mask_s_max": max(r["timing"]["mask_s"] for r in rows),
                "mask_physics_calls_mean": sum(
                    r["timing"]["mask_physics_calls"] for r in rows
                )
                / len(rows),
                "final_s_mean": sum(r["timing"]["final_s"] for r in rows) / len(rows),
                "online_s_per_step": online / len(rows),
                "replay_s_per_step": replay_s / len(rows),
                "evidence_io_s_per_step": io_s / len(rows),
                "compressed_bytes_per_step": metadata["bytes"] / len(rows),
                "canonical_bytes_per_step": metadata["canonical_bytes"] / len(rows),
                "total_cpu_s": time.process_time() - start_cpu,
                "evidence": metadata,
                "memory_snapshot_max_bytes": max(
                    len(canonical_bytes(r["memory"])) for r in rows
                ),
            }
        )
        env.close()
    return measurements


def preflight():
    require(
        not (HERE / "preparation_status.json").exists(), "preflight already executed"
    )
    verify_sources()
    quality = json.loads((HERE / "quality_checks.json").read_text())
    require(
        quality["exit_code"] == 0 and quality["identity"] == source_identity(),
        "synthetic serialization/physical tests not passed under current source",
    )
    identity = source_identity()
    start_cpu = time.process_time()
    start = time.perf_counter()
    search_result = search()
    atomic_json(HERE / "counterexample_search.json", search_result, immutable=True)
    tracks = fixtures()
    tasks = [
        ("V2_FIXTURE__" + name, track, "BANKERS_GREEDY", None, 1800)
        for name, track in tracks.items()
    ]
    tasks.append(
        (
            "V2_FIXTURE__repeated_minimal",
            tracks["constant_100"],
            "V2_SYNTHETIC_MINIMAL",
            VehicleState(velocity_m_s=25.0),
            120,
        )
    )
    ledger = PairLedger(HERE / "fixture_ledger.json", [t[0] for t in tasks], identity)
    results = {}
    for attempt, track, method, state, steps in tasks:
        print("FIXTURE START", attempt, flush=True)
        ledger.start(attempt)
        env = gym.wrappers.TimeLimit(
            LongiControlEnv(track_generator=lambda _: track), 1800
        )
        pair_cpu = time.process_time()
        online_start = time.perf_counter()
        rows, result = run(env, method, max_steps=steps, initial_state=state)
        result["online_s"] = time.perf_counter() - online_start
        path = HERE / "evidence" / (attempt + ".jsonl.gz")
        io_start = time.perf_counter()
        metadata = write_evidence(path, rows, identity, attempt)
        disk_rows = read_evidence(path, metadata, identity, attempt)
        result["evidence_io_s"] = time.perf_counter() - io_start
        replay_start = time.perf_counter()
        replay(
            disk_rows,
            track,
            method,
            result if state is None else None,
            complete=steps == 1800,
            initial_state=state,
        )
        result["replay_s"] = time.perf_counter() - replay_start
        result["pair_cpu_s"] = time.process_time() - pair_cpu
        result["evidence"] = metadata
        result["independent_replay_passed"] = True
        require(
            all(r["speed_excess_m_s"] == 0 for r in rows),
            "BANKERS_V2_COUNTEREXAMPLE_FOUND",
        )
        ledger.complete(attempt, result)
        results[attempt] = result
        atomic_json(HERE / "fixture_results.json", results)
        print("FIXTURE DONE", attempt, result["steps"], result["metrics"], flush=True)
        env.close()
    measurements = microbenchmark()
    worst_step = max(
        m["online_s_per_step"] + m["replay_s_per_step"] + m["evidence_io_s_per_step"]
        for m in measurements
    )
    fixture_step = max(
        (r["online_s"] + r["replay_s"] + r["evidence_io_s"]) / r["steps"]
        for r in results.values()
    )
    # Predeclared conservative projection: all 72 episodes at 1800 steps;
    # all arms charged the most expensive observed high-speed/fixture workload.
    projected_development_s = 72 * 1800 * max(worst_step, fixture_step) * 1.25
    elapsed_cpu = time.process_time() - start_cpu + quality["active_cpu_s"]
    retained = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    max_bytes = max(
        [m["compressed_bytes_per_step"] for m in measurements]
        + [r["evidence"]["bytes"] / r["steps"] for r in results.values()]
    )
    projected_bytes = retained + 72 * 1800 * max_bytes * 1.25
    mask_cost = max(m["mask_s_mean"] + m["final_s_mean"] for m in measurements)
    projection = {
        "measurements": measurements,
        "fixture_results": results,
        "projection_steps": 72 * 1800,
        "overhead_factor": 1.25,
        "charged_worst_step_s": max(worst_step, fixture_step),
        "projected_development_s": projected_development_s,
        "preflight_cpu_s": elapsed_cpu,
        "preflight_wall_s": time.perf_counter() - start,
        "projected_total_active_s": elapsed_cpu + projected_development_s,
        "projected_retained_bytes": projected_bytes,
        "current_retained_bytes": retained,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "future_rl": {
            "seeds": 3,
            "native_steps_per_seed": 300000,
            "executed_steps": 0,
            "online_mask_and_final_s": 900000 * mask_cost,
            "two_masks_per_update_projection_s": 900000 * mask_cost * 2,
            "network_training_cost_included": False,
            "budget_s": 86400,
            "online_mask_only_within_24h": 900000 * mask_cost < 86400,
            "two_mask_cost_within_24h": 900000 * mask_cost * 2 < 86400,
        },
        "python": sys.version,
        "numpy": np.__version__,
    }
    atomic_json(HERE / "resource_measurements.json", projection, immutable=True)
    gates = {
        "formal_semantics": True,
        "serialization_synthetic_tests": True,
        "synthetic_native_safety": True,
        "independent_replay": True,
        "backup": True,
        "resource_compute": projection["projected_total_active_s"] <= 21600,
        "resource_storage": projected_bytes <= 3 * 1024**3,
    }
    prep = {
        "identity": identity,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "preflight_cpu_s": elapsed_cpu,
        "development_episodes": 0,
        "classification": "PREFLIGHT_PASSED"
        if all(gates.values())
        else "BANKERS_V2_RESOURCE_STOP",
        "training_transitions": 0,
        "gradient_updates": 0,
    }
    atomic_json(HERE / "preparation_status.json", prep, immutable=True)
    print(
        "PREFLIGHT",
        prep,
        "projected_s",
        projection["projected_total_active_s"],
        flush=True,
    )
    return prep


if __name__ == "__main__":
    with exclusive_writer():
        preflight()
