"""Frozen resource formula, complete inherited synthetic fixtures and native replay."""

import json
import resource
import time

import gymnasium as gym

from benchmarks.bankers_viability_shield_v1.fixtures import fixtures, make_track
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from .controller_adapters import METHODS
from .evaluation import run
from .independent_verification import replay
from .integrity import (
    HERE,
    PairLedger,
    exclusive_writer,
    require_committed,
    source_identity,
    verify_sources,
)
from .serialization import (
    atomic_json,
    native_journal,
    read_evidence,
    require,
    write_evidence,
)


def benchmark(identity):
    measurements = []
    attempts = [
        f"V4_RESOURCE__{m}__{v.hex()}"
        for m in METHODS
        for v in (0.0, 20 / 3.6, 10.0, 60 / 3.6, 20.0, 25.0, 25.5)
    ]
    ledger = PairLedger(HERE / "resource_ledger.json", attempts, identity)
    for method in METHODS:
        for speed in (0.0, 20 / 3.6, 10.0, 60 / 3.6, 20.0, 25.0, 25.5):
            track = make_track([0], [100])
            state = VehicleState(velocity_m_s=speed)
            env = gym.wrappers.TimeLimit(
                LongiControlEnv(track_generator=lambda _: track), 1800
            )
            attempt = f"V4_RESOURCE__{method}__{speed.hex()}"
            journal_path = HERE / "journals" / f"{attempt}.jsonl"
            ledger.start(attempt)
            start = time.perf_counter()
            with native_journal(journal_path, identity, attempt) as sink:
                rows, result = run(
                    env,
                    method,
                    max_steps=5,
                    initial_state=state,
                    record_sink=sink,
                    evidence_identity=identity,
                    attempt=attempt,
                )
            online = time.perf_counter() - start
            start = time.perf_counter()
            meta = write_evidence(
                HERE / "evidence" / f"{attempt}.gz", rows, identity, attempt
            )
            disk = read_evidence(
                HERE / "evidence" / f"{attempt}.gz", meta, identity, attempt
            )
            io = time.perf_counter() - start
            start = time.perf_counter()
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
            replay_s = time.perf_counter() - start
            measurements.append(
                dict(
                    method=method,
                    speed_m_s=speed,
                    steps=len(rows),
                    online_s_per_step=online / len(rows),
                    replay_s_per_step=replay_s / len(rows),
                    evidence_io_s_per_step=io / len(rows),
                    raw_journal_bytes_per_step=journal_path.stat().st_size / len(rows),
                    compressed_bytes_per_step=meta["bytes"] / len(rows),
                    evidence=meta,
                    mask_s=[r["timing"]["mask_s"] for r in rows],
                    final_s=[r["timing"]["final_s"] for r in rows],
                    safety_overhead_s=[
                        r["timing"]["mask_s"] + r["timing"]["final_s"] for r in rows
                    ],
                    speed_distribution_m_s=[
                        r["causal_pre_state"]["velocity_m_s"] for r in rows
                    ],
                    kinematic_calls=[r["timing"]["mask_physics_calls"] for r in rows],
                )
            )
            ledger.complete(
                attempt,
                dict(evidence=meta, result=result, independent_replay_passed=True),
            )
            env.close()
            journal_path.unlink()
    return measurements


def preflight():
    require(not (HERE / "preparation_status.json").exists(), "preflight already sealed")
    verify_sources()
    commit = require_committed()
    identity = source_identity()
    quality = json.loads((HERE / "quality_checks.json").read_text())
    require(
        quality["exit_code"] == 0 and quality["identity"] == identity,
        "current tests not passed",
    )
    eq = json.loads((HERE / "equivalence_results.json").read_text())
    require(eq["mismatches"] == 0, "equivalence gate")
    start = time.process_time()
    results = {}
    tasks = [
        ("V4_FIXTURE__" + name, track, "BANKERS_GREEDY", None, 1800)
        for name, track in fixtures().items()
    ]
    tasks.append(
        (
            "V4_FIXTURE__repeated_minimal",
            fixtures()["constant_100"],
            "V4_SYNTHETIC_MINIMAL",
            VehicleState(velocity_m_s=25.0),
            120,
        )
    )
    ledger = PairLedger(HERE / "fixture_ledger.json", [t[0] for t in tasks], identity)
    for attempt, track, method, state, steps in tasks:
        ledger.start(attempt)
        print("FIXTURE START", attempt, flush=True)
        env = gym.wrappers.TimeLimit(
            LongiControlEnv(track_generator=lambda _: track), 1800
        )
        t = time.process_time()
        journal_path = HERE / "journals" / f"{attempt}.jsonl"
        with native_journal(journal_path, identity, attempt) as sink:
            rows, result = run(
                env,
                method,
                max_steps=steps,
                initial_state=state,
                record_sink=sink,
                evidence_identity=identity,
                attempt=attempt,
            )
        meta = write_evidence(
            HERE / "evidence" / f"{attempt}.gz", rows, identity, attempt
        )
        disk = read_evidence(
            HERE / "evidence" / f"{attempt}.gz", meta, identity, attempt
        )
        replay(
            disk,
            track,
            method,
            result,
            complete=steps == 1800,
            initial_state=state,
            evidence_identity=identity,
            attempt=attempt,
        )
        result.update(
            evidence=meta, independent_replay_passed=True, cpu_s=time.process_time() - t
        )
        ledger.complete(attempt, result)
        results[attempt] = result
        env.close()
        journal_path.unlink()
        print("FIXTURE DONE", attempt, result["steps"], result["metrics"], flush=True)
    atomic_json(HERE / "fixture_results.json", results, immutable=True)
    measurements = benchmark(identity)
    atomic_json(HERE / "resource_measurements.json", measurements, immutable=True)
    # Formula fixed in canonical.json and RESOURCE_PROTOCOL.md, committed above.
    per_method = {
        m: max(
            x["online_s_per_step"]
            + x["replay_s_per_step"]
            + x["evidence_io_s_per_step"]
            for x in measurements
            if x["method"] == m
        )
        for m in METHODS
    }
    preflight_cpu = time.process_time() - start
    prior_cpu = quality["cpu_s"]
    projection = (
        1.5 * 18 * 1800 * sum(per_method.values()) + preflight_cpu + prior_cpu + 900.0
    )
    retained = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    journal_bound = 1800 * max(x["raw_journal_bytes_per_step"] for x in measurements)
    inherited = sum(
        f.stat().st_size
        for folder in (
            HERE.parent / ("bankers_viability_shield_v" + str(i)) for i in (1, 2, 3)
        )
        for f in folder.rglob("*")
        if f.is_file() and "__pycache__" not in f.parts
    )
    native_sources = sum(
        f.stat().st_size
        for f in (HERE.parents[1] / "src").rglob("*")
        if f.is_file() and "__pycache__" not in f.parts
    )
    temporary_bound = 1800 * max(x["compressed_bytes_per_step"] for x in measurements)
    disk_projection = (
        retained
        + inherited
        + native_sources
        + temporary_bound
        + 1.5
        * 18
        * 1800
        * sum(
            max(
                x["compressed_bytes_per_step"] for x in measurements if x["method"] == m
            )
            for m in METHODS
        )
        + journal_bound
        + 16 * 1024**2
    )
    passed = projection <= 21600 and disk_projection <= 3 * 1024**3
    prep = dict(
        all_gates_passed=passed,
        identity=identity,
        freeze_commit=commit,
        classification="PREFLIGHT_PASS" if passed else "BANKERS_V4_RESOURCE_STOP",
        method_worst_step_s=per_method,
        projected_active_compute_s=projection,
        projected_retained_bytes=disk_projection,
        preflight_cpu_s=preflight_cpu,
        prior_measured_cpu_s=prior_cpu,
        preparation_reporting_allowance_s=900.0,
        measured_retained_bytes=retained,
        inherited_retained_bytes=inherited,
        native_source_bytes=native_sources,
        live_journal_bound_bytes=journal_bound,
        atomic_temporary_bound_bytes=temporary_bound,
        method_compressed_bytes_per_step={
            m: max(
                x["compressed_bytes_per_step"] for x in measurements if x["method"] == m
            )
            for m in METHODS
        },
        method_raw_journal_bytes_per_step={
            m: max(
                x["raw_journal_bytes_per_step"]
                for x in measurements
                if x["method"] == m
            )
            for m in METHODS
        },
        peak_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        development_tasks_constructed=0,
        rl_transitions=0,
        gradient_updates=0,
    )
    verify_sources(identity)
    atomic_json(HERE / "preparation_status.json", prep, immutable=True)
    print("PREFLIGHT", json.dumps(prep), flush=True)


if __name__ == "__main__":
    try:
        with exclusive_writer():
            preflight()
    except Exception as error:
        if not (HERE / "integrity_stop.json").exists():
            atomic_json(
                HERE / "integrity_stop.json",
                dict(
                    classification="BANKERS_V4_COUNTEREXAMPLE_FOUND"
                    if "COUNTEREXAMPLE" in str(error)
                    else "BANKERS_V4_INTEGRITY_STOP",
                    error=str(error),
                    stage="preflight",
                    rerun_forbidden=True,
                ),
                immutable=True,
            )
        raise
