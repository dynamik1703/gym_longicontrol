"""Freeze after preflight; execute exactly 72 pairs once, no recovery or learner."""

import argparse
import json
import subprocess
import time

import gymnasium as gym

from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .controller_adapters import METHODS
from .evaluation import CounterexampleFound, run
from .independent_verification import replay
from .integrity import (
    HERE,
    ROOT,
    PairLedger,
    build_split,
    exclusive_writer,
    guarded_track,
    require_committed,
    source_identity,
    verify_sources,
)
from .serialization import (
    atomic_json,
    file_hash,
    native_journal,
    read_evidence,
    require,
    write_evidence,
)


def freeze():
    prep = json.loads((HERE / "preparation_status.json").read_text())
    require(prep["all_gates_passed"], "Development preflight closed")
    verify_sources(prep["identity"])
    require_committed()
    split = build_split()
    attempts = [
        f"BANKERS_DEV_V4__{t['numeric_seed']}__{m}__ATTEMPT_01"
        for t in split["ordered_tasks"]
        for m in METHODS
    ]
    manifest = dict(
        identity=source_identity(),
        split_file_sha256=file_hash(HERE / "split_ledger.json"),
        ordered_panel_sha256=split["ordered_sha256"],
        attempts=attempts,
        methods=list(METHODS),
        maximum_episodes=72,
        no_recovery_protocol=True,
    )
    atomic_json(HERE / "execution_manifest.json", manifest, immutable=True)
    PairLedger(HERE / "pair_ledger.json", attempts, manifest)


def develop():
    require(not (HERE / "integrity_stop.json").exists(), "recorded integrity stop")
    prep = json.loads((HERE / "preparation_status.json").read_text())
    require(prep["all_gates_passed"], "Development closed")
    manifest = json.loads((HERE / "execution_manifest.json").read_text())
    verify_sources(manifest["identity"])
    commit = require_committed()
    require(
        file_hash(HERE / "split_ledger.json") == manifest["split_file_sha256"],
        "split mutation",
    )
    for name in ("execution_manifest.json", "split_ledger.json"):
        require(
            subprocess.check_output(
                ["git", "show", f"HEAD:{(HERE / name).relative_to(ROOT)}"], cwd=ROOT
            )
            == (HERE / name).read_bytes(),
            "uncommitted panel",
        )
    split = json.loads((HERE / "split_ledger.json").read_text())
    ledger = PairLedger(HERE / "pair_ledger.json", manifest["attempts"], manifest)
    prior = (
        prep["preflight_cpu_s"]
        + prep["prior_measured_cpu_s"]
        + prep["preparation_reporting_allowance_s"]
        + sum(
            x.get("result", {}).get("pair_cpu_s", 0)
            for x in ledger.data["pairs"].values()
        )
    )
    start = time.process_time()
    counter = 0
    maximum_disk = 0
    source_check_cpu = 0.0
    inherited_bytes = prep["inherited_retained_bytes"] + prep["native_source_bytes"]

    def budget(extra_transient_bytes=0):
        nonlocal counter, maximum_disk
        require(
            prior + time.process_time() - start <= 21600,
            "BANKERS_V4_RESOURCE_STOP: active compute",
        )
        if counter % 100 == 0 or extra_transient_bytes:
            current = (
                inherited_bytes
                + extra_transient_bytes
                + sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
            )
            maximum_disk = max(maximum_disk, current)
            atomic_json(
                HERE / "budget_monitor.json",
                dict(
                    peak_observed_total_bytes=maximum_disk,
                    active_cpu_s=prior + time.process_time() - start,
                    peak_rss_bytes_macos=__import__("resource")
                    .getrusage(__import__("resource").RUSAGE_SELF)
                    .ru_maxrss,
                ),
            )
            require(current <= 3 * 1024**3, "BANKERS_V4_RESOURCE_STOP: storage")
        counter += 1

    for task in split["ordered_tasks"]:
        seed = task["numeric_seed"]
        track = guarded_track(seed)
        fp = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        require(
            fp.route_sha256 == task["route_sha256"]
            and fp.full_task_sha256 == task["full_task_sha256"],
            "task identity",
        )
        for method in METHODS:
            attempt = f"BANKERS_DEV_V4__{seed}__{method}__ATTEMPT_01"
            if ledger.data["pairs"][attempt]["state"] == "COMPLETED":
                meta = ledger.data["pairs"][attempt]["result"]["evidence"]
                read_evidence(
                    HERE / "evidence" / meta["file"],
                    meta,
                    manifest["identity"],
                    attempt,
                )
                continue
            budget()
            check_start = time.process_time()
            verify_sources(manifest["identity"])
            source_check_cpu += time.process_time() - check_start
            ledger.start(attempt)
            print("PAIR START", attempt, flush=True)
            cpu = time.process_time()
            online = time.perf_counter()
            env = gym.wrappers.TimeLimit(StochasticTrack(), 1800)
            journal_path = HERE / "journals" / f"{attempt}.jsonl"
            with native_journal(journal_path, manifest["identity"], attempt) as sink:
                rows, result = run(
                    env,
                    method,
                    seed=seed,
                    budget=budget,
                    record_sink=sink,
                    evidence_identity=manifest["identity"],
                    attempt=attempt,
                )
            result["online_wall_s"] = time.perf_counter() - online
            meta = write_evidence(
                HERE / "evidence" / f"{attempt}.gz", rows, manifest["identity"], attempt
            )
            disk = read_evidence(
                HERE / "evidence" / f"{attempt}.gz", meta, manifest["identity"], attempt
            )
            t = time.perf_counter()
            replay(
                disk,
                track,
                method,
                result,
                evidence_identity=manifest["identity"],
                attempt=attempt,
            )
            result.update(
                independent_replay_s=time.perf_counter() - t,
                pair_cpu_s=time.process_time() - cpu,
                evidence=meta,
                task=task,
                method=method,
                freeze_commit=commit,
                independent_replay_passed=True,
            )
            verify_sources(manifest["identity"])
            budget(meta["bytes"])
            ledger.complete(attempt, result)
            journal_path.unlink()
            env.close()
            print(
                "PAIR DONE", attempt, result["feasible"], result["metrics"], flush=True
            )
    require(
        all(p["state"] == "COMPLETED" for p in ledger.data["pairs"].values()),
        "incomplete matrix",
    )
    verify_sources(manifest["identity"])
    atomic_json(
        HERE / "development_seal.json",
        dict(
            ledger_sha256=file_hash(HERE / "pair_ledger.json"),
            completed_pairs=72,
            source_verification_cpu_s=source_check_cpu,
            peak_observed_total_bytes=maximum_disk,
            accounted_active_compute_s=prior + time.process_time() - start,
            identity=manifest["identity"],
            freeze_commit=commit,
        ),
        immutable=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "develop"))
    args = parser.parse_args()
    try:
        with exclusive_writer():
            freeze() if args.mode == "freeze" else develop()
    except Exception as error:
        classification = (
            "BANKERS_V4_COUNTEREXAMPLE_FOUND"
            if isinstance(error, CounterexampleFound)
            else "BANKERS_V4_RESOURCE_STOP"
            if "RESOURCE_STOP" in str(error)
            else "BANKERS_V4_INTEGRITY_STOP"
        )
        if not (HERE / "integrity_stop.json").exists():
            atomic_json(
                HERE / "integrity_stop.json",
                dict(
                    classification=classification,
                    error=str(error),
                    exception_type=type(error).__name__,
                    mode=args.mode,
                    rerun_forbidden=True,
                ),
                immutable=True,
            )
        raise


if __name__ == "__main__":
    main()
