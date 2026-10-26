"""Exactly-once bounded Development execution; no RL implementation."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time

import gymnasium as gym

from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .evaluation import CounterexampleFound, run
from .independent_verification import replay
from .inherited_shield_adapter import METHODS
from .integrity import (
    HERE,
    ROOT,
    PairLedger,
    build_split,
    exclusive_writer,
    source_identity,
    verify_sources,
)
from .serialization import (
    atomic_json,
    file_hash,
    read_evidence,
    require,
    write_evidence,
)


def freeze():
    prep = json.loads((HERE / "preparation_status.json").read_text())
    require(prep["all_gates_passed"], "Development preflight closed")
    identity = source_identity()
    verify_sources(prep["identity"])
    split = build_split()
    attempts = [
        f"BANKERS_DEV_V2__{t['numeric_seed']}__{m}__ATTEMPT_01"
        for t in split["ordered_tasks"]
        for m in METHODS
    ]
    manifest = {
        "identity": identity,
        "split_file_sha256": file_hash(HERE / "split_ledger.json"),
        "ordered_panel_sha256": split["ordered_sha256"],
        "attempts": attempts,
        "methods": list(METHODS),
        "maximum_episodes": 72,
        "no_recovery_protocol": True,
    }
    atomic_json(HERE / "execution_manifest.json", manifest, immutable=True)
    PairLedger(HERE / "pair_ledger.json", attempts, manifest)
    return manifest


def develop():
    require(not (HERE / "integrity_stop.json").exists(), "recorded integrity stop")
    prep = json.loads((HERE / "preparation_status.json").read_text())
    require(prep["all_gates_passed"], "Development preflight closed")
    manifest = json.loads((HERE / "execution_manifest.json").read_text())
    require(
        file_hash(HERE / "split_ledger.json") == manifest["split_file_sha256"],
        "split mutation",
    )
    verify_sources(manifest["identity"])
    # Freeze must be committed before the first environment reset.
    for name in (
        "execution_manifest.json",
        "canonical.json",
        "DEVELOPMENT_PROTOCOL.md",
    ):
        tracked = subprocess.check_output(
            ["git", "show", f"HEAD:{(HERE / name).relative_to(ROOT)}"], cwd=ROOT
        )
        require(
            tracked == (HERE / name).read_bytes(),
            "protocol not committed before execution",
        )
    freeze_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    split = json.loads((HERE / "split_ledger.json").read_text())
    ledger = PairLedger(HERE / "pair_ledger.json", manifest["attempts"], manifest)
    prior_cpu = prep["preflight_cpu_s"] + sum(
        v.get("result", {}).get("pair_cpu_s", 0) for v in ledger.data["pairs"].values()
    )
    cpu = time.process_time()

    def budget():
        require(
            prior_cpu + time.process_time() - cpu <= 21600,
            "BANKERS_V2_RESOURCE_STOP: active compute",
        )
        require(
            sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
            <= 3 * 1024**3,
            "BANKERS_V2_RESOURCE_STOP: storage",
        )

    for task in split["ordered_tasks"]:
        seed = task["numeric_seed"]
        require(seed not in PAPER_SEEDS, "Paper sealed")
        track = track_from_seed(seed)
        fp = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        require(
            fp.route_sha256 == task["route_sha256"]
            and fp.full_task_sha256 == task["full_task_sha256"],
            "task identity",
        )
        for method in METHODS:
            attempt = f"BANKERS_DEV_V2__{seed}__{method}__ATTEMPT_01"
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
            verify_sources(manifest["identity"])
            print("PAIR START", attempt, flush=True)
            ledger.start(attempt)
            pair_cpu = time.process_time()
            env = gym.wrappers.TimeLimit(StochasticTrack(), 1800)
            try:
                rows, result = run(env, method, seed=seed, budget=budget)
            except CounterexampleFound as error:
                write_evidence(
                    HERE / "evidence" / (attempt + "__COUNTEREXAMPLE.jsonl.gz"),
                    error.rows,
                    manifest["identity"],
                    attempt,
                )
                raise
            path = HERE / "evidence" / (attempt + ".jsonl.gz")
            result["evidence"] = write_evidence(
                path, rows, manifest["identity"], attempt
            )
            disk_rows = read_evidence(
                path, result["evidence"], manifest["identity"], attempt
            )
            replay_start = time.perf_counter()
            replay(disk_rows, track, method, result)
            result.update(
                independent_replay_s=time.perf_counter() - replay_start,
                pair_cpu_s=time.process_time() - pair_cpu,
                task=task,
                method=method,
                freeze_commit=freeze_commit,
            )
            ledger.complete(attempt, result)
            env.close()
            print(
                "PAIR DONE", attempt, result["feasible"], result["metrics"], flush=True
            )
    verify_sources(manifest["identity"])
    atomic_json(
        HERE / "development_seal.json",
        {
            "ledger_sha256": file_hash(HERE / "pair_ledger.json"),
            "completed_pairs": 72,
            "active_compute_s": prior_cpu + time.process_time() - cpu,
            "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "identity": manifest["identity"],
            "freeze_commit": freeze_commit,
        },
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
        if not (HERE / "integrity_stop.json").exists():
            classification = (
                "BANKERS_V2_COUNTEREXAMPLE_FOUND"
                if "COUNTEREXAMPLE_FOUND" in str(error)
                else "BANKERS_V2_RESOURCE_STOP"
                if "RESOURCE_STOP" in str(error)
                else "BANKERS_V2_INTEGRITY_STOP"
            )
            atomic_json(
                HERE / "integrity_stop.json",
                {
                    "classification": classification,
                    "error": str(error),
                    "type": type(error).__name__,
                    "mode": args.mode,
                    "started_pair_rerun_forbidden": True,
                },
                immutable=True,
            )
        raise


if __name__ == "__main__":
    main()
