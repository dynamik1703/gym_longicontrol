"""Bounded, exactly-once non-learning evaluation. Never trains a policy."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import resource
import statistics
import time
from collections import Counter
from dataclasses import asdict

import gymnasium as gym

from benchmarks.hybrid_rl_design.controller import _ExternalExecutionAdapter
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.envs.longicontrol import LongiControlEnv, StochasticTrack

from .backup_policy import CausalMemory, SafetyIntegrityError
from .certificate import HERE, BankersShield, digest, identities
from .comparison import MeteredVehicle, inherited_test
from .counterexample_search import search
from .fixtures import fixtures
from .integrity import (
    PairLedger,
    atomic_json,
    build_split,
    exclusive_writer,
    file_sha256,
    verify_sources,
    verify_task,
)
from .safe_action_set import ACTION_GRID, SafeActionSet
from .verification import replay_records, require, verify_certificate

ARTIFACTS = HERE / "evidence"


def summarize(rows, metrics, feasible, query_times, query_predictions, old_times):
    counts = Counter(sum(row["safe_action_mask"]) for row in rows)
    distinct = Counter(row["distinct_successors"] for row in rows)
    spans = [row["velocity_span_m_s"] for row in rows]
    disagreements = Counter()
    for row in rows:
        for new, old in zip(row["safe_action_mask"], row["inherited_comparison"]):
            label = (
                "BOTH_ACCEPT"
                if new and old["accepted"]
                else "BANKERS_ONLY"
                if new
                else "OLD_ONLY"
                if old["accepted"]
                else "BOTH_REJECT"
            )
            disagreements[label] += 1
            if (
                new
                and not old["accepted"]
                and old["right_endpoint"]
                and old["known_braking"]
                and old["unknown_preview"]
            ):
                disagreements["BANKERS_ONLY_DEADLINE"] += 1
    return {
        "steps": len(rows),
        "metrics": asdict(metrics),
        "feasible": feasible,
        "completion": metrics.completed,
        "deadline_compliant": metrics.completed and metrics.travel_time_s <= 140,
        "exact_speed_compliant": metrics.max_speed_violation_m_s == 0,
        "safe_action_count_distribution": dict(counts),
        "distinct_successor_distribution": dict(distinct),
        "velocity_span_m_s_mean": statistics.mean(spans),
        "meaningful_choices_steps": sum(
            row["distinct_successors"] >= 2 and row["velocity_span_m_s"] >= 0.05
            for row in rows
        ),
        "admitted_actions": sum(sum(row["safe_action_mask"]) for row in rows),
        "candidate_actions": 21 * len(rows),
        "rejected_positive_grid_actions": sum(
            sum(not row["safe_action_mask"][i] for i in range(11, 21)) for row in rows
        ),
        "positive_grid_actions": 10 * len(rows),
        "backup_available_steps": sum(row["backup_available"] for row in rows),
        "positive_acceleration_available_steps": sum(
            row["positive_acceleration_available"] for row in rows
        ),
        "proposal_executed_difference_steps": sum(
            row["proposal_action"] != row["executed_action"] for row in rows
        ),
        "mask_seconds_total": sum(query_times),
        "mask_seconds_mean": statistics.mean(query_times),
        "mask_seconds_max": max(query_times),
        "mask_predictions_total": sum(query_predictions),
        "mask_predictions_mean": statistics.mean(query_predictions),
        "old_comparison_seconds_total": sum(old_times),
        "matched_comparison_counts": dict(disagreements),
        "distance_to_constraint_m_mean": statistics.mean(
            row["distance_to_active_constraint_m"] for row in rows
        ),
        "certificate_verification_failures": 0,
        "final_verification_seconds_total": sum(
            row["final_verification_seconds"] for row in rows
        ),
        "final_verification_predictions_total": sum(
            row["final_verification_predictions"] for row in rows
        ),
    }


def run_episode(environment, method, *, seed=None, compare=True):
    vehicle = MeteredVehicle()
    shield = BankersShield(vehicle)
    memory = CausalMemory()
    inherited = _ExternalExecutionAdapter(vehicle=vehicle)
    # Caller has already durably recorded STARTED for any Development pair.
    ordinary, _ = environment.reset(seed=seed)
    inherited.reset()
    rows, times, predictions, old_times = [], [], [], []
    cpu_start = time.process_time()
    for step in range(1800):
        memory.ingest(ordinary)
        fallback = inherited.action(ordinary)
        require(inherited.state == memory.state, "matched causal state")
        transitions = tuple(
            Transition(position, limit)
            for position, limit in sorted(inherited._known_transitions.items())
        )
        started = time.perf_counter()
        menu = SafeActionSet(
            tuple(shield.certify(ordinary, memory, a) for a in ACTION_GRID)
        )
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        predictions.append(menu.physics_prediction_count)
        old_started = time.perf_counter()
        old = (
            [
                inherited_test(
                    memory.state,
                    a,
                    float(ordinary[2] * 37),
                    transitions,
                    fallback,
                    vehicle,
                )
                for a in ACTION_GRID
            ]
            if compare
            else []
        )
        old_times.append(time.perf_counter() - old_started)
        selected = certificate = None
        final_seconds = 0.0
        final_predictions = 0
        if method == "CAUSAL60":
            action = fallback
        elif method == "OLD_POST":
            action = 1.0 if old[20]["accepted"] else fallback
        elif method == "BANKERS_PRE":
            require(menu.mask[0] and any(menu.mask), "certified backup/action nonempty")
            selected = max(i for i, valid in enumerate(menu.mask) if valid)
            action = ACTION_GRID[selected]
            final_start = time.perf_counter()
            final = shield.certify(ordinary, memory, action)
            final_seconds = time.perf_counter() - final_start
            final_predictions = final.physics_prediction_count
            require(
                final.accepted and final == menu.certificates[selected],
                "final certificate discrepancy",
            )
            certificate = final.as_dict()
        else:
            raise ValueError(method)
        next_ordinary, _, terminated, truncated, info = environment.step([action])
        row = {
            "step": step,
            "method": method,
            "causal_pre_state": asdict(memory.state),
            "observation": ordinary.tolist(),
            "memory": memory.snapshot(),
            "safe_action_mask": list(menu.mask),
            "mask_sha256": menu.sha256,
            "certificate_summaries": [
                {
                    "accepted": c.accepted,
                    "sha256": c.sha256,
                    "physics_prediction_count": c.physics_prediction_count,
                    "rejection_reason": c.rejection_reason,
                }
                for c in menu.certificates
            ],
            "proposal_action": 1.0,
            "selected_index": selected,
            "selected_certificate": certificate,
            "executed_action": action,
            "actual_next_state": asdict(environment.unwrapped.state),
            "actual_speed_limit_m_s": info["speed_limit_m_s"],
            "speed_excess_m_s": info["speed_excess_m_s"],
            "signed_energy_increment_kwh": info["step_energy_kwh"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "backup_available": menu.mask[0],
            "known_transition_braking": (
                certificate["known_transition_braking_feasible"]
                if certificate
                else "DIAGNOSTIC_MASK_ONLY"
            ),
            "unknown_preview_recovery": (
                certificate["unknown_preview_recoverable"]
                if certificate
                else "DIAGNOSTIC_MASK_ONLY"
            ),
            "inherited_comparison": old,
            "completion_bound_s": old[20]["completion_bound_s"] if old else None,
            "deadline_diagnostic_applicable": bool(old),
            "distinct_successors": menu.distinct_successors,
            "velocity_span_m_s": menu.velocity_span_m_s,
            "positive_acceleration_available": any(
                c.accepted
                and c.predicted_next_state["velocity_m_s"] > memory.state.velocity_m_s
                for c in menu.certificates
            ),
            "distance_to_active_constraint_m": (
                menu.certificates[selected or 0].distance_to_active_constraint_m or 0.0
            ),
            "mask_seconds": elapsed,
            "mask_physics_predictions": menu.physics_prediction_count,
            "final_verification_seconds": final_seconds,
            "final_verification_predictions": final_predictions,
        }
        rows.append(row)
        inherited.observe_external_action(action)
        memory.execute(action, vehicle)
        require(
            asdict(memory.state) == row["actual_next_state"], "actual physics parity"
        )
        if method == "BANKERS_PRE" and info["speed_excess_m_s"] != 0:
            metadata = write_evidence("COUNTEREXAMPLE__" + str(time.time_ns()), rows)
            atomic_json(HERE / "invariant_counterexample.json", metadata)
            raise SafetyIntegrityError("ADMITTED_ACTION_SPEED_COUNTEREXAMPLE")
        ordinary = next_ordinary
        if terminated or truncated:
            break
        if time.process_time() - cpu_start > 21600:
            raise SafetyIntegrityError("episode CPU budget exceeded")
    metrics = EpisodeMetrics(**info["episode_metrics"])
    summary = summarize(
        rows,
        metrics,
        is_feasible(metrics, TaskSpecification(140, 0)),
        times,
        predictions,
        old_times,
    )
    summary["physics_predictions_including_checks_and_memory"] = vehicle.predictions
    summary["episode_cpu_s"] = time.process_time() - cpu_start
    return rows, summary


def write_evidence(key, rows):
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / (key + ".jsonl.gz")
    if path.exists():
        raise SafetyIntegrityError("immutable evidence already exists")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (
                        json.dumps(
                            row, allow_nan=False, sort_keys=True, separators=(",", ":")
                        )
                        + "\n"
                    ).encode()
                )
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.relative_to(HERE)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "native_records": len(rows),
    }


def read_evidence(metadata):
    path = HERE / metadata["path"]
    require(file_sha256(path) == metadata["sha256"], "evidence content digest")
    with gzip.open(path, "rt") as stream:
        result = [json.loads(line) for line in stream]
    require(len(result) == metadata["native_records"], "native record accounting")
    return result


def preflight():
    verify_sources()
    started, cpu = time.perf_counter(), time.process_time()
    search_results = search()
    fixture_ledger_path = HERE / "fixture_ledger.json"
    require(
        not fixture_ledger_path.exists(),
        "fixture preflight already started; rerun forbidden",
    )
    fixture_ledger = {name: {"state": "NOT_STARTED"} for name in fixtures()}
    atomic_json(fixture_ledger_path, fixture_ledger)
    output = {}
    for name, track in fixtures().items():
        print("FIXTURE START", name, flush=True)
        fixture_ledger[name] = {
            "state": "STARTED",
            "identity": digest([name, identities()]),
        }
        atomic_json(fixture_ledger_path, fixture_ledger)
        env = gym.wrappers.TimeLimit(
            LongiControlEnv(track_generator=lambda _rng: track), 1800
        )
        rows, result = run_episode(env, "BANKERS_PRE", compare=True)
        result["evidence"] = write_evidence("FIXTURE__" + name, rows)
        replay_start = time.perf_counter()
        replay_records(rows, track, "BANKERS_PRE", result, full_masks=True)
        result["independent_replay_seconds"] = time.perf_counter() - replay_start
        for row in rows:
            verify_certificate(
                row["selected_certificate"],
                row["observation"],
                CausalMemory.restore(row["memory"]),
            )
        # Readback is byte/hash/count checked before any preflight gate passes.
        require(read_evidence(result["evidence"]) == rows, "atomic fixture readback")
        result["max_memory_envelope_knots"] = max(
            len(r["memory"]["envelope"]) for r in rows
        )
        output[name] = result
        fixture_ledger[name].update(state="COMPLETED", evidence=result["evidence"])
        atomic_json(fixture_ledger_path, fixture_ledger)
        atomic_json(HERE / "fixture_results.json", output)
        print("FIXTURE DONE", name, result["steps"], result["metrics"], flush=True)
        env.close()
    worst_mask = max(v["mask_seconds_mean"] for v in output.values())
    old_cost = max(
        v["old_comparison_seconds_total"] / v["steps"] for v in output.values()
    )
    replay_cost = max(
        v["independent_replay_seconds"] / v["steps"] for v in output.values()
    )
    # Complete 54x1800 panel plus one full independent replay, final verification,
    # evidence work and generous overhead. No smaller panel substitution.
    projected = 54 * 1800 * (worst_mask + old_cost + replay_cost) * 1.35
    max_bytes_step = max(v["evidence"]["bytes"] / v["steps"] for v in output.values())
    projected_bytes = max_bytes_step * 54 * 1800 * 1.25 + sum(
        v["evidence"]["bytes"] for v in output.values()
    )
    gates = {
        "fixture_speed_safe": all(v["exact_speed_compliant"] for v in output.values()),
        "backup_nonempty": all(
            v["backup_available_steps"] == v["steps"] for v in output.values()
        ),
        "independent_native_replay": True,
        "source_immutable": True,
        "projected_total_compute_within_6h": projected + time.perf_counter() - started
        < 21600,
        "projected_artifacts_within_3GiB": projected_bytes < 3 * 1024**3,
    }
    report = {
        "classification": "FORMALLY_SUPPORTED_UNDER_STATED_MODEL",
        "counterexample_search": search_results,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "projected_full_development_seconds": projected,
        "projected_retained_bytes": projected_bytes,
        "preflight_wall_s": time.perf_counter() - started,
        "preflight_cpu_s": time.process_time() - cpu,
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "source_identity": identities(),
        "fixtures": output,
        "training_transitions": 0,
        "gradient_updates": 0,
    }
    atomic_json(HERE / "preparation_status.json", report)
    print("PREFLIGHT", gates, projected, projected_bytes, flush=True)
    return report


def develop():
    verify_sources()
    prep = json.loads((HERE / "preparation_status.json").read_text())
    require(prep["all_gates_passed"], "Development preflight gate")
    require(prep["source_identity"] == identities(), "post-preflight source change")
    with exclusive_writer(HERE):
        split = (
            json.loads((HERE / "split_ledger.json").read_text())
            if (HERE / "split_ledger.json").exists()
            else build_split()
        )
        manifest = {
            "identity": identities(),
            "split_sha256": file_sha256(HERE / "split_ledger.json"),
            "ordered_tasks_sha256": split["ordered_sha256"],
            "methods": ["CAUSAL60", "OLD_POST", "BANKERS_PRE"],
            "maximum_episodes": 54,
        }
        manifest_path = HERE / "execution_manifest.json"
        if manifest_path.exists():
            require(
                json.loads(manifest_path.read_text()) == manifest, "immutable manifest"
            )
        else:
            atomic_json(manifest_path, manifest)
        ledger = PairLedger(HERE / "pair_ledger.json", split["ordered_tasks"], manifest)
        started = time.perf_counter()
        prior_cpu = prep["preflight_cpu_s"] + sum(
            v.get("result", {}).get("pair_cpu_s", 0)
            for v in ledger.data["pairs"].values()
        )
        start_cpu = time.process_time()
        for task in split["ordered_tasks"]:
            verify_task(task)
            for method in manifest["methods"]:
                key = f"{method}__{task['numeric_seed']}"
                if ledger.data["pairs"][key]["state"] == "COMPLETED":
                    read_evidence(ledger.data["pairs"][key]["result"]["evidence"])
                    continue
                require(
                    prior_cpu + time.process_time() - start_cpu < 21600,
                    "6h active budget",
                )
                require(
                    sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
                    < 3 * 1024**3,
                    "3GiB budget",
                )
                print("PAIR START", key, flush=True)
                ledger.start(key)
                pair_cpu = time.process_time()
                env = gym.wrappers.TimeLimit(StochasticTrack(), 1800)
                rows, result = run_episode(
                    env, method, seed=task["numeric_seed"], compare=method != "CAUSAL60"
                )
                result["evidence"] = write_evidence(key, rows)
                replay_start = time.perf_counter()
                replay_records(
                    rows, env.unwrapped.track, method, result, full_masks=True
                )
                result["independent_replay_seconds"] = (
                    time.perf_counter() - replay_start
                )
                require(
                    read_evidence(result["evidence"]) == rows,
                    "immutable native readback",
                )
                result["task"] = {
                    k: task[k]
                    for k in ("numeric_seed", "route_sha256", "full_task_sha256")
                }
                result["method"] = method
                result["pair_cpu_s"] = time.process_time() - pair_cpu
                ledger.complete(key, result)
                env.close()
                print(
                    "PAIR DONE",
                    key,
                    result["steps"],
                    result["feasible"],
                    result["metrics"],
                    flush=True,
                )
        verify_sources()
        final = aggregate(ledger.data)
        final.update(
            development_wall_s=time.perf_counter() - started,
            active_compute_s=prior_cpu + time.process_time() - start_cpu,
            peak_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            retained_artifact_bytes=sum(
                p.stat().st_size for p in HERE.rglob("*") if p.is_file()
            ),
            source_config_identity=identities(),
            split_ordered_sha256=split["ordered_sha256"],
            pair_ledger_sha256=file_sha256(HERE / "pair_ledger.json"),
        )
        atomic_json(HERE / "results.json", final)
        print("DEVELOPMENT COMPLETE", final["classification"], flush=True)


def aggregate(data):
    all_results = [
        v["result"] for v in data["pairs"].values() if v["state"] == "COMPLETED"
    ]
    methods = {}
    for method in ("CAUSAL60", "OLD_POST", "BANKERS_PRE"):
        results = [r for r in all_results if r["method"] == method]
        steps = sum(r["steps"] for r in results)
        feasible = [r for r in results if r["feasible"]]
        counts, distinct, comparison = Counter(), Counter(), Counter()
        for result in results:
            counts.update(
                {int(k): v for k, v in result["safe_action_count_distribution"].items()}
            )
            distinct.update(
                {
                    int(k): v
                    for k, v in result["distinct_successor_distribution"].items()
                }
            )
            comparison.update(result["matched_comparison_counts"])
        methods[method] = {
            "episodes": len(results),
            "steps": steps,
            "feasible_episodes": len(feasible),
            "raw_RSR": len(feasible) / len(results),
            "completed": sum(r["completion"] for r in results),
            "deadline_failures": sum(not r["deadline_compliant"] for r in results),
            "speed_violation_episodes": sum(
                not r["exact_speed_compliant"] for r in results
            ),
            "max_speed_violation_m_s": max(
                r["metrics"]["max_speed_violation_m_s"] for r in results
            ),
            "feasible_only_signed_energy_kwh_mean": statistics.mean(
                r["metrics"]["energy_kwh"] for r in feasible
            )
            if feasible
            else None,
            "travel_time_s_mean": statistics.mean(
                r["metrics"]["travel_time_s"] for r in results
            ),
            "safe_action_count_distribution": dict(sorted(counts.items())),
            "distinct_successor_distribution": dict(sorted(distinct.items())),
            "meaningful_choices_fraction": sum(
                r["meaningful_choices_steps"] for r in results
            )
            / steps,
            "positive_acceleration_available_fraction": sum(
                r["positive_acceleration_available_steps"] for r in results
            )
            / steps,
            "admitted_action_fraction": sum(r["admitted_actions"] for r in results)
            / (21 * steps),
            "rejected_positive_action_fraction": sum(
                r["rejected_positive_grid_actions"] for r in results
            )
            / (10 * steps),
            "backup_available_fraction": sum(
                r["backup_available_steps"] for r in results
            )
            / steps,
            "mask_seconds_mean": sum(r["mask_seconds_total"] for r in results) / steps,
            "mask_predictions_mean": sum(r["mask_predictions_total"] for r in results)
            / steps,
            "matched_comparison_counts": dict(comparison),
            "failures": [
                {
                    "seed": r["task"]["numeric_seed"],
                    "metrics": r["metrics"],
                    "deadline_compliant": r["deadline_compliant"],
                    "speed_compliant": r["exact_speed_compliant"],
                }
                for r in results
                if not r["feasible"]
            ],
        }
    bank = methods["BANKERS_PRE"]
    integrity = len(all_results) == 54 and all(
        v["state"] == "COMPLETED" for v in data["pairs"].values()
    )
    useful = (
        bank["meaningful_choices_fraction"] >= 0.25
        and bank["positive_acceleration_available_fraction"] >= 0.05
    )
    classification = (
        "BANKERS_PRE_SHIELD_INTEGRITY_STOP"
        if not integrity
        else "BANKERS_PRE_SHIELD_COUNTEREXAMPLE_FOUND"
        if bank["speed_violation_episodes"]
        else "BANKERS_PRE_SHIELD_CERTIFIED_READY_FOR_RL_DESIGN"
        if useful
        else "BANKERS_PRE_SHIELD_VALID_BUT_TOO_CONSERVATIVE"
    )
    return {
        "classification": classification,
        "methods": methods,
        "performance_episodes": len(all_results),
        "all_pairs_independently_replayed": True,
        "certificate_verification_failures": 0,
        "observed_invariant_counterexamples": 0,
        "training_transitions": 0,
        "gradient_updates": 0,
        "validation_performance_episodes": 0,
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_ids": list(range(4000, 4018)),
            "constructed_tasks": 0,
            "performance_episodes": 0,
        },
        "authority_thresholds_predeclared": {
            "meaningful_choices_fraction": 0.25,
            "positive_acceleration_available_fraction": 0.05,
            "meaningful_velocity_span_m_s": 0.05,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "develop"))
    args = parser.parse_args()
    try:
        if args.mode == "preflight":
            with exclusive_writer(HERE):
                preflight()
        else:
            develop()
    except Exception as error:
        atomic_json(
            HERE / "integrity_stop.json",
            {
                "mode": args.mode,
                "error_type": type(error).__name__,
                "error": str(error),
                "classification": "BANKERS_PRE_SHIELD_INTEGRITY_STOP",
                "rerun_started_pairs_forbidden": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
