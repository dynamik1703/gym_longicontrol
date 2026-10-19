"""Measured serial-runtime, checkpoint, evaluation, and disk hard gate."""

from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import track_from_seed

from .checkpointing import file_sha256
from .config import load_configuration
from .evaluation import evaluate_actor_on_seeds
from .execution import apply_collector_schedule
from .split import build_split_ledger
from .streams import build_fixture_environment

ARCHIVED_FULL_CHECKPOINT_MAX_BYTES = 151_275_789
ARCHIVED_FULL_CHECKPOINT_MAX_WRITE_S = 15.777613833954092
ARCHIVED_CHECKPOINT_SOURCE = (
    "completed multistep-physics-mbrl-v1 300k verified checkpoints"
)


def _fixture_seed_pool(count: int = 32) -> tuple[int, ...]:
    ledger = build_split_ledger()
    protected = [
        row
        for split in ledger["evaluation_sets"].values()
        for row in split["accepted_tasks"]
    ] + [
        row
        for stream in ledger["canonical_streams"].values()
        for row in stream["accepted_tasks"]
    ]
    routes = {row["route_sha256"] for row in protected}
    full = {row["full_task_sha256"] for row in protected}
    result: list[int] = []
    candidate = 0
    while len(result) < count:
        fingerprint = fingerprint_track(
            track_from_seed(candidate), v2_semantics(), numeric_seed=candidate
        )
        if (
            fingerprint.route_sha256 not in routes
            and fingerprint.full_task_sha256 not in full
        ):
            result.append(candidate)
        candidate += 1
    return tuple(result)


def _pilot_collector(collector_name: str, transitions: int) -> dict[str, Any]:
    import torch
    from tianshou.data import Collector, ReplayBuffer

    configuration = load_configuration()
    fixture_seeds = _fixture_seed_pool()
    environment = build_fixture_environment(
        configuration,
        reset_seeds=fixture_seeds,
        max_simulator_steps=transitions,
    )
    agent, _logger = build_agent(
        configuration.v2,
        environment,
        training_seed=0,
        device="cpu",
        threads=4,
    )
    policy = agent.policy
    policy.train()
    collector = Collector(
        policy,
        environment,
        ReplayBuffer(configuration.v2.algorithm.buffer_size),
        exploration_noise=True,
    )
    counters = {"real_transitions": 0, "rl_gradient_updates": 0}
    previous_completed = 0
    started = time.perf_counter()
    while counters["real_transitions"] < transitions:
        environment.set_actor_version(counters["rl_gradient_updates"])
        stats = collector.collect(n_step=1)
        if int(stats["n/st"]) != 1:
            raise RuntimeError("Pilot transition accounting changed")
        counters["real_transitions"] += 1
        completed = environment.completed_episodes != previous_completed
        costs = None
        if completed:
            costs = np.asarray(environment.last_completed_episode.costs)
        target = math.floor(counters["real_transitions"] * 0.1 + 1e-12)
        apply_collector_schedule(
            collector_name=collector_name,
            completed_costs=costs,
            policy=policy,
            replay_buffer=collector.buffer,
            counters=counters,
            target_updates=target,
            batch_size=configuration.v2.algorithm.batch_size,
        )
        if completed:
            previous_completed = environment.completed_episodes
    elapsed = time.perf_counter() - started
    ledger = build_split_ledger()
    protected = [
        row
        for split in ledger["evaluation_sets"].values()
        for row in split["accepted_tasks"]
    ] + [
        row
        for stream in ledger["canonical_streams"].values()
        for row in stream["accepted_tasks"]
    ]
    protected_routes = {row["route_sha256"] for row in protected}
    protected_full = {row["full_task_sha256"] for row in protected}
    used = environment.used_exposures
    if any(
        row["route_sha256"] in protected_routes
        or row["full_task_sha256"] in protected_full
        for row in used
    ):
        raise RuntimeError("Resource pilot fixture matched frozen scientific content")
    with tempfile.TemporaryDirectory(prefix="longicontrol-v3-resource-") as temporary:
        checkpoint = Path(temporary) / "pilot.pt"
        checkpoint_started = time.perf_counter()
        torch.save(
            {
                "policy": policy.state_dict(),
                "replay_buffer": collector.buffer,
                "environment_state": environment,
                "counters": counters,
            },
            checkpoint,
        )
        checkpoint_write_s = time.perf_counter() - checkpoint_started
        checkpoint_size = checkpoint.stat().st_size
        checkpoint_sha = file_sha256(checkpoint)
        load_started = time.perf_counter()
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_load_s = time.perf_counter() - load_started
        if restored["counters"] != counters:
            raise RuntimeError("Resource checkpoint round-trip changed counters")
    environment.close()
    return {
        "collector": collector_name,
        "fixture_training_seed": 0,
        "transitions": transitions,
        "updates": counters["rl_gradient_updates"],
        "completed_episodes": previous_completed,
        "used_fixture_tasks": used,
        "walltime_s": elapsed,
        "transitions_per_second_including_updates": transitions / elapsed,
        "checkpoint": {
            "rows": len(collector.buffer),
            "replay_capacity": collector.buffer.maxsize,
            "size_bytes": checkpoint_size,
            "sha256": checkpoint_sha,
            "write_s": checkpoint_write_s,
            "load_s": checkpoint_load_s,
        },
    }


def _evaluation_pilot(policy: Any) -> dict[str, Any]:
    configuration = load_configuration()
    seeds = tuple(configuration.raw["resource_gate"]["fixture_seeds"])
    ledger = build_split_ledger()
    protected = [
        row
        for split in ledger["evaluation_sets"].values()
        for row in split["accepted_tasks"]
    ] + [
        row
        for stream in ledger["canonical_streams"].values()
        for row in stream["accepted_tasks"]
    ]
    protected_routes = {row["route_sha256"] for row in protected}
    protected_full = {row["full_task_sha256"] for row in protected}
    for seed in seeds:
        fingerprint = fingerprint_track(
            track_from_seed(seed), v2_semantics(), numeric_seed=seed
        )
        if (
            fingerprint.route_sha256 in protected_routes
            or fingerprint.full_task_sha256 in protected_full
        ):
            raise RuntimeError(
                "Evaluation pilot fixture matches frozen scientific content"
            )
    started = time.perf_counter()
    result = evaluate_actor_on_seeds(
        policy,
        configuration,
        evaluation_seeds=seeds,
        real_transitions=0,
        evaluation_id="RESOURCE_FIXTURE_NON_INTERFERENCE",
    )
    elapsed = time.perf_counter() - started
    return {
        "fixture_seeds": list(seeds),
        "episodes": result["episode_count"],
        "simulator_steps": sum(row["step_count"] for row in result["episodes"]),
        "walltime_s": elapsed,
        "seconds_per_episode": elapsed / result["episode_count"],
        "training_rng_unchanged": result["training_rng_unchanged"],
        "actor_unchanged": (
            result["actor_sha256_before"] == result["actor_sha256_after"]
        ),
        "outcomes_used_for_science": False,
    }


def run_resource_preflight() -> dict[str, Any]:
    configuration = load_configuration()
    transitions = int(
        configuration.raw["resource_gate"]["pilot_transitions_per_collector"]
    )
    episodic = _pilot_collector("EPISODIC", transitions)
    transitional = _pilot_collector("TRANSITIONAL", transitions)

    # Evaluation timing uses a fresh untrained learner on three numeric fixture tasks.
    environment = build_fixture_environment(
        configuration,
        reset_seeds=_fixture_seed_pool(),
        max_simulator_steps=1,
    )
    try:
        agent, _logger = build_agent(
            configuration.v2,
            environment,
            training_seed=0,
            device="cpu",
            threads=4,
        )
        evaluation = _evaluation_pilot(agent.policy)
    finally:
        environment.close()

    budget = configuration.raw["real_transition_budget"]
    episodic_policy_s = episodic["walltime_s"] * budget / transitions
    transitional_policy_s = transitional["walltime_s"] * budget / transitions
    evaluation_s = evaluation["seconds_per_episode"] * 756
    sample = max(
        episodic["checkpoint"]["size_bytes"],
        transitional["checkpoint"]["size_bytes"],
    )
    # Tianshou allocates replay arrays at their 100k capacity on first use; the
    # fixture artifact is therefore already capacity-sized rather than 2.048% full.
    # Bound it further by larger verified 300k checkpoints from the completed MVE
    # study, which also retained a physics sidecar absent from V3.
    full_checkpoint_bytes = max(
        math.ceil(sample * 2.0), ARCHIVED_FULL_CHECKPOINT_MAX_BYTES
    )
    checkpoint_sample_s = max(
        episodic["checkpoint"]["write_s"],
        transitional["checkpoint"]["write_s"],
    )
    full_checkpoint_s = max(
        checkpoint_sample_s * 2.0, ARCHIVED_FULL_CHECKPOINT_MAX_WRITE_S
    )
    checkpoint_writes = 6 * (budget // 10_000)
    checkpoint_s = full_checkpoint_s * checkpoint_writes
    total_s = (
        3 * episodic_policy_s + 3 * transitional_policy_s + evaluation_s + checkpoint_s
    )
    retained_checkpoints = 6 * 7
    retained_bytes = full_checkpoint_bytes * retained_checkpoints
    total_h = total_s / 3600.0
    disk_gb = retained_bytes / 1_000_000_000
    runtime_limit = float(
        configuration.raw["resource_gate"]["maximum_total_serial_walltime_h"]
    )
    disk_limit = float(configuration.raw["resource_gate"]["maximum_retained_disk_gb"])
    return {
        "schema_version": 1,
        "fixture_policy_outcomes_used": False,
        "diagnostic_tasks_used": False,
        "collector_pilots": {"EPISODIC": episodic, "TRANSITIONAL": transitional},
        "diagnostic_evaluation_pilot": evaluation,
        "projection": {
            "EPISODIC_policy_s": episodic_policy_s,
            "TRANSITIONAL_policy_s": transitional_policy_s,
            "all_diagnostic_evaluation_s": evaluation_s,
            "all_checkpoint_writes_s": checkpoint_s,
            "total_serial_walltime_h": total_h,
            "full_checkpoint_size_bytes_conservative": full_checkpoint_bytes,
            "retained_checkpoint_count": retained_checkpoints,
            "retained_disk_gb": disk_gb,
            "checkpoint_projection_method": (
                "live capacity-preallocated checkpoint bounded by archived "
                "verified full-100k checkpoint maxima"
            ),
            "archived_checkpoint_source": ARCHIVED_CHECKPOINT_SOURCE,
            "archived_full_checkpoint_max_bytes": (ARCHIVED_FULL_CHECKPOINT_MAX_BYTES),
            "archived_full_checkpoint_max_write_s": (
                ARCHIVED_FULL_CHECKPOINT_MAX_WRITE_S
            ),
        },
        "gates": {
            "runtime_limit_h": runtime_limit,
            "runtime_pass": total_h <= runtime_limit,
            "disk_limit_gb": disk_limit,
            "disk_pass": disk_gb <= disk_limit,
        },
        "passed": total_h <= runtime_limit and disk_gb <= disk_limit,
    }


def write_resource_report(path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps(run_resource_preflight(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
