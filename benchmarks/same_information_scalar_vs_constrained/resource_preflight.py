"""Preparation-only Scalar throughput and conservative resource projection."""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks.constrained_v2_collector_replication.checkpointing import file_sha256
from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import track_from_seed

from .agent import build_scalar_agent
from .config import load_configuration
from .split import canonical_streams, evaluation_sets, opened_split_rows
from .streams import build_fixture_environment

ARCHIVED_EVALUATION_SECONDS_PER_EPISODE = 0.3485704306513071
ARCHIVED_EVALUATION_SOURCE = (
    "completed collector-replication resource pilot; identical deterministic actor path"
)
ARCHIVED_FULL_CHECKPOINT_MAX_BYTES = 151_275_789
ARCHIVED_FULL_CHECKPOINT_MAX_WRITE_S = 15.777613833954092


def _fixture_seeds(count: int = 32) -> tuple[int, ...]:
    protected = (
        [row for rows in opened_split_rows().values() for row in rows]
        + [
            row
            for split in evaluation_sets().values()
            for row in split["accepted_tasks"]
        ]
        + [
            {
                "route_sha256": row["route_sha256"],
                "full_task_sha256": row["full_task_sha256"],
            }
            for stream in canonical_streams().values()
            for row in stream["accepted_tasks"]
        ]
    )
    routes = {row["route_sha256"] for row in protected}
    full = {row["full_task_sha256"] for row in protected}
    result = []
    seed = 0
    while len(result) < count:
        fingerprint = fingerprint_track(
            track_from_seed(seed), v2_semantics(), numeric_seed=seed
        )
        if (
            fingerprint.route_sha256 not in routes
            and fingerprint.full_task_sha256 not in full
        ):
            result.append(seed)
        seed += 1
    return tuple(result)


def run_resource_preflight() -> dict[str, Any]:
    import torch
    from tianshou.data import Collector, ReplayBuffer

    configuration = load_configuration()
    transitions = int(configuration.raw["resource_gate"]["pilot_transitions"])
    if (
        transitions
        > configuration.raw["resource_gate"]["maximum_preparation_transitions"]
    ):
        raise PermissionError("Preparation transition allowance exceeded")
    seeds = _fixture_seeds()
    environment = build_fixture_environment(
        configuration, reset_seeds=seeds, max_simulator_steps=transitions
    )
    agent, _ = build_scalar_agent(environment, training_seed=0, device="cpu", threads=4)
    policy = agent.policy
    policy.train()
    collector = Collector(
        policy,
        environment,
        ReplayBuffer(configuration.raw["algorithm"]["replay_capacity"]),
        exploration_noise=True,
    )
    counters = {"real_transitions": 0, "rl_gradient_updates": 0}
    collection_s = 0.0
    update_s = 0.0
    started = time.perf_counter()
    while counters["real_transitions"] < transitions:
        environment.set_actor_version(counters["rl_gradient_updates"])
        before = time.perf_counter()
        stats = collector.collect(n_step=1)
        collection_s += time.perf_counter() - before
        if int(stats["n/st"]) != 1:
            raise RuntimeError("Pilot transition accounting changed")
        counters["real_transitions"] += 1
        target = math.floor(counters["real_transitions"] * 0.1 + 1e-12)
        before = time.perf_counter()
        while counters["rl_gradient_updates"] < target and len(collector.buffer) >= 256:
            policy.update(256, collector.buffer)
            counters["rl_gradient_updates"] += 1
        update_s += time.perf_counter() - before
    walltime_s = time.perf_counter() - started
    with tempfile.TemporaryDirectory(prefix="sameinfo-scalar-resource-") as temporary:
        checkpoint = Path(temporary) / "checkpoint.pt"
        before = time.perf_counter()
        torch.save(
            {
                "policy": policy.state_dict(),
                "replay_buffer": collector.buffer,
                "environment_state": environment,
                "counters": counters,
            },
            checkpoint,
        )
        checkpoint_write_s = time.perf_counter() - before
        checkpoint_size = checkpoint.stat().st_size
        checkpoint_sha = file_sha256(checkpoint)
        before = time.perf_counter()
        restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_load_s = time.perf_counter() - before
        if restored["counters"] != counters:
            raise RuntimeError("Pilot checkpoint round trip changed counters")
    used = environment.used_exposures
    environment.close()
    budget = configuration.raw["real_transition_budget"]
    one_policy_s = walltime_s * budget / transitions
    checkpoint_s = 3 * (budget // 10_000) * ARCHIVED_FULL_CHECKPOINT_MAX_WRITE_S
    evaluation_episodes = 6 * 6 * 18 + 6 * 18
    evaluation_s = evaluation_episodes * ARCHIVED_EVALUATION_SECONDS_PER_EPISODE
    total_h = (3 * one_policy_s + checkpoint_s + evaluation_s) / 3600
    retained_count = 3 * 7
    retained_gb = retained_count * ARCHIVED_FULL_CHECKPOINT_MAX_BYTES / 1e9
    runtime_limit = configuration.raw["resource_gate"][
        "maximum_new_main_study_walltime_h"
    ]
    disk_limit = configuration.raw["resource_gate"]["maximum_retained_new_artifacts_gb"]
    return {
        "schema_version": 1,
        "preparation_simulator_transitions": transitions,
        "new_policy_evaluations": 0,
        "pilot": {
            "fixture_seeds_used": [row["reset_seed"] for row in used],
            "transitions": transitions,
            "updates": counters["rl_gradient_updates"],
            "walltime_s": walltime_s,
            "collection_walltime_s": collection_s,
            "update_walltime_s": update_s,
            "transitions_per_second_including_updates": transitions / walltime_s,
            "updates_per_second": counters["rl_gradient_updates"] / update_s,
            "checkpoint": {
                "size_bytes": checkpoint_size,
                "sha256": checkpoint_sha,
                "write_s": checkpoint_write_s,
                "load_s": checkpoint_load_s,
            },
            "outcomes_used_for_science": False,
        },
        "diagnostic_evaluation_cost": {
            "seconds_per_episode": ARCHIVED_EVALUATION_SECONDS_PER_EPISODE,
            "source": ARCHIVED_EVALUATION_SOURCE,
            "new_evaluation_run": False,
        },
        "projection": {
            "one_scalar_policy_s": one_policy_s,
            "three_scalar_policies_s": 3 * one_policy_s,
            "diagnostic_episodes": 648,
            "validation_episodes": 108,
            "all_evaluation_s": evaluation_s,
            "checkpoint_overhead_s": checkpoint_s,
            "new_main_study_walltime_h": total_h,
            "retained_checkpoint_count": retained_count,
            "conservative_checkpoint_size_bytes": ARCHIVED_FULL_CHECKPOINT_MAX_BYTES,
            "retained_new_artifacts_gb": retained_gb,
        },
        "gates": {
            "runtime_limit_h": runtime_limit,
            "runtime_pass": total_h <= runtime_limit,
            "disk_limit_gb": disk_limit,
            "disk_pass": retained_gb <= disk_limit,
        },
        "passed": total_h <= runtime_limit and retained_gb <= disk_limit,
    }
