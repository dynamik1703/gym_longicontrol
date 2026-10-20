"""Bounded actual-pipeline resource benchmark on unreserved physical fixtures."""

from __future__ import annotations

import resource
import tempfile
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from benchmarks.hybrid_rl_design.controller import HybridTargetSpeedController
from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from .agent import ConstrainedSAC
from .checkpointing import file_sha256
from .runner import HybridRunner

BENCHMARK_NATIVE_TRANSITIONS = 3_000
BENCHMARK_UPDATES = 10
BENCHMARK_WARMUP = 2_560
BENCHMARK_BATCH_SIZE = 256
CANONICAL_REPLAY_CAPACITY = 300_000
POLICY_COUNT = 3
NATIVE_TRANSITIONS_PER_POLICY = 300_000
UPDATES_PER_POLICY = 30_000
DIAGNOSTIC_EPISODES = 18 * 6 * POLICY_COUNT
FINAL_EVALUATION_EPISODES = 36 * (POLICY_COUNT + 2)
CHECKPOINTS_RETAINED = 6 * POLICY_COUNT + 1
CHECKPOINT_WRITES_TOTAL = 30 * POLICY_COUNT


class _FixedTrackGenerator:
    def __init__(self, track) -> None:
        self.track = track

    def __call__(self, _rng):
        return self.track


def _fixture_environment() -> gym.Env:
    track = physical_fixtures()["consecutive_downward_changes"]
    return gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=_FixedTrackGenerator(track)),
        max_episode_steps=1_800,
    )


def _actor_only_fixture(agent: ConstrainedSAC) -> dict[str, float | int]:
    environment = _fixture_environment()
    controller = HybridTargetSpeedController(strategy_enabled=True)
    native_steps = 0
    macro_decisions = 0
    actor_time_s = 0.0
    started = time.perf_counter()
    try:
        observation, _ = environment.reset(seed=20261020)
        controller.reset()
        held = 1.0
        ended = False
        while not ended:
            if native_steps % 10 == 0:
                strategy_observation = controller.prepare_strategy_step(observation)
                inference_started = time.perf_counter()
                held = agent.act(strategy_observation, deterministic=True)
                actor_time_s += time.perf_counter() - inference_started
                macro_decisions += 1
            action = controller.action(observation, held)
            observation, _, terminated, truncated, _ = environment.step([action])
            controller.observe_executed_action(action)
            native_steps += 1
            ended = bool(terminated or truncated)
    finally:
        environment.close()
    return {
        "wall_time_s": time.perf_counter() - started,
        "native_steps": native_steps,
        "macro_decisions": macro_decisions,
        "actor_inference_time_s": actor_time_s,
    }


def benchmark_actual_pipeline() -> dict[str, Any]:
    torch.set_num_threads(1)
    configuration_sha256 = "resource-benchmark-configuration"
    source_hashes = {"resource_fixture": "unreserved-physical-fixture"}
    runner = HybridRunner(
        training_seed=11,
        attempt_id="RESOURCE-BENCHMARK-NOT-SCIENTIFIC-TRAINING",
        configuration_sha256=configuration_sha256,
        source_hashes=source_hashes,
        environment=_fixture_environment(),
        native_budget=BENCHMARK_NATIVE_TRANSITIONS,
        update_budget=BENCHMARK_UPDATES,
        warmup_native_transitions=BENCHMARK_WARMUP,
        batch_size=BENCHMARK_BATCH_SIZE,
        replay_capacity=CANONICAL_REPLAY_CAPACITY,
        verify_track_content=False,
    )
    update_time_s = 0.0
    original_update = runner.agent.update

    def timed_update(*args, **kwargs):
        nonlocal update_time_s
        started = time.perf_counter()
        result = original_update(*args, **kwargs)
        update_time_s += time.perf_counter() - started
        return result

    runner.agent.update = timed_update  # type: ignore[method-assign]
    started = time.perf_counter()
    runner.run_to_budget()
    pipeline_wall_time_s = time.perf_counter() - started
    collection_time_s = pipeline_wall_time_s - update_time_s
    evaluation = _actor_only_fixture(runner.agent)
    with tempfile.TemporaryDirectory(prefix="hybrid-rl-resource-") as directory:
        checkpoint = Path(directory) / "checkpoint.pkl"
        checkpoint_started = time.perf_counter()
        metadata = runner.save(checkpoint)
        checkpoint_write_s = time.perf_counter() - checkpoint_started
        restore_started = time.perf_counter()
        restored = HybridRunner.restore(
            checkpoint,
            expected_sha256=file_sha256(checkpoint),
            configuration_sha256=configuration_sha256,
            source_hashes=source_hashes,
            attempt_id=runner.attempt_id,
        )
        checkpoint_restore_s = time.perf_counter() - restore_started
        restored.close()
    runner.close()
    mean_collection_s = collection_time_s / BENCHMARK_NATIVE_TRANSITIONS
    mean_update_s = update_time_s / BENCHMARK_UPDATES
    mean_evaluation_episode_s = float(evaluation["wall_time_s"])
    collection_projection_s = (
        mean_collection_s * NATIVE_TRANSITIONS_PER_POLICY * POLICY_COUNT
    )
    update_projection_s = mean_update_s * UPDATES_PER_POLICY * POLICY_COUNT
    evaluation_projection_s = mean_evaluation_episode_s * (
        DIAGNOSTIC_EPISODES + FINAL_EVALUATION_EPISODES
    )
    checkpoint_projection_s = checkpoint_write_s * CHECKPOINT_WRITES_TOTAL
    raw_projection_s = (
        collection_projection_s
        + update_projection_s
        + evaluation_projection_s
        + checkpoint_projection_s
    )
    uncertainty_factor = 2.0
    projected_runtime_s = raw_projection_s * uncertainty_factor
    retained_checkpoint_bytes = metadata["size_bytes"] * CHECKPOINTS_RETAINED
    retained_diagnostics_bytes = 1_000_000_000
    projected_disk_bytes = retained_checkpoint_bytes + retained_diagnostics_bytes
    return {
        "schema_version": 1,
        "bounded_preparation_only": True,
        "fixture": "consecutive_downward_changes",
        "representative_replay_capacity": CANONICAL_REPLAY_CAPACITY,
        "actual_pipeline": {
            "native_transitions": BENCHMARK_NATIVE_TRANSITIONS,
            "macro_transitions": runner.macro_transition_count,
            "gradient_updates": runner.gradient_update_count,
            "replay_occupancy": len(runner.replay),
            "batch_size": BENCHMARK_BATCH_SIZE,
            "pipeline_wall_time_s": pipeline_wall_time_s,
            "collection_macro_shield_time_s": collection_time_s,
            "update_time_s": update_time_s,
            "mean_collection_time_per_native_transition_s": mean_collection_s,
            "mean_actual_sac_update_time_s": mean_update_s,
            "checkpoint_size_bytes": metadata["size_bytes"],
            "checkpoint_write_s": checkpoint_write_s,
            "checkpoint_restore_s": checkpoint_restore_s,
            "actor_only_fixture_evaluation": evaluation,
            "maximum_rss_platform_units": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
        "future_complete_three_policy_study": {
            "real_native_transitions": NATIVE_TRANSITIONS_PER_POLICY * POLICY_COUNT,
            "gradient_updates": UPDATES_PER_POLICY * POLICY_COUNT,
            "diagnostic_episodes": DIAGNOSTIC_EPISODES,
            "final_evaluation_episodes": FINAL_EVALUATION_EPISODES,
            "collection_projection_s": collection_projection_s,
            "update_projection_s": update_projection_s,
            "evaluation_projection_s": evaluation_projection_s,
            "checkpoint_projection_s": checkpoint_projection_s,
            "raw_projection_s": raw_projection_s,
            "uncertainty_factor": uncertainty_factor,
            "projected_total_runtime_s": projected_runtime_s,
            "projected_total_runtime_hours": projected_runtime_s / 3600.0,
            "retained_checkpoint_count": CHECKPOINTS_RETAINED,
            "projected_retained_checkpoint_bytes": retained_checkpoint_bytes,
            "projected_retained_diagnostics_bytes": retained_diagnostics_bytes,
            "projected_retained_disk_bytes": projected_disk_bytes,
            "projected_retained_disk_gb": projected_disk_bytes / 1e9,
        },
        "limits": {"runtime_hours": 24.0, "retained_disk_gb": 12.0},
        "runtime_gate_passed": projected_runtime_s <= 24.0 * 3600.0,
        "disk_gate_passed": projected_disk_bytes <= 12_000_000_000,
        "no_reserved_diagnostic_or_validation_task_used": True,
        "no_main_training_occurred": True,
        "preparation_interactions": {
            "real_native_transitions": BENCHMARK_NATIVE_TRANSITIONS,
            "gradient_updates": BENCHMARK_UPDATES,
            "unreserved_fixture_evaluation_episodes": 1,
        },
    }
