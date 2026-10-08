"""Bounded synthetic and fixed-action preparation resource measurement."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import gym_longicontrol  # noqa: F401
from gym_longicontrol.domain.task import TaskSpecification

from .checkpointing import checkpoint_size_bytes, save_checkpoint
from .config import PLANNED_TRANSITIONS_PER_POLICY, PLANNED_UPDATE_CYCLES, GCSLConfig
from .diagnostics import DiagnosticsAccumulator
from .execution import atomic_json
from .learner import GCSLBatch, GCSLLearner
from .replay import TrajectoryReplay


def synthetic_batch(config: GCSLConfig, rng: np.random.Generator) -> GCSLBatch:
    return GCSLBatch(
        states=rng.normal(size=(config.batch_size, config.state_dim)).astype(
            np.float32
        ),
        actions=rng.uniform(-0.95, 0.95, size=(config.batch_size, 1)).astype(
            np.float32
        ),
        goals=rng.uniform(size=(config.batch_size, config.goal_dim)).astype(np.float32),
        lags=rng.integers(1, 100, size=config.batch_size),
    )


def physical_probe(transitions: int) -> dict[str, object]:
    if not 0 <= transitions <= 1000:
        raise ValueError("preparation physical probe is capped at 1,000 transitions")
    if transitions == 0:
        return {
            "transitions": 0,
            "elapsed_seconds": 0.0,
            "transitions_per_second": None,
            "track_seeds": [],
            "fixed_action": 0.0,
            "used_for_learning": False,
        }
    environment = gym.make("StochasticTrack-v1")
    seeds = []
    completed = 0
    seed = 2000
    started = time.perf_counter()
    try:
        environment.reset(seed=seed)
        seeds.append(seed)
        while completed < transitions:
            _, _, terminated, truncated, _ = environment.step(
                np.asarray([0.0], dtype=np.float32)
            )
            completed += 1
            if (terminated or truncated) and completed < transitions:
                seed = 2000 + len(seeds) % 9
                environment.reset(seed=seed)
                seeds.append(seed)
    finally:
        environment.close()
    elapsed = time.perf_counter() - started
    return {
        "transitions": completed,
        "elapsed_seconds": elapsed,
        "transitions_per_second": completed / elapsed,
        "track_seeds": seeds,
        "fixed_action": 0.0,
        "used_for_learning": False,
    }


def measure(*, updates: int, physical_transitions: int) -> dict[str, object]:
    if not 2 <= updates <= 100:
        raise ValueError("synthetic updates must be in [2, 100]")
    config = GCSLConfig()
    learner = GCSLLearner(config, seed=123)
    rng = np.random.default_rng(123)
    batch = synthetic_batch(config, rng)
    learner.update(batch)  # measured warm-up; included in the preparation cap
    started = time.perf_counter()
    final = None
    for _ in range(updates - 1):
        final = learner.update(batch)
    update_elapsed = time.perf_counter() - started
    timed_updates = updates - 1
    states = batch.states[:1]
    goals = batch.goals[:1]
    inference_iterations = 2000
    started = time.perf_counter()
    for _ in range(inference_iterations):
        learner.deterministic_action(states, goals)
    inference_elapsed = time.perf_counter() - started
    replay = TrajectoryReplay(config.replay_capacity)
    replay.size = replay.capacity
    for field in (
        "states",
        "actions",
        "source_outcomes",
        "outcomes",
        "historical_rewards",
        "terminated",
        "truncated",
        "episode_ids",
        "episode_steps",
    ):
        getattr(replay, field).fill(0)
    episode_length = 1800
    replay.episode_ids[:] = np.arange(replay.capacity) // episode_length
    replay.episode_steps[:] = np.arange(replay.capacity) % episode_length
    replay.truncated[:] = replay.episode_steps == episode_length - 1
    eligible = np.arange(replay.capacity, dtype=np.int64)
    replay.eligible_size = len(eligible)
    replay._eligible_sources[: replay.eligible_size] = eligible
    replay_memory_bytes = replay.memory_bytes
    replay_sample_iterations = 500
    replay_rng = np.random.default_rng(9182)
    started = time.perf_counter()
    for _ in range(replay_sample_iterations):
        replay.sample(
            config.batch_size,
            gamma=config.future_discount,
            rng=replay_rng,
            task=TaskSpecification(140.0, 0.0),
        )
    replay_sample_elapsed = time.perf_counter() - started
    runtime = {
        "training_seed": 11,
        "transition_count": 0,
        "update_cycle_count": 0,
        "episode_id": 0,
        "episode_step": 0,
        "current_episode_ended": False,
        "observation": np.zeros(8),
        "current_outcome": np.zeros(4),
        "policy_action_rng_state": torch.Generator().get_state(),
        "replay_rng_state": np.random.default_rng(1).bit_generator.state,
        "track_rng_state": np.random.default_rng(2).bit_generator.state,
        "track_seeds_used": [],
        "diagnostics": DiagnosticsAccumulator().to_state(),
        "training_outcomes": [],
        "development_completed": [],
        "first_canonical_target_available_transition": None,
        "first_physical_canonical_success_transition": None,
    }
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "synthetic-full-replay.ckpt"
        save_checkpoint(
            checkpoint,
            learner_state=learner.state_dict(),
            replay=replay,
            environment=None,
            runtime=runtime,
            configuration_sha256="0" * 64,
            scientific_source_sha256={},
            execution_source_sha256={},
        )
        checkpoint_bytes = checkpoint_size_bytes(checkpoint)
    seconds_per_update = update_elapsed / timed_updates
    seconds_per_inference = inference_elapsed / inference_iterations
    physical = physical_probe(physical_transitions)
    simulator_seconds = (
        PLANNED_TRANSITIONS_PER_POLICY / physical["transitions_per_second"]
        if physical["transitions_per_second"]
        else None
    )
    measured_compute = (
        PLANNED_UPDATE_CYCLES * seconds_per_update
        + PLANNED_UPDATE_CYCLES
        * replay_sample_elapsed
        / replay_sample_iterations
        + PLANNED_TRANSITIONS_PER_POLICY * seconds_per_inference
    )
    return {
        "training_performed": False,
        "validation_tracks_used": False,
        "paper_tracks_used": False,
        "synthetic_updates": updates,
        "parameter_count": learner.parameter_count,
        "backend": {
            "device": str(learner.device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "batch_256": {
            "timed_updates": timed_updates,
            "elapsed_seconds": update_elapsed,
            "updates_per_second": timed_updates / update_elapsed,
            "seconds_per_update": seconds_per_update,
            "finite_final_metrics": bool(
                final is None or all(np.isfinite(value) for value in final.values())
            ),
        },
        "actor_inference": {
            "iterations": inference_iterations,
            "elapsed_seconds": inference_elapsed,
            "mean_seconds": seconds_per_inference,
        },
        "replay": {
            "capacity": replay.capacity,
            "allocated_bytes": replay_memory_bytes,
            "sample_iterations": replay_sample_iterations,
            "sample_elapsed_seconds": replay_sample_elapsed,
            "mean_sample_seconds": (
                replay_sample_elapsed / replay_sample_iterations
            ),
        },
        "checkpoint": {
            "synthetic_full_replay_bytes": checkpoint_bytes,
        },
        "simulator": physical,
        "extrapolation": {
            "one_policy_compute_seconds_excluding_simulator": measured_compute,
            "one_policy_simulator_seconds": simulator_seconds,
            "one_policy_total_seconds": (
                measured_compute + simulator_seconds
                if simulator_seconds is not None
                else None
            ),
            "three_policy_total_seconds_sequential": (
                3 * (measured_compute + simulator_seconds)
                if simulator_seconds is not None
                else None
            ),
            "development_and_io_overhead_included": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--physical-transitions", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(
        updates=args.updates, physical_transitions=args.physical_transitions
    )
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
