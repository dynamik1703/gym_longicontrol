"""Bounded preparation-only resource benchmark for the frozen GCSL design."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import gym_longicontrol  # noqa: F401
from gym_longicontrol.domain.task import TaskSpecification

from .checkpointing import load_checkpoint, save_checkpoint
from .config import GCSLConfig, load_configuration
from .diagnostics import DiagnosticsAccumulator
from .evaluation import raw_outcome
from .goal_adapter import OutcomeScales, augment_policy_state, canonical_command
from .learner import GCSLLearner
from .replay import TrajectoryReplay

TASK = TaskSpecification(140.0, 0.0)
FIXTURE_SEED_START = 98_000_000
MAX_PREPARATION_SIMULATOR_TRANSITIONS = 5_000


def _synthetic_replay(transitions: int, config: GCSLConfig) -> TrajectoryReplay:
    replay = TrajectoryReplay(config.replay_capacity)
    rng = np.random.default_rng(0x4743534C)
    episode_length = 100
    for sequence in range(transitions):
        episode = sequence // episode_length
        step = sequence % episode_length
        source_position = float(step * 5)
        position = float((step + 1) * 5)
        replay.append(
            state=rng.normal(size=config.state_dim).astype(np.float32),
            action=rng.uniform(-0.95, 0.95, size=config.action_dim).astype(np.float32),
            source_outcome=np.asarray(
                (
                    source_position,
                    max(0.0, source_position - 5.0),
                    step * 0.1,
                    0.0,
                ),
                dtype=np.float64,
            ),
            outcome=np.asarray(
                (position, source_position, (step + 1) * 0.1, 0.0),
                dtype=np.float64,
            ),
            historical_reward=0.0,
            terminated=False,
            truncated=step == episode_length - 1,
            episode_id=episode,
            episode_step=step,
        )
    return replay


def _runtime_state() -> dict[str, Any]:
    return {
        "training_seed": 11,
        "attempt_id": "resource-fixture-not-a-training-attempt",
        "transition_count": 0,
        "update_count": 0,
        "episode_id": 0,
        "episode_step": 0,
        "pending_update": False,
        "pending_reset": False,
        "observation": np.zeros(8, dtype=np.float64),
        "current_outcome": np.zeros(4, dtype=np.float64),
        "policy_action_rng_state": torch.Generator().get_state(),
        "replay_rng_state": np.random.default_rng(1).bit_generator.state,
        "task_stream_state": {"training_seed": 11, "cursor": 0},
        "current_task": {},
        "training_exposures": [],
        "diagnostics": DiagnosticsAccumulator().to_state(),
        "training_outcomes": [],
        "diagnostic_checkpoints_completed": [],
        "first_canonical_support_transition": None,
        "first_canonical_success_transition": None,
        "walltime_state": {"completed_segments_s": 0.0},
    }


def _collection_probe(transitions: int, learner: GCSLLearner) -> dict[str, Any]:
    if not 0 <= transitions <= MAX_PREPARATION_SIMULATOR_TRANSITIONS:
        raise ValueError("preparation collection fixture exceeds 5,000 transitions")
    if transitions == 0:
        return {
            "transitions": 0,
            "elapsed_s": 0.0,
            "transitions_per_s": None,
            "fixture_seeds": [],
        }
    environment = gym.make("StochasticTrack-v1", max_episode_steps=1800)
    generator = torch.Generator().manual_seed(0x5245534F)
    fixture_seeds: list[int] = []
    completed = 0
    started = time.perf_counter()
    try:
        fixture_seed = FIXTURE_SEED_START
        observation, info = environment.reset(seed=fixture_seed)
        fixture_seeds.append(fixture_seed)
        outcome = raw_outcome(info, float(info["position_m"]))
        while completed < transitions:
            state = augment_policy_state(observation, outcome, OutcomeScales())
            action = learner.sample_action(
                state[None, :],
                canonical_command()[None, :],
                generator=generator,
            )[0]
            previous_position = float(outcome[0])
            observation, _reward, terminated, truncated, info = environment.step(action)
            outcome = raw_outcome(info, previous_position)
            completed += 1
            if (terminated or truncated) and completed < transitions:
                fixture_seed += 1
                observation, info = environment.reset(seed=fixture_seed)
                fixture_seeds.append(fixture_seed)
                outcome = raw_outcome(info, float(info["position_m"]))
    finally:
        environment.close()
    elapsed = time.perf_counter() - started
    return {
        "transitions": completed,
        "elapsed_s": elapsed,
        "transitions_per_s": completed / elapsed,
        "fixture_seeds": fixture_seeds,
        "data_retained": False,
        "used_for_learning": False,
        "outcomes_interpreted_as_policy_performance": False,
    }


def measure(
    *,
    synthetic_updates: int = 30,
    collection_transitions: int = 1_000,
    replay_transitions: int = 5_000,
) -> dict[str, Any]:
    if not 2 <= synthetic_updates <= 100:
        raise ValueError("synthetic update count must be in [2, 100]")
    if not 1 <= replay_transitions <= 10_000:
        raise ValueError("synthetic replay rows must be in [1, 10,000]")
    configuration = load_configuration()
    config = configuration.algorithm
    learner = GCSLLearner(config, seed=123)
    replay = _synthetic_replay(replay_transitions, config)
    replay_rng = np.random.default_rng(9182)

    sampling_iterations = 200
    started = time.perf_counter()
    for _ in range(sampling_iterations):
        sampled = replay.sample(
            config.batch_size,
            gamma=config.future_discount,
            rng=replay_rng,
            task=TASK,
        )
    sampling_elapsed = time.perf_counter() - started

    learner.update(sampled.batch)
    started = time.perf_counter()
    final_metrics = None
    for _ in range(synthetic_updates - 1):
        final_metrics = learner.update(sampled.batch)
    update_elapsed = time.perf_counter() - started
    timed_updates = synthetic_updates - 1

    states = sampled.batch.states[:1]
    goals = sampled.batch.goals[:1]
    inference_iterations = 2_000
    started = time.perf_counter()
    for _ in range(inference_iterations):
        learner.deterministic_action(states, goals)
    inference_elapsed = time.perf_counter() - started

    with tempfile.TemporaryDirectory() as directory:
        fixture_path = Path(directory) / "fixture.pkl"
        empty_path = Path(directory) / "empty.pkl"
        fixture_metadata = save_checkpoint(
            fixture_path,
            learner_state=learner.state_dict(),
            replay=replay,
            environment=None,
            runtime=_runtime_state(),
            configuration_sha256="a" * 64,
            source_hashes={"resource_fixture": "b" * 64},
        )
        empty_metadata = save_checkpoint(
            empty_path,
            learner_state=learner.state_dict(),
            replay=TrajectoryReplay(config.replay_capacity),
            environment=None,
            runtime=_runtime_state(),
            configuration_sha256="a" * 64,
            source_hashes={"resource_fixture": "b" * 64},
        )
        started = time.perf_counter()
        loaded = load_checkpoint(
            fixture_path,
            expected_sha256=fixture_metadata["sha256"],
            expected_configuration_sha256="a" * 64,
            expected_source_hashes={"resource_fixture": "b" * 64},
            expected_attempt_id="resource-fixture-not-a-training-attempt",
        )
        restore_s = time.perf_counter() - started
        assert loaded.replay.size == replay_transitions

    collection = _collection_probe(collection_transitions, learner)
    seconds_per_collection = collection["elapsed_s"] / collection["transitions"]
    seconds_per_sample = sampling_elapsed / sampling_iterations
    seconds_per_update = update_elapsed / timed_updates
    seconds_per_inference = inference_elapsed / inference_iterations

    fixed_bytes = empty_metadata["size_bytes"]
    bytes_per_replay_row = (
        fixture_metadata["size_bytes"] - fixed_bytes
    ) / replay_transitions
    projected_full_checkpoint_bytes = fixed_bytes + 300_000 * bytes_per_replay_row
    projected_full_checkpoint_write_s = max(
        empty_metadata["write_s"],
        fixture_metadata["write_s"]
        * projected_full_checkpoint_bytes
        / fixture_metadata["size_bytes"],
    )

    updates_per_policy = configuration.updates_per_policy
    one_policy_collection_s = 300_000 * seconds_per_collection
    one_policy_optimizer_s = updates_per_policy * (
        seconds_per_sample + seconds_per_update
    )
    one_policy_checkpoint_s = 30 * projected_full_checkpoint_write_s
    one_policy_training_s = (
        one_policy_collection_s + one_policy_optimizer_s + one_policy_checkpoint_s
    )

    maximum_episode_steps = 1_800
    diagnostic_transitions_per_set = 18 * maximum_episode_steps
    one_diagnostic_set_s = diagnostic_transitions_per_set * (seconds_per_collection)
    all_diagnostic_sets = 3 * 6
    all_diagnostic_s = all_diagnostic_sets * one_diagnostic_set_s
    gcsl_validation_s = 3 * one_diagnostic_set_s
    constrained_contextual_validation_s = 3 * one_diagnostic_set_s
    three_policy_total_s = 3 * one_policy_training_s + all_diagnostic_s

    checkpoints_retained_per_policy = 7
    projected_checkpoint_artifacts = (
        3 * checkpoints_retained_per_policy * projected_full_checkpoint_bytes
    )
    projected_noncheckpoint_artifacts = 512 * 1024 * 1024
    projected_artifacts_bytes = (
        projected_checkpoint_artifacts + projected_noncheckpoint_artifacts
    )

    walltime_gate_s = 12 * 60 * 60
    disk_gate_bytes = 12 * 1024**3
    result = {
        "schema_version": 1,
        "scientific_configuration_sha256": configuration.configuration_sha256,
        "final_selected_budget": {
            "name": "A_CRL_MATCHED",
            "updates_per_policy": updates_per_policy,
            "batch_size": config.batch_size,
            "sampled_hindsight_tuples_per_policy": (
                updates_per_policy * config.batch_size
            ),
        },
        "preparation_only": {
            "main_training_performed": False,
            "policy_evaluation_performed": False,
            "gcsl_diagnostic_episodes": 0,
            "gcsl_validation_episodes": 0,
            "constrained_validation_episodes": 0,
            "paper_content_instantiated": False,
            "synthetic_updates": synthetic_updates,
            "simulator_transitions": collection_transitions,
        },
        "backend": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(learner.device),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "measurements": {
            "collection": collection,
            "actor_inference": {
                "iterations": inference_iterations,
                "elapsed_s": inference_elapsed,
                "mean_s": seconds_per_inference,
            },
            "replay_hindsight_sampling": {
                "fixture_transitions": replay_transitions,
                "iterations": sampling_iterations,
                "elapsed_s": sampling_elapsed,
                "mean_s": seconds_per_sample,
            },
            "supervised_update": {
                "timed_updates": timed_updates,
                "elapsed_s": update_elapsed,
                "mean_s": seconds_per_update,
                "finite_final_metrics": bool(
                    final_metrics is not None
                    and all(np.isfinite(value) for value in final_metrics.values())
                ),
            },
            "checkpoint": {
                "fixture_replay_rows": replay_transitions,
                "fixture_size_bytes": fixture_metadata["size_bytes"],
                "fixture_write_s": fixture_metadata["write_s"],
                "fixture_restore_s": restore_s,
                "empty_size_bytes": fixed_bytes,
                "empty_write_s": empty_metadata["write_s"],
                "estimated_bytes_per_replay_row": bytes_per_replay_row,
                "projected_full_size_bytes": projected_full_checkpoint_bytes,
                "projected_full_write_s": projected_full_checkpoint_write_s,
            },
            "diagnostic_evaluation": {
                "mode": (
                    "projection from measured collection throughput; "
                    "no policy evaluation was run"
                ),
                "maximum_transitions_per_18_episode_set": (
                    diagnostic_transitions_per_set
                ),
                "projected_s_per_set": one_diagnostic_set_s,
            },
        },
        "projections": {
            "one_300k_policy_training_s_excluding_diagnostics": (one_policy_training_s),
            "three_policy_training_s_excluding_diagnostics": (
                3 * one_policy_training_s
            ),
            "all_18_scheduled_diagnostic_sets_s": all_diagnostic_s,
            "three_policy_main_training_s_including_diagnostics": (
                three_policy_total_s
            ),
            "future_gcsl_validation_54_episodes_s": gcsl_validation_s,
            "future_constrained_contextual_validation_54_episodes_s": (
                constrained_contextual_validation_s
            ),
            "retained_artifacts_bytes": projected_artifacts_bytes,
            "retained_artifacts_gb_decimal": projected_artifacts_bytes / 1e9,
        },
        "hard_gates": {
            "three_policy_main_training_walltime_limit_h": 12.0,
            "three_policy_main_training_projected_h": three_policy_total_s / 3600,
            "walltime_pass": three_policy_total_s <= walltime_gate_s,
            "retained_artifacts_limit_gb": 12.0,
            "retained_artifacts_projected_gb_decimal": (
                projected_artifacts_bytes / 1e9
            ),
            "disk_pass": projected_artifacts_bytes <= disk_gate_bytes,
            "computational_feasibility_verified": (
                three_policy_total_s <= walltime_gate_s
                and projected_artifacts_bytes <= disk_gate_bytes
            ),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-updates", type=int, default=30)
    parser.add_argument("--collection-transitions", type=int, default=1_000)
    parser.add_argument("--replay-transitions", type=int, default=5_000)
    args = parser.parse_args()
    result = measure(
        synthetic_updates=args.synthetic_updates,
        collection_transitions=args.collection_transitions,
        replay_transitions=args.replay_transitions,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
