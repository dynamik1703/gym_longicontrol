"""Mandatory actual-workload benchmark for the M=60 compute gate."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.scalar_sac.experiment import _base_environment
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.track import Track

from .adapter import ModelState, RealPhysicsReplay, RecordedTransition, TrackContext
from .checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_states,
    collector_state,
    policy_optimizer_state,
    write_verified_checkpoint,
)
from .config import load_configuration
from .diagnostics import TargetAccumulator
from .execution import runtime_versions, source_hashes
from .physics_rollout import batched_step, pack_states
from .value_expansion import build_expanded_targets, real_replay_update


def _fixture(
    *, rows: int = 2048, training_seed: int = 811
) -> tuple[Any, RealPhysicsReplay, TrackContext, Any, Any]:
    from tianshou.data import Batch, ReplayBuffer

    from benchmarks.model_based_rl.adapter import model_step
    from benchmarks.model_based_rl.physics_model import PhysicsDynamicsModel

    configuration = load_configuration()
    environment = DenseDeadlineTaskWrapper(
        _base_environment(configuration.v2),
        task=configuration.v2.task,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
    )
    policy = build_agent(
        configuration.v2,
        environment,
        training_seed=training_seed,
        device="cpu",
        threads=4,
    )[0].policy
    policy.train()
    base = environment.unwrapped
    track = Track(
        np.asarray([0.0, 200.0, 400.0, 600.0, 800.0]),
        np.asarray([80.0, 40.0, 70.0, 30.0, 60.0]) / 3.6,
    )
    context = TrackContext(
        track,
        base.config.sensor_range_m,
        base.config.track_length_m,
        base.energy_factor,
    )
    physics = PhysicsDynamicsModel(base.vehicle, base.config)
    buffer = ReplayBuffer(max(rows + 16, 4096))
    sidecar = RealPhysicsReplay(max(rows + 16, 4096))
    state = ModelState(VehicleState())
    rng = np.random.default_rng(919)
    for transition_id in range(rows):
        action = float(np.sin(transition_id * 0.031))
        projected = model_step(
            physics,
            state,
            action,
            context=context,
            vehicle=base.vehicle,
            config=base.config,
            energy_scale_kwh=0.25,
            deadline_s=140.0,
            max_episode_steps=1800,
            source_real_transition_id=transition_id,
            rng=rng,
        )
        buffer.add(
            Batch(
                obs=projected.observation.astype(np.float32),
                act=np.asarray([action], dtype=np.float32),
                rew=np.float32(projected.objective),
                terminated=projected.terminated,
                truncated=projected.truncated,
                obs_next=projected.next_observation.astype(np.float32),
                info=Batch(
                    cost=np.asarray(projected.costs, dtype=np.float32),
                    real_transition_id=transition_id,
                ),
                policy=Batch(),
            )
        )
        sidecar.append(
            RecordedTransition(
                transition_id=transition_id,
                state=state,
                action=action,
                next_state=projected.next_state,
                context=context,
                observation=projected.observation,
                next_observation=projected.next_observation,
            )
        )
        state = projected.next_state
        if projected.terminated or projected.truncated:
            state = ModelState(VehicleState())
    return policy, sidecar, context, buffer, environment


def _time_repeated(function, repeats: int) -> tuple[float, list[float]]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return float(np.mean(samples)), samples


def run_benchmark(
    *,
    target_repeats: int = 5,
    update_repeats: int = 5,
    collection_transitions: int = 512,
) -> dict[str, Any]:
    """Measure warmed execution and return reproducible extrapolations."""

    import torch
    from tianshou.data import Collector, ReplayBuffer

    configuration = load_configuration()
    torch.set_num_threads(4)
    policy, sidecar, context, buffer, environment = _fixture()
    baseline_policy, _unused_sidecar, _unused_context, baseline_buffer, baseline_env = (
        _fixture(training_seed=812)
    )
    base = environment.unwrapped
    rollout_rng = np.random.default_rng([811, 0x4D564552])

    fixture_states = [
        sidecar.get(index).next_state for index in range(256, 512)
    ]
    physics_state = pack_states(fixture_states)
    contexts = [context] * 256
    actions = np.linspace(-1.0, 1.0, 256)

    def physics_probe():
        nonlocal physics_state
        transition = batched_step(
            physics_state,
            actions,
            contexts=contexts,
            vehicle=base.vehicle,
            config=base.config,
        )
        physics_state = transition.state

    for _ in range(10):
        physics_probe()
    physics_seconds, physics_samples = _time_repeated(physics_probe, 200)
    physics_steps = 200 * 256

    sampled_batch, sampled_indices = buffer.sample(256)

    def target_probe():
        build_expanded_targets(
            policy,
            buffer[sampled_indices],
            buffer,
            sampled_indices,
            sidecar,
            model_steps=configuration.model_expansion_steps,
            rollout_rng=rollout_rng,
            vehicle=base.vehicle,
            simulation_config=base.config,
        )

    target_probe()
    target_seconds, target_samples = _time_repeated(target_probe, target_repeats)

    np.random.seed(123)
    torch.manual_seed(123)
    baseline_policy.update(256, baseline_buffer)

    def baseline_update():
        baseline_policy.update(256, baseline_buffer)

    baseline_seconds, baseline_samples = _time_repeated(
        baseline_update, update_repeats
    )

    real_replay_update(
        policy,
        buffer,
        sidecar,
        model_steps=configuration.model_expansion_steps,
        rollout_rng=rollout_rng,
        vehicle=base.vehicle,
        simulation_config=base.config,
    )

    def mve_update():
        real_replay_update(
            policy,
            buffer,
            sidecar,
            model_steps=configuration.model_expansion_steps,
            rollout_rng=rollout_rng,
            vehicle=base.vehicle,
            simulation_config=base.config,
        )

    mve_seconds, mve_samples = _time_repeated(mve_update, update_repeats)

    collection_environment = DenseDeadlineTaskWrapper(
        _base_environment(configuration.v2),
        task=configuration.v2.task,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
        initial_seed=0,
    )
    collection_policy = build_agent(
        configuration.v2,
        collection_environment,
        training_seed=901,
        device="cpu",
        threads=4,
    )[0].policy
    collection_policy.train()
    collector = Collector(
        collection_policy,
        collection_environment,
        ReplayBuffer(collection_transitions + 16),
        exploration_noise=True,
    )
    collection_started = time.perf_counter()
    collection_stats = collector.collect(n_step=collection_transitions)
    collection_seconds = time.perf_counter() - collection_started

    counters = {
        "real_transitions": len(buffer),
        "synthetic_transitions": 0,
        "model_target_transitions": 0,
        "rl_gradient_updates": update_repeats,
    }
    checkpoint_payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "condition": "physics_mve",
        "training_seed": 811,
        "attempt_id": "resource-benchmark",
        "configuration_sha256": configuration.configuration_sha256,
        "source_hashes": source_hashes(),
        "runtime_versions": runtime_versions(),
        "policy": copy.deepcopy(policy.state_dict()),
        "policy_optimizers": copy.deepcopy(policy_optimizer_state(policy)),
        "lagrangian_states": [
            copy.deepcopy(item.state_dict()) for item in policy.lag_optims
        ],
        "real_replay": buffer,
        "physics_sidecar": sidecar,
        "environment_state": {},
        "collector_state": collector_state(collector),
        "track_stream_state": {},
        "counters": counters,
        "diagnostics": [],
        "target_accumulator": TargetAccumulator(),
        "anticipation_probes": [],
        "rng_states": capture_rng_states(rollout_rng),
        "pid_real_episode_updates": 0,
        "previous_completed_episodes": 0,
        "development_checkpoints_completed": [],
    }
    with TemporaryDirectory(prefix="mve-checkpoint-") as temporary:
        checkpoint = Path(temporary) / "checkpoint.pt"
        checkpoint_started = time.perf_counter()
        checkpoint_metadata = write_verified_checkpoint(
            checkpoint, checkpoint_payload
        )
        checkpoint_roundtrip_seconds = time.perf_counter() - checkpoint_started

    updates = configuration.raw["compute_gate"]["updates_per_policy"]
    collection_rate = int(collection_stats["n/st"]) / collection_seconds
    collection_per_policy_s = 300_000 / collection_rate
    # Six Development evaluations, conservatively allowing 1,800 steps on 9 tracks.
    development_steps_per_policy = 6 * 9 * configuration.v2.max_episode_steps
    development_per_policy_s = development_steps_per_policy / collection_rate
    checkpoints_per_policy = 300_000 // configuration.raw[
        "recovery_checkpoint_interval"
    ]
    # The 2,048-row checkpoint is intentionally scaled by the 100k replay ratio.
    checkpoint_scale = configuration.v2.algorithm.buffer_size / len(buffer)
    checkpoint_per_policy_s = (
        checkpoint_roundtrip_seconds * checkpoint_scale * checkpoints_per_policy
    )
    baseline_policy_s = (
        collection_per_policy_s
        + updates * baseline_seconds
        + development_per_policy_s
        + checkpoint_per_policy_s
    )
    mve_policy_s = (
        collection_per_policy_s
        + updates * mve_seconds
        + development_per_policy_s
        + checkpoint_per_policy_s
    )
    serial_s = 3 * baseline_policy_s + 3 * mve_policy_s
    result = {
        "schema_version": 1,
        "measured_at_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "runtime": runtime_versions(),
        "protocol": {
            "batch_size": 256,
            "model_expansion_steps": configuration.model_expansion_steps,
            "updates_per_policy": updates,
            "real_transitions_per_policy": 300_000,
            "conditions": 2,
            "seeds": 3,
            "threads": 4,
        },
        "measurements": {
            "exact_physics": {
                "state_steps": physics_steps,
                "seconds": sum(physics_samples),
                "state_steps_per_second": physics_steps / sum(physics_samples),
                "mean_batch_step_s": physics_seconds,
            },
            "target_construction": {
                "repeats": target_repeats,
                "mean_s": target_seconds,
                "samples_s": target_samples,
            },
            "real_only_saclag_update": {
                "repeats": update_repeats,
                "mean_s": baseline_seconds,
                "samples_s": baseline_samples,
            },
            "mve_saclag_update": {
                "repeats": update_repeats,
                "mean_s": mve_seconds,
                "samples_s": mve_samples,
            },
            "actor_environment_collection": {
                "transitions": int(collection_stats["n/st"]),
                "seconds": collection_seconds,
                "transitions_per_second": collection_rate,
            },
            "checkpoint": {
                **checkpoint_metadata,
                "verified_roundtrip_s": checkpoint_roundtrip_seconds,
                "fixture_replay_rows": len(buffer),
                "projection_scale_to_100k": checkpoint_scale,
            },
        },
        "projection": {
            "collection_per_policy_h": collection_per_policy_s / 3600.0,
            "development_per_policy_h": development_per_policy_s / 3600.0,
            "checkpoint_per_policy_h": checkpoint_per_policy_s / 3600.0,
            "baseline_policy_h": baseline_policy_s / 3600.0,
            "mve_policy_h": mve_policy_s / 3600.0,
            "six_policy_serial_h": serial_s / 3600.0,
            "cpu_hours_at_four_threads": serial_s / 3600.0 * 4,
        },
        "gate": {
            "maximum_mve_policy_h": configuration.raw["compute_gate"][
                "maximum_mve_policy_walltime_h"
            ],
            "maximum_six_policy_serial_h": configuration.raw["compute_gate"][
                "maximum_six_policy_serial_walltime_h"
            ],
            "mve_policy_pass": mve_policy_s / 3600.0 <= 12.0,
            "six_policy_serial_pass": serial_s / 3600.0 <= 36.0,
            "ready_for_main_training": (
                mve_policy_s / 3600.0 <= 12.0 and serial_s / 3600.0 <= 36.0
            ),
        },
        "preparation_simulator_transitions": int(collection_stats["n/st"]),
    }
    environment.close()
    baseline_env.close()
    collection_environment.close()
    return result
