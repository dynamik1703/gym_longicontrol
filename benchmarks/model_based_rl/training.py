"""Execution-complete six-policy constrained Dyna/MBPO-style trainer.

Importing this module is inert. ``run_policy`` additionally checks the frozen
authorization flags, which are false in the preparation snapshot.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.scalar_sac.experiment import _base_environment

from .checkpointing import capture_rng_states
from .config import load_configuration
from .evaluation import evaluate_development
from .execution import RunLease, TrainingTrackStream, atomic_json, scientific_hashes
from .fsrl_adapter import mixed_policy_update, model_disabled_update
from .imagination import (
    RealEpisodePIDGate,
    RealReplaySource,
    RealSourceReplay,
    SyntheticReplay,
    exact_mixed_batch,
    generate_synthetic_transitions,
    refresh_due,
)
from .learned_model import ProbabilisticEnsemble
from .model_state import ModelState, TrackContext
from .model_training import RealModelDataset, RealModelExample, train_ensemble
from .physics_model import PhysicsDynamicsModel
from .recovery import (
    CHECKPOINT_SCHEMA_VERSION,
    HEARTBEAT_INTERVAL,
    apply_retention,
    collector_state,
    execution_hashes,
    heartbeat_payload,
    publish_latest_recovery,
    recovery_checkpoint_due,
    restore_collector,
    restore_policy_state,
    restore_runtime_rng,
    runtime_versions,
    verify_recovery_checkpoint,
    write_heartbeat,
    write_verified_checkpoint,
)


class TrainingSeedWrapper(gym.Wrapper):
    def __init__(self, environment: gym.Env, stream: TrainingTrackStream):
        super().__init__(environment)
        self.stream = stream
        self.seeds_used: list[int] = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise ValueError("Training track seeds are owned by TrainingTrackStream")
        seed = self.stream.next_seed()
        self.seeds_used.append(seed)
        return self.env.reset(seed=seed, options=options)


def state_from_environment(environment: gym.Env, episode_step: int) -> ModelState:
    base = environment.unwrapped
    metrics = base.episode_metrics
    return ModelState(
        vehicle=base.state,
        maximum_speed_excess_m_s=metrics.max_speed_violation_m_s,
        speed_violation_count=metrics.speed_violation_count,
        violation_active=bool(base._metrics.speed_excess_m_s > 0),  # noqa: SLF001
        integrated_speed_violation_m=metrics.integrated_speed_violation_m,
        episode_step=episode_step,
    )


class ModelTransitionRecorder(gym.Wrapper):
    """Observe internal model state without changing the public actor observation."""

    def __init__(self, environment: DenseDeadlineTaskWrapper):
        super().__init__(environment)
        self.episode_step = 0
        self.transition_id = 0
        self.records: list[tuple] = []
        self.completed_episodes = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.episode_step = 0
        return observation, info

    def step(self, action):
        base = self.env.unwrapped
        before = state_from_environment(self.env, self.episode_step)
        context = TrackContext(
            track=base.track,
            sensor_range_m=base.config.sensor_range_m,
            track_length_m=base.config.track_length_m,
            energy_factor=base.energy_factor,
        )
        before_observation = np.asarray(base._observation(), dtype=np.float64)  # noqa: SLF001
        result = self.env.step(action)
        observation, reward, terminated, truncated, info = result
        self.episode_step += 1
        after = state_from_environment(self.env, self.episode_step)
        self.records.append(
            (
                self.transition_id,
                before,
                float(np.asarray(action).reshape(-1)[0]),
                after,
                float(info["step_energy_kwh"]),
                context,
                before_observation,
            )
        )
        self.transition_id += 1
        if terminated or truncated:
            self.completed_episodes += 1
        return observation, reward, terminated, truncated, info

    def drain(self) -> list[tuple]:
        records, self.records = self.records, []
        return records


def _stochastic_actor(policy):
    def act(observation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        import torch
        from tianshou.data import Batch, to_numpy

        with torch.no_grad():
            logits, _state = policy.actor(
                np.asarray(observation, dtype=np.float32)[None, :],
                state=None,
                info=Batch(),
            )
        mean, std = (to_numpy(item)[0] for item in logits)
        raw = mean + std * rng.normal(size=mean.shape)
        action = np.tanh(raw)[None, :]
        return np.asarray(policy.map_action(action), dtype=np.float64).reshape(1)

    return act


def _policy_optimizer_state(policy) -> dict[str, Any]:
    state = {
        "actor": policy.actor_optim.state_dict(),
        "critics": policy.critics_optim.state_dict(),
    }
    if getattr(policy, "_is_auto_alpha", False):
        state["alpha"] = policy._alpha_optim.state_dict()  # noqa: SLF001
        state["log_alpha"] = policy._log_alpha.detach().cpu()  # noqa: SLF001
    return state


def _make_checkpoint_payload(
    *,
    condition: str,
    policy,
    collector,
    recorder,
    track_stream,
    real_model_data,
    real_sources,
    synthetic,
    model,
    model_rng,
    synthetic_rng,
    counters,
    diagnostics,
    hashes,
    attempt_id,
    training_seed,
    previous_completed,
    pid_updates,
    development_checkpoints_completed,
    last_recovery_checkpoint,
    last_scientific_checkpoint,
) -> dict[str, Any]:
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "training_seed": training_seed,
        "model_condition": condition,
        "policy": policy.state_dict(),
        "policy_optimizers": _policy_optimizer_state(policy),
        "lagrangian_states": [item.state_dict() for item in policy.lag_optims],
        "real_rl_replay": collector.buffer,
        "real_model_replay": real_model_data,
        "real_source_replay": real_sources,
        "synthetic_replay": synthetic,
        "environment_state": copy.deepcopy(recorder),
        "collector_state": collector_state(collector),
        "track_stream_state": track_stream.state_dict(),
        "counters": dict(counters),
        "diagnostics": diagnostics,
        "hashes": hashes,
        "runtime_versions": runtime_versions(),
        "previous_completed_episodes": previous_completed,
        "pid_real_episode_updates": pid_updates,
        "development_checkpoints_completed": sorted(
            development_checkpoints_completed
        ),
        "last_recovery_checkpoint": last_recovery_checkpoint,
        "last_scientific_checkpoint": last_scientific_checkpoint,
        "rng_states": capture_rng_states(
            model_rng=model_rng, synthetic_rng=synthetic_rng
        ),
    }
    if condition == "learned":
        model_state = model.state_dict()
        payload.update(
            {
                "learned_model": model_state["members"],
                "model_optimizer_states": model_state["optimizers"],
                "normalization": {
                    "input": model_state["input_normalization"],
                    "target": model_state["target_normalization"],
                },
                "learned_model_full_state": model_state,
            }
        )
    return payload


def run_policy(
    *,
    condition: str,
    training_seed: int,
    output_root: str | Path,
    device: str = "cpu",
    threads: int = 4,
    resume: bool = False,
    restart: bool = False,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Execute one authorized policy. Preparation snapshots always refuse this call."""

    configuration = load_configuration()
    authorization = configuration.raw["authorization"]
    if not (
        authorization["main_training_authorized"]
        and authorization["main_training_enabled"]
    ):
        raise PermissionError("Main MBRL training is frozen but not authorized/enabled")
    if condition not in configuration.raw["conditions"]:
        raise ValueError("Unauthorized model condition")
    if training_seed not in configuration.raw["training_seeds"]:
        raise ValueError("Unauthorized training seed")
    if resume != (checkpoint_path is not None):
        raise ValueError("Exact resume requires one explicit recovery checkpoint")
    import tianshou
    from tianshou.data import Collector, ReplayBuffer

    if tianshou.__version__ != configuration.v2.algorithm.tianshou_version:
        raise RuntimeError(
            f"Expected Tianshou {configuration.v2.algorithm.tianshou_version}, "
            f"found {tianshou.__version__}"
        )

    run_id = f"{condition}-seed-{training_seed}"
    with RunLease(
        output_root, run_id, resume=resume, restart=restart
    ) as lease:
        track_stream = TrainingTrackStream(training_seed)
        seeded = TrainingSeedWrapper(_base_environment(configuration.v2), track_stream)
        dense = DenseDeadlineTaskWrapper(
            seeded,
            task=configuration.v2.task,
            energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
            deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
            max_simulator_steps=configuration.v2.simulator_step_budget,
        )
        recorder = ModelTransitionRecorder(dense)
        agent, logger = build_agent(
            configuration.v2,
            recorder,
            training_seed=training_seed,
            device=device,
            threads=threads,
        )
        policy = agent.policy
        policy.train()
        collector = Collector(
            policy,
            recorder,
            ReplayBuffer(configuration.v2.algorithm.buffer_size),
            exploration_noise=True,
        )
        base = recorder.unwrapped
        physics = PhysicsDynamicsModel(base.vehicle, base.config)
        model_rng = np.random.default_rng(
            np.random.SeedSequence([training_seed, 0x4D4F444C])
        )
        synthetic_rng = np.random.default_rng(
            np.random.SeedSequence([training_seed, 0x53594E54])
        )
        model = (
            ProbabilisticEnsemble(
                configuration.model, base.config, seed=training_seed, device=device
            )
            if condition == "learned"
            else physics
        )
        real_model_data = RealModelDataset(configuration.v2.algorithm.buffer_size)
        real_sources = RealSourceReplay(configuration.v2.algorithm.buffer_size)
        synthetic = SyntheticReplay(
            configuration.imagination.synthetic_replay_capacity
        )
        pid_gate = RealEpisodePIDGate()
        counters = {
            "real_transitions": 0,
            "synthetic_transitions": 0,
            "rl_gradient_updates": 0,
            "model_gradient_updates": 0,
            "model_refreshes": 0,
        }
        diagnostics: list[dict[str, Any]] = []
        checkpoints = set(configuration.raw["real_transition_checkpoints"])
        final_checkpoint: Path | None = None
        previous_completed = 0
        development_checkpoints_completed: set[int] = set()
        last_recovery_checkpoint: str | None = None
        last_scientific_checkpoint: str | None = None
        if resume:
            assert checkpoint_path is not None
            payload, metadata = verify_recovery_checkpoint(
                checkpoint_path,
                configuration_sha256=configuration.configuration_sha256,
                expected_attempt_id=lease.attempt_id,
            )
            if payload["model_condition"] != condition:
                raise RuntimeError("Checkpoint model condition differs")
            if int(payload["training_seed"]) != training_seed:
                raise RuntimeError("Checkpoint training seed differs")
            restore_policy_state(policy, payload)
            restored_recorder = payload["environment_state"]
            recorder.close()
            restore_collector(
                collector,
                state=payload["collector_state"],
                recorder=restored_recorder,
                replay_buffer=payload["real_rl_replay"],
            )
            recorder = restored_recorder
            dense = recorder.env
            track_stream = dense.env.stream
            track_stream.load_state_dict(payload["track_stream_state"])
            base = recorder.unwrapped
            real_model_data = payload["real_model_replay"]
            real_sources = payload["real_source_replay"]
            synthetic = payload["synthetic_replay"]
            counters = dict(payload["counters"])
            diagnostics = payload["diagnostics"]
            previous_completed = int(payload["previous_completed_episodes"])
            pid_gate.real_episode_updates = int(payload["pid_real_episode_updates"])
            development_checkpoints_completed = set(
                int(value)
                for value in payload["development_checkpoints_completed"]
            )
            last_recovery_checkpoint = str(checkpoint_path)
            last_scientific_checkpoint = payload["last_scientific_checkpoint"]
            if last_scientific_checkpoint is not None:
                final_checkpoint = Path(last_scientific_checkpoint)
            if condition == "learned":
                model.load_state_dict(payload["learned_model_full_state"])
            restore_runtime_rng(
                payload, model_rng=model_rng, synthetic_rng=synthetic_rng
            )
            lease.attempt.setdefault("resume_events", [])[-1].update(
                {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": metadata["sha256"],
                    "transition_count": metadata["transition_count"],
                }
            )
            atomic_json(lease.manifest_path, lease.manifest)
        run_directory = Path(output_root) / run_id
        write_heartbeat(
            run_directory,
            heartbeat_payload(
                condition=condition,
                seed=training_seed,
                attempt_id=lease.attempt_id,
                counters=counters,
                recorder=recorder,
                synthetic=synthetic,
                status="RUNNING",
                last_recovery_checkpoint=last_recovery_checkpoint,
                last_scientific_checkpoint=last_scientific_checkpoint,
            ),
        )
        while counters["real_transitions"] < 300_000:
            stats = collector.collect(n_step=1)
            records = recorder.drain()
            if len(records) != 1 or int(stats["n/st"]) != 1:
                raise RuntimeError("Native 1:1 transition accounting changed")
            (
                transition_id,
                state,
                action,
                next_state,
                step_energy,
                context,
                obs,
            ) = records[0]
            real_model_data.append(
                RealModelExample(
                    transition_id, state, action, next_state, step_energy
                )
            )
            real_sources.append(
                RealReplaySource(transition_id, state, context, obs)
            )
            counters["real_transitions"] += 1
            real_count = counters["real_transitions"]
            if recorder.completed_episodes != previous_completed:
                completed = dense.last_completed_episode
                if completed is None:
                    raise RuntimeError("Physical episode completion was lost")
                pid_gate.update(
                    policy,
                    source="real",
                    stats={"cost": np.asarray(completed.costs)},
                    batch_size=configuration.v2.algorithm.batch_size,
                    buffer=collector.buffer,
                    update_per_step=configuration.v2.algorithm.update_per_step,
                )
                policy.post_update_fn(stats_train={"cost": completed.costs})
                previous_completed = recorder.completed_episodes
            if refresh_due(real_count, configuration.imagination):
                write_heartbeat(
                    run_directory,
                    heartbeat_payload(
                        condition=condition,
                        seed=training_seed,
                        attempt_id=lease.attempt_id,
                        counters=counters,
                        recorder=recorder,
                        synthetic=synthetic,
                        status="MODEL_REFRESH_RUNNING",
                        last_recovery_checkpoint=last_recovery_checkpoint,
                        last_scientific_checkpoint=last_scientific_checkpoint,
                    ),
                )
                report = None
                if condition == "learned":
                    report = train_ensemble(
                        model, real_model_data, rng=model_rng
                    )
                    counters["model_gradient_updates"] += report.gradient_updates
                generated = generate_synthetic_transitions(
                    model=model,
                    real_replay=real_sources,
                    actor=_stochastic_actor(policy),
                    count=configuration.imagination.synthetic_transitions_per_refresh,
                    rng=synthetic_rng,
                    vehicle=base.vehicle,
                    simulation_config=base.config,
                    energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
                    deadline_s=configuration.v2.task.max_time_s,
                    max_episode_steps=configuration.v2.max_episode_steps,
                )
                synthetic.extend(generated)
                counters["synthetic_transitions"] += len(generated)
                counters["model_refreshes"] += 1
                diagnostics.append(
                    {
                        "real_transitions": real_count,
                        "model_refresh": counters["model_refreshes"],
                        "model_training": None if report is None else asdict(report),
                        "synthetic_generated": len(generated),
                    }
                )
            target_updates = math.floor(
                real_count * configuration.v2.algorithm.update_per_step + 1e-12
            )
            while (
                counters["rl_gradient_updates"] < target_updates
                and len(collector.buffer) >= configuration.v2.algorithm.batch_size
            ):
                if real_count < configuration.imagination.warmup_real_transitions:
                    model_disabled_update(
                        policy,
                        collector.buffer,
                        configuration.v2.algorithm.batch_size,
                    )
                else:
                    real_stub = [object()] * configuration.imagination.real_batch_size
                    _real, model_rows = exact_mixed_batch(
                        real_stub,
                        synthetic,
                        config=configuration.imagination,
                        rng=synthetic_rng,
                    )
                    td = mixed_policy_update(policy, collector.buffer, model_rows)
                    diagnostics.append(
                        {
                            "real_transitions": real_count,
                            "fraction_real": 0.5,
                            "fraction_model": 0.5,
                            **td,
                        }
                    )
                counters["rl_gradient_updates"] += 1
            scientific_checkpoint = real_count in checkpoints
            recovery_checkpoint = recovery_checkpoint_due(real_count)
            if scientific_checkpoint:
                episodes, summary = evaluate_development(policy, configuration)
                checkpoint_dir = Path(output_root) / run_id / f"step-{real_count:06d}"
                atomic_json(
                    checkpoint_dir / "development.json",
                    {
                        "episodes": [asdict(item) for item in episodes],
                        "summary": asdict(summary),
                    },
                )
                development_checkpoints_completed.add(real_count)
            elif recovery_checkpoint:
                checkpoint_dir = (
                    Path(output_root)
                    / run_id
                    / "recovery"
                    / f"step-{real_count:06d}"
                )
            if scientific_checkpoint or recovery_checkpoint:
                checkpoint = checkpoint_dir / "checkpoint.pt"
                current_recovery = str(checkpoint)
                current_scientific = (
                    str(checkpoint)
                    if scientific_checkpoint
                    else last_scientific_checkpoint
                )
                payload = _make_checkpoint_payload(
                    condition=condition,
                    policy=policy,
                    collector=collector,
                    recorder=recorder,
                    track_stream=track_stream,
                    real_model_data=real_model_data,
                    real_sources=real_sources,
                    synthetic=synthetic,
                    model=model,
                    model_rng=model_rng,
                    synthetic_rng=synthetic_rng,
                    counters=counters,
                    diagnostics=diagnostics,
                    hashes={
                        "configuration": configuration.configuration_sha256,
                        "scientific_sources": scientific_hashes(),
                        "execution_sources": execution_hashes(),
                    },
                    attempt_id=lease.attempt_id,
                    training_seed=training_seed,
                    previous_completed=previous_completed,
                    pid_updates=pid_gate.real_episode_updates,
                    development_checkpoints_completed=(
                        development_checkpoints_completed
                    ),
                    last_recovery_checkpoint=current_recovery,
                    last_scientific_checkpoint=current_scientific,
                )
                if scientific_checkpoint and recovery_checkpoint:
                    kind = "scientific_and_recovery"
                elif scientific_checkpoint:
                    kind = "scientific"
                else:
                    kind = "recovery"
                metadata = write_verified_checkpoint(
                    checkpoint, payload, kind=kind
                )
                last_recovery_checkpoint = current_recovery
                last_scientific_checkpoint = current_scientific
                publish_latest_recovery(run_directory, metadata)
                lease.record_checkpoint(counters, metadata)
                if not scientific_checkpoint:
                    apply_retention(run_directory)
                if scientific_checkpoint:
                    final_checkpoint = checkpoint
            if (
                real_count % HEARTBEAT_INTERVAL == 0
                or scientific_checkpoint
                or recovery_checkpoint
            ):
                write_heartbeat(
                    run_directory,
                    heartbeat_payload(
                        condition=condition,
                        seed=training_seed,
                        attempt_id=lease.attempt_id,
                        counters=counters,
                        recorder=recorder,
                        synthetic=synthetic,
                        status="RUNNING",
                        last_recovery_checkpoint=last_recovery_checkpoint,
                        last_scientific_checkpoint=last_scientific_checkpoint,
                    ),
                )
        if counters["rl_gradient_updates"] != 30_000 or final_checkpoint is None:
            raise RuntimeError("Frozen policy budget/update count was not reached")
        if development_checkpoints_completed != checkpoints:
            raise RuntimeError("Frozen Development checkpoint schedule is incomplete")
        lease.manifest["runs"][run_id]["development_complete"] = True
        lease.complete(
            {**counters, "development_complete": True}, final_checkpoint
        )
        write_heartbeat(
            run_directory,
            heartbeat_payload(
                condition=condition,
                seed=training_seed,
                attempt_id=lease.attempt_id,
                counters=counters,
                recorder=recorder,
                synthetic=synthetic,
                status="COMPLETED",
                last_recovery_checkpoint=last_recovery_checkpoint,
                last_scientific_checkpoint=last_scientific_checkpoint,
            ),
        )
        return final_checkpoint
