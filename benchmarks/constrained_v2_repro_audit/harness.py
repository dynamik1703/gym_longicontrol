"""Bounded deterministic fixtures for the Constrained-V2 reproducibility audit.

Nothing in this module trains a policy. Track fingerprints use reset only, and the
collector fixture uses a one-step synthetic environment rather than LongiControl.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_rl_v2.config import load_configuration
from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.constrained_rl_v2.experiment import _training_environment
from benchmarks.multistep_physics_mbrl.adapter import (
    RealTransitionRecorder,
    TrainingSeedWrapper,
)
from benchmarks.multistep_physics_mbrl.execution import TrainingTrackStream
from benchmarks.scalar_sac.experiment import _base_environment

TRAINING_SEEDS = (11, 29, 47)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(named_tensors: Iterable[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(named_tensors):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _module_sha256(module: Any) -> str:
    return _tensor_sha256(module.state_dict().items())


def _track_row(environment: Any, observation: np.ndarray, index: int) -> dict[str, Any]:
    base = environment.unwrapped
    payload = {
        "positions_m": np.asarray(base.track.positions_m).tolist(),
        "limits_m_s": np.asarray(base.track.limits_m_s).tolist(),
        "energy_factor": float(base.energy_factor),
        "initial_observation": np.asarray(observation).tolist(),
    }
    return {
        "episode_index": index,
        "sha256": _json_sha256(payload),
        **payload,
    }


def historical_track_fingerprints(
    training_seed: int, count: int = 100
) -> list[dict[str, Any]]:
    """Reconstruct used historical tracks under the installed runtime.

    FSRL ``FastCollector.collect(n_episode=1)`` resets a completed environment and
    then calls ``reset_env`` again before returning. The first of those two tracks is
    discarded. This routine mirrors that exact reset consumption without stepping the
    simulator. It is a reconstruction, not proof of historical bytes, because the
    historical NumPy and Gymnasium versions were not recorded.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    configuration = load_configuration()
    environment = _training_environment(configuration, training_seed)
    rows: list[dict[str, Any]] = []
    try:
        observation, _ = environment.reset()
        for index in range(count):
            rows.append(_track_row(environment, observation, index))
            if index + 1 < count:
                environment.reset()  # reset performed for the terminal environment
                observation, _ = environment.reset()  # FastCollector final reset
    finally:
        environment.close()
    return rows


def current_track_fingerprints(
    training_seed: int, count: int = 100
) -> list[dict[str, Any]]:
    """Reconstruct the current MVE runner's first used training tracks."""

    if count <= 0:
        raise ValueError("count must be positive")
    configuration = load_configuration()
    stream = TrainingTrackStream(training_seed)
    seeded = TrainingSeedWrapper(_base_environment(configuration), stream)
    environment = DenseDeadlineTaskWrapper(
        seeded,
        task=configuration.task,
        energy_scale_kwh=configuration.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.deadline_cost.normalization_s,
        max_simulator_steps=configuration.simulator_step_budget,
    )
    rows: list[dict[str, Any]] = []
    try:
        for index in range(count):
            observation, _ = environment.reset()
            rows.append(
                {
                    **_track_row(environment, observation, index),
                    "explicit_reset_seed": seeded.seeds_used[-1],
                }
            )
    finally:
        environment.close()
    return rows


def _policy_fingerprint(policy: Any) -> dict[str, Any]:
    alpha = float(policy._alpha.detach().cpu())  # noqa: SLF001
    return {
        "actor_sha256": _module_sha256(policy.actor),
        "critic_sha256": [_module_sha256(item) for item in policy.critics],
        "target_critic_sha256": [_module_sha256(item) for item in policy.critics_old],
        "all_tensor_sha256": _tensor_sha256(
            (name, value)
            for name, value in policy.state_dict().items()
            if hasattr(value, "detach")
        ),
        "alpha": alpha,
        "target_entropy": float(policy._target_entropy),  # noqa: SLF001
        "lagrangian": [float(item.get_lag()) for item in policy.lag_optims],
    }


def _build_historical_policy(training_seed: int):
    configuration = load_configuration()
    environment = _training_environment(configuration, training_seed)
    agent, _ = build_agent(
        configuration,
        environment,
        training_seed=training_seed,
        device="cpu",
        threads=2,
    )
    return environment, agent.policy


def _build_current_policy(training_seed: int):
    configuration = load_configuration()
    stream = TrainingTrackStream(training_seed)
    seeded = TrainingSeedWrapper(_base_environment(configuration), stream)
    dense = DenseDeadlineTaskWrapper(
        seeded,
        task=configuration.task,
        energy_scale_kwh=configuration.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.deadline_cost.normalization_s,
        max_simulator_steps=configuration.simulator_step_budget,
    )
    environment = RealTransitionRecorder(dense)
    agent, _ = build_agent(
        configuration,
        environment,
        training_seed=training_seed,
        device="cpu",
        threads=2,
    )
    return environment, agent.policy


def initial_policy_fingerprints(training_seed: int) -> dict[str, Any]:
    """Compare both constructors under one installed runtime, before collection."""

    historical_environment, historical_policy = _build_historical_policy(training_seed)
    try:
        historical = _policy_fingerprint(historical_policy)
    finally:
        historical_environment.close()
    current_environment, current_policy = _build_current_policy(training_seed)
    try:
        current = _policy_fingerprint(current_policy)
    finally:
        current_environment.close()
    return {
        "training_seed": training_seed,
        "scope": (
            "current-runtime constructor reconstruction; not a historical checkpoint"
        ),
        "historical_constructor": historical,
        "current_constructor": current,
        "identical": historical == current,
    }


def divergence_ladder(audit_seed: int = 0) -> dict[str, Any]:
    """Stop the end-to-end ladder at its first deterministic divergence."""

    initialized = initial_policy_fingerprints(audit_seed)
    historical = historical_track_fingerprints(audit_seed, count=1)[0]
    current = current_track_fingerprints(audit_seed, count=1)[0]
    observation_equal = (
        historical["initial_observation"] == current["initial_observation"]
    )
    levels: list[dict[str, Any]] = [
        {
            "level": 1,
            "name": "initialized network parameters",
            "equal": initialized["identical"],
            "historical_sha256": initialized["historical_constructor"][
                "all_tensor_sha256"
            ],
            "current_sha256": initialized["current_constructor"]["all_tensor_sha256"],
        },
        {
            "level": 2,
            "name": "first generated training track",
            "equal": historical["sha256"] == current["sha256"],
            "historical_sha256": historical["sha256"],
            "current_sha256": current["sha256"],
            "current_explicit_reset_seed": current["explicit_reset_seed"],
        },
        {
            "level": 3,
            "name": "first reset observation",
            "equal": observation_equal,
            "historical": historical["initial_observation"],
            "current": current["initial_observation"],
        },
    ]
    for level, name in enumerate(
        (
            "first stochastic action",
            "first environment transition",
            "first replay insertion",
            "first sampled replay indices",
            "first target tensors",
            "first gradient update",
            "actor/critic hashes after first update",
            "first completed real episode",
            "first PID update",
        ),
        start=4,
    ):
        levels.append(
            {
                "level": level,
                "name": name,
                "status": "NOT_RUN",
                "reason": "stopped after earlier track divergence",
            }
        )
    return {
        "audit_seed": audit_seed,
        "seed_is_reserved": False,
        "runtime_scope": "current-runtime reconstruction of both source paths",
        "earliest_divergence_level": 2,
        "earliest_divergence": "first generated training track",
        "longicontrol_simulator_transitions": 0,
        "levels": levels,
    }


def collector_reset_counts() -> dict[str, Any]:
    """Measure reset consumption with the installed collectors and a dummy env."""

    import gymnasium as gym
    from fsrl.data import FastCollector
    from tianshou.data import Batch, Collector, ReplayBuffer

    class OneStepEnvironment(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

        def __init__(self):
            self.reset_count = 0

        def reset(self, *, seed=None, options=None):
            del seed, options
            self.reset_count += 1
            return np.asarray([self.reset_count], dtype=np.float32), {}

        def step(self, action):
            del action
            return (
                np.zeros(1, dtype=np.float32),
                0.0,
                True,
                False,
                {"cost": np.asarray([0.0, 0.0])},
            )

    class ZeroPolicy:
        @staticmethod
        def __call__(batch, state=None):
            del state
            return Batch(act=np.zeros((len(batch), 1)), state=None)

        @staticmethod
        def map_action(action):
            return action

        @staticmethod
        def map_action_inverse(action):
            return action

        @staticmethod
        def exploration_noise(action, batch):
            del batch
            return action

    rows = {}
    for name, collector_type, collect_kwargs in (
        ("historical_fast_collector", FastCollector, {"n_episode": 1}),
        ("current_collector", Collector, {"n_step": 1}),
    ):
        environment = OneStepEnvironment()
        collector = collector_type(
            ZeroPolicy(), environment, ReplayBuffer(8), exploration_noise=False
        )
        after_initialization = environment.reset_count
        result = collector.collect(**collect_kwargs)
        rows[name] = {
            "resets_after_initialization": after_initialization,
            "resets_after_one_terminal_transition": environment.reset_count,
            "collected_steps": int(result["n/st"]),
            "simulator": "synthetic dummy; zero LongiControl transitions",
        }
    return rows


def collector_buffer_fixture() -> dict[str, Any]:
    """Compare stored transitions when both collectors see one fixed episode."""

    import gymnasium as gym
    from fsrl.data import FastCollector
    from tianshou.data import Batch, Collector, ReplayBuffer

    class ThreeStepEnvironment(gym.Env):
        observation_space = gym.spaces.Box(-100, 100, (1,), dtype=np.float32)
        action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

        def __init__(self):
            self.episode = -1
            self.step_index = 0

        def reset(self, *, seed=None, options=None):
            del seed, options
            self.episode += 1
            self.step_index = 0
            return np.asarray([self.episode * 10], dtype=np.float32), {}

        def step(self, action):
            del action
            self.step_index += 1
            value = self.episode * 10 + self.step_index
            return (
                np.asarray([value], dtype=np.float32),
                float(value),
                self.step_index == 3,
                False,
                {"cost": np.asarray([value / 10, value / 5])},
            )

    class FixedPolicy:
        @staticmethod
        def __call__(batch, state=None):
            del state
            return Batch(act=np.full((len(batch), 1), 0.25), state=None)

        @staticmethod
        def map_action(action):
            return action

        @staticmethod
        def map_action_inverse(action):
            return action

        @staticmethod
        def exploration_noise(action, batch):
            del batch
            return action

    rows = {}
    for name, collector_type in (
        ("historical_fast_collector", FastCollector),
        ("current_collector", Collector),
    ):
        buffer = ReplayBuffer(16)
        collector = collector_type(
            FixedPolicy(), ThreeStepEnvironment(), buffer, exploration_noise=False
        )
        if collector_type is FastCollector:
            collector.collect(n_episode=1)
        else:
            for _ in range(3):
                collector.collect(n_step=1)
        payload = {
            "obs": np.asarray(buffer.obs[:3]).tolist(),
            "act": np.asarray(buffer.act[:3]).tolist(),
            "reward": np.asarray(buffer.rew[:3]).tolist(),
            "done": np.asarray(buffer.done[:3]).tolist(),
            "obs_next": np.asarray(buffer.obs_next[:3]).tolist(),
            "cost": np.asarray(buffer.info.cost[:3]).tolist(),
        }
        rows[name] = {"sha256": _json_sha256(payload), **payload}
    rows["identical"] = (
        rows["historical_fast_collector"]["sha256"]
        == rows["current_collector"]["sha256"]
    )
    return rows


def _fixture_buffer():
    from tianshou.data import Batch, ReplayBuffer

    buffer = ReplayBuffer(16)
    for index in range(8):
        done = index in (3, 7)
        buffer.add(
            Batch(
                obs=np.asarray([index, index + 0.25], dtype=np.float32),
                act=np.asarray([index / 10], dtype=np.float32),
                rew=float(index),
                terminated=done,
                truncated=False,
                done=done,
                obs_next=np.asarray([index + 1, index + 1.25], dtype=np.float32),
                info=Batch(cost=np.asarray([index / 10, index / 5], dtype=np.float64)),
                policy=Batch(),
            )
        )
    return buffer


def replay_fixture() -> dict[str, Any]:
    """Fingerprint sampling, successor selection, and two-step vector returns."""

    import torch

    policy_environment, policy = _build_historical_policy(11)
    buffer = _fixture_buffer()
    sample_state = np.random.get_state()
    try:
        np.random.seed(20261018)
        sampled_indices = buffer.sample_indices(6)
        np.random.set_state(sample_state)
        np.random.seed(20261018)
        repeated_indices = buffer.sample_indices(6)

        indices = np.asarray([0, 2, 3, 6], dtype=np.int64)
        batch = copy.deepcopy(buffer[indices])

        def target_q(_buffer, terminal):
            del _buffer
            size = len(terminal)
            return [
                torch.full((size, 1), value, dtype=torch.float32)
                for value in (10.0, 20.0, 30.0)
            ]

        processed = policy.compute_nstep_returns(
            batch, buffer, indices, target_q, n_step=2
        )
        rets = processed.rets.detach().cpu().numpy()
    finally:
        np.random.set_state(sample_state)
        policy_environment.close()
    metrics = policy.get_metrics(buffer[indices])
    return {
        "sample_seed": 20261018,
        "sampled_indices": sampled_indices.tolist(),
        "repeat_identical": np.array_equal(sampled_indices, repeated_indices),
        "fixture_indices": indices.tolist(),
        "successor_indices": buffer.next(indices).tolist(),
        "done": np.asarray(buffer.done[indices]).tolist(),
        "reward": np.asarray(metrics[0]).tolist(),
        "speed_cost": np.asarray(metrics[1]).tolist(),
        "deadline_cost": np.asarray(metrics[2]).tolist(),
        "two_step_returns": rets.tolist(),
        "two_step_returns_sha256": _json_sha256(rets.tolist()),
    }


def update_schedule_fixture(
    episode_steps: int = 1_800,
    update_per_step: float = 0.1,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Return the exact first-episode PID/learner ordering of both runners."""

    historical_updates = round(update_per_step * episode_steps)
    first_current_update = batch_size
    current_updates_before_pid = math.floor(
        (episode_steps - 1) * update_per_step + 1e-12
    )
    current_updates_after_terminal = (
        math.floor(episode_steps * update_per_step + 1e-12) - current_updates_before_pid
    )
    return {
        "episode_steps": episode_steps,
        "historical": {
            "event_order": ["collect_whole_episode", "pid", "learner_updates"],
            "updates_after_pid": historical_updates,
            "first_policy_change_step": episode_steps,
        },
        "current": {
            "event_order": [
                "collect_each_transition",
                "interleaved_learner_updates",
                "pid_at_terminal",
                "terminal_step_learner_update",
            ],
            "first_policy_change_step": first_current_update,
            "updates_before_pid": current_updates_before_pid,
            "updates_after_pid_on_terminal_step": current_updates_after_terminal,
        },
    }


def summarize_track_overlap(
    left: list[Mapping[str, Any]], right: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Summarize position-wise and set-wise equality of track fingerprints."""

    left_hashes = [str(row["sha256"]) for row in left]
    right_hashes = [str(row["sha256"]) for row in right]
    return {
        "count": min(len(left_hashes), len(right_hashes)),
        "same_at_same_index": sum(
            left_hash == right_hash
            for left_hash, right_hash in zip(left_hashes, right_hashes, strict=False)
        ),
        "shared_unique_fingerprints": len(set(left_hashes) & set(right_hashes)),
        "first_divergence_index": next(
            (
                index
                for index, (left_hash, right_hash) in enumerate(
                    zip(left_hashes, right_hashes, strict=False)
                )
                if left_hash != right_hash
            ),
            None,
        ),
    }
