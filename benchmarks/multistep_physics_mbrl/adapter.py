"""Asymmetric training-state adapter for real replay and exact physics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from benchmarks.model_based_rl.model_state import ModelState, TrackContext


def state_from_environment(environment: gym.Env, episode_step: int) -> ModelState:
    """Capture benchmark-internal state without changing the public observation."""

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


@dataclass(frozen=True)
class RecordedTransition:
    transition_id: int
    state: ModelState
    action: float
    next_state: ModelState
    context: TrackContext
    observation: np.ndarray
    next_observation: np.ndarray


class RealPhysicsReplay:
    """Bounded sidecar keyed by immutable real-transition IDs.

    The SACLag replay remains the sole sampling distribution. This sidecar contains
    only the internal physical state needed to continue a sampled real prefix.
    """

    def __init__(self, capacity: int = 100_000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._order: deque[int] = deque()
        self._items: dict[int, RecordedTransition] = {}

    def append(self, transition: RecordedTransition) -> None:
        key = int(transition.transition_id)
        if key in self._items:
            raise ValueError("A real transition ID may be recorded only once")
        self._items[key] = transition
        self._order.append(key)
        while len(self._order) > self.capacity:
            del self._items[self._order.popleft()]

    def get(self, transition_id: int) -> RecordedTransition:
        try:
            return self._items[int(transition_id)]
        except KeyError as error:
            raise KeyError(
                "Sampled real transition is absent from physics sidecar"
            ) from error

    def __len__(self) -> int:
        return len(self._order)


class TrainingSeedWrapper(gym.Wrapper):
    """Delegate every reset seed to the checkpointable training stream."""

    def __init__(self, environment: gym.Env, stream: Any):
        super().__init__(environment)
        self.stream = stream
        self.seeds_used: list[int] = []

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise ValueError("Training seeds are owned by the frozen stream")
        seed = self.stream.next_seed()
        self.seeds_used.append(seed)
        return self.env.reset(seed=seed, options=options)


class RealTransitionRecorder(gym.Wrapper):
    """Record exact internal states and attach a stable ID to real replay info."""

    def __init__(self, environment: gym.Env):
        super().__init__(environment)
        self.episode_step = 0
        self.transition_id = 0
        self.records: list[RecordedTransition] = []
        self.completed_episodes = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.episode_step = 0
        return result

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
        observation, reward, terminated, truncated, raw_info = self.env.step(action)
        self.episode_step += 1
        after = state_from_environment(self.env, self.episode_step)
        info = dict(raw_info)
        info["real_transition_id"] = self.transition_id
        self.records.append(
            RecordedTransition(
                transition_id=self.transition_id,
                state=before,
                action=float(np.asarray(action).reshape(-1)[0]),
                next_state=after,
                context=context,
                observation=before_observation,
                next_observation=np.asarray(observation, dtype=np.float64),
            )
        )
        self.transition_id += 1
        if terminated or truncated:
            self.completed_episodes += 1
        return observation, reward, terminated, truncated, info

    def drain(self) -> list[RecordedTransition]:
        records, self.records = self.records, []
        return records


class RealEpisodePIDGate:
    """Permit PID multiplier updates only from completed physical episodes."""

    def __init__(self):
        self.real_episode_updates = 0

    def update(self, policy: Any, *, source: str, stats: dict[str, Any]) -> None:
        if source != "real_completed_episode":
            raise ValueError("Model rollouts must never update PID multipliers")
        costs = np.asarray(stats.get("cost"), dtype=np.float64)
        if costs.shape != (2,) or not np.isfinite(costs).all():
            raise ValueError("PID updates require two finite real episodic costs")
        policy.pre_update_fn(stats_train={**stats, "cost": costs})
        self.real_episode_updates += 1


__all__ = [
    "ModelState",
    "RealEpisodePIDGate",
    "RealPhysicsReplay",
    "RealTransitionRecorder",
    "RecordedTransition",
    "TrackContext",
    "TrainingSeedWrapper",
    "state_from_environment",
]
