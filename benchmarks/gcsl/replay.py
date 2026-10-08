"""Checkpointable episode-safe replay for direct hindsight supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gym_longicontrol.domain.task import TaskSpecification

from .goal_adapter import OUTCOME_DIM, project_outcome
from .learner import GCSLBatch


@dataclass(frozen=True)
class SampledSupervision:
    batch: GCSLBatch
    source_outcomes: np.ndarray
    future_outcomes: np.ndarray
    future_terminated: np.ndarray
    source_indices: np.ndarray
    future_indices: np.ndarray
    episode_ids: np.ndarray
    historical_rewards: np.ndarray


class TrajectoryReplay:
    """Append-only V1 replay; source actions pair only with strict episode futures."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        capacity: int,
        *,
        state_dim: int = 12,
        action_dim: int = 1,
        outcome_dim: int = OUTCOME_DIM,
    ):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.outcome_dim = int(outcome_dim)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, action_dim), dtype=np.float32)
        self.source_outcomes = np.empty((capacity, outcome_dim), dtype=np.float64)
        self.outcomes = np.empty((capacity, outcome_dim), dtype=np.float64)
        self.historical_rewards = np.empty(capacity, dtype=np.float64)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.episode_ids = np.empty(capacity, dtype=np.int64)
        self.episode_steps = np.empty(capacity, dtype=np.int64)
        self._eligible_sources = np.empty(capacity, dtype=np.int64)
        self.eligible_size = 0
        self.size = 0

    def append(
        self,
        *,
        state,
        action,
        source_outcome,
        outcome,
        historical_reward: float,
        terminated: bool,
        truncated: bool,
        episode_id: int,
        episode_step: int,
    ) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("append-only replay capacity is exhausted")
        if self.size:
            previous_episode = int(self.episode_ids[self.size - 1])
            previous_ended = bool(
                self.terminated[self.size - 1] or self.truncated[self.size - 1]
            )
            if episode_id == previous_episode:
                if previous_ended:
                    raise ValueError("post-terminal source row is forbidden")
                if episode_step != int(self.episode_steps[self.size - 1]) + 1:
                    raise ValueError("episode steps must be uninterrupted")
            elif episode_id != previous_episode + 1 or episode_step != 0:
                raise ValueError("episode identity/reset sequence changed")
        if np.asarray(state).shape != (self.state_dim,):
            raise ValueError("state dimension changed")
        if np.asarray(action).shape != (self.action_dim,):
            raise ValueError("action dimension changed")
        if np.asarray(source_outcome).shape != (self.outcome_dim,) or np.asarray(
            outcome
        ).shape != (self.outcome_dim,):
            raise ValueError("outcome dimension changed")
        index = self.size
        self.states[index] = state
        self.actions[index] = action
        self.source_outcomes[index] = source_outcome
        self.outcomes[index] = outcome
        self.historical_rewards[index] = historical_reward
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.episode_ids[index] = episode_id
        self.episode_steps[index] = episode_step
        self._eligible_sources[self.eligible_size] = index
        self.eligible_size += 1
        self.size += 1

    def eligible_sources(self) -> np.ndarray:
        """Transition rows having at least their own strict post-action outcome."""

        return self._eligible_sources[: self.eligible_size].copy()

    def sample(
        self,
        batch_size: int,
        *,
        gamma: float,
        rng: np.random.Generator,
        task: TaskSpecification,
    ) -> SampledSupervision:
        eligible = self._eligible_sources[: self.eligible_size]
        if not len(eligible):
            raise RuntimeError("replay has no strict-future supervised tuple")
        sources = rng.choice(eligible, size=batch_size, replace=True)
        episode_ids = self.episode_ids[: self.size]
        episode_ends = np.searchsorted(
            episode_ids, self.episode_ids[sources] + 1, side="left"
        ) - 1
        maximum_lags = episode_ends - sources + 1
        if np.any(maximum_lags <= 0):
            raise RuntimeError("eligible replay source lost its future")
        uniforms = rng.random(batch_size)
        inverse_argument = 1.0 - uniforms * (1.0 - gamma**maximum_lags)
        lags = np.ceil(np.log(inverse_argument) / np.log(gamma)).astype(np.int64)
        lags = np.clip(lags, 1, maximum_lags)
        futures = sources + lags - 1
        goals = project_outcome(
            self.outcomes[futures],
            terminated=self.terminated[futures],
            task=task,
        ).astype(np.float32)
        batch = GCSLBatch(
            states=self.states[sources].copy(),
            actions=self.actions[sources].copy(),
            goals=goals,
            lags=lags,
        )
        return SampledSupervision(
            batch=batch,
            source_outcomes=self.source_outcomes[sources].copy(),
            future_outcomes=self.outcomes[futures].copy(),
            future_terminated=self.terminated[futures].copy(),
            source_indices=sources,
            future_indices=futures,
            episode_ids=self.episode_ids[sources].copy(),
            historical_rewards=self.historical_rewards[sources].copy(),
        )

    @property
    def memory_bytes(self) -> int:
        arrays = (
            self.states,
            self.actions,
            self.source_outcomes,
            self.outcomes,
            self.historical_rewards,
            self.terminated,
            self.truncated,
            self.episode_ids,
            self.episode_steps,
            self._eligible_sources,
        )
        return sum(array.nbytes for array in arrays)

    def to_state(self) -> dict[str, Any]:
        fields = (
            "states",
            "actions",
            "source_outcomes",
            "outcomes",
            "historical_rewards",
            "terminated",
            "truncated",
            "episode_ids",
            "episode_steps",
        )
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "outcome_dim": self.outcome_dim,
            "size": self.size,
            "eligible_size": self.eligible_size,
            **{
                field: getattr(self, field)[: self.size].copy() for field in fields
            },
            "eligible_sources": self._eligible_sources[
                : self.eligible_size
            ].copy(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TrajectoryReplay:
        if state.get("schema_version") != cls.SCHEMA_VERSION:
            raise RuntimeError("unsupported replay schema")
        replay = cls(
            int(state["capacity"]),
            state_dim=int(state["state_dim"]),
            action_dim=int(state["action_dim"]),
            outcome_dim=int(state["outcome_dim"]),
        )
        replay.size = int(state["size"])
        if not 0 <= replay.size <= replay.capacity:
            raise RuntimeError("invalid replay size")
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
            value = np.asarray(state[field])
            if len(value) != replay.size:
                raise RuntimeError(f"misaligned replay checkpoint field: {field}")
            getattr(replay, field)[: replay.size] = value
        eligible = np.asarray(state["eligible_sources"], dtype=np.int64)
        replay.eligible_size = int(state["eligible_size"])
        if len(eligible) != replay.eligible_size:
            raise RuntimeError("misaligned replay eligible-source index")
        replay._eligible_sources[: replay.eligible_size] = eligible
        return replay
