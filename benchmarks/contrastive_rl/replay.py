"""Fixed-capacity replay and frozen strict-future sampling for CRL execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sampling import future_probabilities


@dataclass(frozen=True)
class SampledFutureBatch:
    """One learning batch together with its exact replay provenance."""

    states: np.ndarray
    actions: np.ndarray
    source_outcomes: np.ndarray
    future_outcomes: np.ndarray
    future_terminated: np.ndarray
    historical_rewards: np.ndarray
    source_indices: np.ndarray
    future_indices: np.ndarray
    lags: np.ndarray
    episode_ids: np.ndarray


class TransitionReplay:
    """Append-only replay sized exactly to one frozen 300k policy budget.

    Sampling uses strict futures from the same uninterrupted episode,
    proportional to ``gamma**lag``. Terminal outcomes can be futures but their
    rows cannot be sources because no post-terminal data are inserted.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        capacity: int,
        *,
        state_dim: int = 12,
        action_dim: int = 1,
        outcome_dim: int = 4,
    ):
        if min(capacity, state_dim, action_dim, outcome_dim) <= 0:
            raise ValueError("replay dimensions and capacity must be positive")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.outcome_dim = int(outcome_dim)
        self.states = np.empty((capacity, state_dim), dtype=np.float64)
        self.actions = np.empty((capacity, action_dim), dtype=np.float64)
        self.source_outcomes = np.empty((capacity, outcome_dim), dtype=np.float64)
        self.outcomes = np.empty((capacity, outcome_dim), dtype=np.float64)
        self.historical_rewards = np.empty(capacity, dtype=np.float64)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.truncated = np.empty(capacity, dtype=np.bool_)
        self.episode_ids = np.empty(capacity, dtype=np.int64)
        self.episode_steps = np.empty(capacity, dtype=np.int64)
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
    ) -> int:
        if self.size >= self.capacity:
            raise RuntimeError("frozen replay capacity is exhausted")
        state = np.asarray(state, dtype=np.float64)
        action = np.asarray(action, dtype=np.float64)
        source_outcome = np.asarray(source_outcome, dtype=np.float64)
        outcome = np.asarray(outcome, dtype=np.float64)
        expected = (
            (state, (self.state_dim,), "state"),
            (action, (self.action_dim,), "action"),
            (source_outcome, (self.outcome_dim,), "source_outcome"),
            (outcome, (self.outcome_dim,), "outcome"),
        )
        for value, shape, name in expected:
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        if not np.isfinite(historical_reward):
            raise ValueError("historical_reward must be finite")
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise ValueError("termination flags must be boolean")
        if episode_id < 0 or episode_step < 0:
            raise ValueError("episode identifiers must be nonnegative")
        if self.size:
            previous = self.size - 1
            same_episode = self.episode_ids[previous] == episode_id
            previous_end = self.terminated[previous] or self.truncated[previous]
            if same_episode and previous_end:
                raise ValueError("post-terminal replay rows require a new episode")
            if same_episode and self.episode_steps[previous] + 1 != episode_step:
                raise ValueError("episode steps must be contiguous")
            if not same_episode and episode_step != 0:
                raise ValueError("a new episode must begin at step zero")
        elif episode_step != 0:
            raise ValueError("the first replay row must begin at episode step zero")

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
        self.size += 1
        return index

    def _episode_last_indices(self) -> np.ndarray:
        if not self.size:
            return np.empty(0, dtype=np.int64)
        ids = self.episode_ids[: self.size]
        last = np.r_[ids[1:] != ids[:-1], True]
        return np.flatnonzero(last)

    def eligible_source_indices(self) -> np.ndarray:
        """Return rows with at least one later row in the same episode."""

        if self.size < 2:
            return np.empty(0, dtype=np.int64)
        mask = np.ones(self.size, dtype=np.bool_)
        mask[self._episode_last_indices()] = False
        return np.flatnonzero(mask)

    def sample(
        self,
        batch_size: int,
        *,
        gamma: float,
        rng: np.random.Generator,
    ) -> SampledFutureBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        eligible = self.eligible_source_indices()
        if not len(eligible):
            raise RuntimeError("replay contains no strict future pair")
        sources = np.asarray(rng.choice(eligible, size=batch_size), dtype=np.int64)
        episode_last = self._episode_last_indices()
        episode_ids = self.episode_ids[: self.size]
        last_by_episode = {
            int(episode_ids[index]): int(index) for index in episode_last
        }
        futures = np.empty(batch_size, dtype=np.int64)
        lags = np.empty(batch_size, dtype=np.int64)
        for row, source in enumerate(sources):
            last = last_by_episode[int(episode_ids[source])]
            candidates = np.arange(source + 1, last + 1, dtype=np.int64)
            candidate_lags = candidates - source
            future = int(
                rng.choice(
                    candidates,
                    p=future_probabilities(candidate_lags, gamma=gamma),
                )
            )
            futures[row] = future
            lags[row] = future - source
        return SampledFutureBatch(
            states=self.states[sources].copy(),
            actions=self.actions[sources].copy(),
            source_outcomes=self.source_outcomes[sources].copy(),
            future_outcomes=self.outcomes[futures].copy(),
            future_terminated=self.terminated[futures].copy(),
            historical_rewards=self.historical_rewards[sources].copy(),
            source_indices=sources,
            future_indices=futures,
            lags=lags,
            episode_ids=episode_ids[sources].copy(),
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "outcome_dim": self.outcome_dim,
            "size": self.size,
            "states": self.states[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "source_outcomes": self.source_outcomes[: self.size].copy(),
            "outcomes": self.outcomes[: self.size].copy(),
            "historical_rewards": self.historical_rewards[: self.size].copy(),
            "terminated": self.terminated[: self.size].copy(),
            "truncated": self.truncated[: self.size].copy(),
            "episode_ids": self.episode_ids[: self.size].copy(),
            "episode_steps": self.episode_steps[: self.size].copy(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TransitionReplay:
        if state.get("schema_version") != cls.SCHEMA_VERSION:
            raise RuntimeError("unsupported replay checkpoint schema")
        replay = cls(
            int(state["capacity"]),
            state_dim=int(state["state_dim"]),
            action_dim=int(state["action_dim"]),
            outcome_dim=int(state["outcome_dim"]),
        )
        size = int(state["size"])
        if not 0 <= size <= replay.capacity:
            raise RuntimeError("invalid replay size in checkpoint")
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
        for field in fields:
            value = np.asarray(state[field])
            if len(value) != size:
                raise RuntimeError(f"checkpoint replay field is misaligned: {field}")
            getattr(replay, field)[:size] = value
        replay.size = size
        return replay
