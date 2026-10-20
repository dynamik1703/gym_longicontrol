"""Checkpointable ring replay with exact same-episode future supervision."""

from __future__ import annotations

from collections import Counter
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
    """Transition-capacity ring with implicit, rebuildable eligibility indices.

    A stored source action at physical step ``t`` may pair with the post-action
    outcome in its own row (lag one) or any later retained outcome before that
    episode ends.  Absolute sequence IDs and per-episode retained counts make
    overwrite safe: no cached link can outlive either endpoint.
    """

    SCHEMA_VERSION = 2

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
        self.sequences = np.full(capacity, -1, dtype=np.int64)
        self.size = 0
        self.total_appended = 0
        self._episode_counts: Counter[int] = Counter()
        self._closed_episode_ends: dict[int, int] = {}
        self._last_episode_id: int | None = None
        self._last_episode_step: int | None = None
        self._last_row_ended = False

    @property
    def first_sequence(self) -> int:
        return self.total_appended - self.size

    @property
    def eligible_size(self) -> int:
        return self.size

    def _position(self, sequence: int) -> int:
        if not self.first_sequence <= sequence < self.total_appended:
            raise IndexError("replay sequence is no longer retained")
        position = sequence % self.capacity
        if int(self.sequences[position]) != sequence:
            raise RuntimeError("replay ring sequence/index integrity failed")
        return position

    def _validate_continuity(self, episode_id: int, episode_step: int) -> None:
        if self._last_episode_id is None:
            if episode_step != 0:
                raise ValueError("first replay row must begin at episode step zero")
            return
        if episode_id == self._last_episode_id:
            if self._last_row_ended:
                raise ValueError("post-terminal source row is forbidden")
            if episode_step != self._last_episode_step + 1:
                raise ValueError("episode steps must be uninterrupted")
            return
        if episode_id != self._last_episode_id + 1 or episode_step != 0:
            raise ValueError("episode identity/reset sequence changed")
        if not self._last_row_ended:
            raise ValueError("new episode started before the prior episode ended")

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
        self._validate_continuity(int(episode_id), int(episode_step))
        values = {
            "state": (np.asarray(state), (self.state_dim,)),
            "action": (np.asarray(action), (self.action_dim,)),
            "source_outcome": (np.asarray(source_outcome), (self.outcome_dim,)),
            "outcome": (np.asarray(outcome), (self.outcome_dim,)),
        }
        for name, (value, shape) in values.items():
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
        if not np.isfinite(historical_reward):
            raise ValueError("historical reward must be finite provenance")
        if np.any(np.abs(values["action"][0]) > 1.0):
            raise ValueError("executed action lies outside [-1, 1]")

        sequence = self.total_appended
        position = sequence % self.capacity
        if self.size == self.capacity:
            overwritten_episode = int(self.episode_ids[position])
            self._episode_counts[overwritten_episode] -= 1
            if self._episode_counts[overwritten_episode] == 0:
                del self._episode_counts[overwritten_episode]
                self._closed_episode_ends.pop(overwritten_episode, None)
        else:
            self.size += 1

        self.states[position] = values["state"][0]
        self.actions[position] = values["action"][0]
        self.source_outcomes[position] = values["source_outcome"][0]
        self.outcomes[position] = values["outcome"][0]
        self.historical_rewards[position] = historical_reward
        self.terminated[position] = bool(terminated)
        self.truncated[position] = bool(truncated)
        self.episode_ids[position] = int(episode_id)
        self.episode_steps[position] = int(episode_step)
        self.sequences[position] = sequence
        self._episode_counts[int(episode_id)] += 1
        self.total_appended += 1
        ended = bool(terminated or truncated)
        if ended:
            self._closed_episode_ends[int(episode_id)] = sequence
        self._last_episode_id = int(episode_id)
        self._last_episode_step = int(episode_step)
        self._last_row_ended = ended
        return sequence

    @staticmethod
    def _sample_lags(
        maximum_lags: np.ndarray, *, gamma: float, rng: np.random.Generator
    ) -> np.ndarray:
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")
        uniforms = rng.random(len(maximum_lags))
        inverse_argument = 1.0 - uniforms * (1.0 - gamma**maximum_lags)
        lags = np.ceil(np.log(inverse_argument) / np.log(gamma)).astype(np.int64)
        return np.clip(lags, 1, maximum_lags)

    def sample(
        self,
        batch_size: int,
        *,
        gamma: float,
        rng: np.random.Generator,
        task: TaskSpecification,
    ) -> SampledSupervision:
        if self.size == 0:
            raise RuntimeError("replay has no strict-future supervised tuple")
        sources = rng.integers(
            self.first_sequence, self.total_appended, size=batch_size, dtype=np.int64
        )
        source_positions = np.fromiter(
            (self._position(int(value)) for value in sources),
            dtype=np.int64,
            count=batch_size,
        )
        source_episodes = self.episode_ids[source_positions]
        episode_ends = np.asarray(
            [
                self._closed_episode_ends.get(int(episode), self.total_appended - 1)
                for episode in source_episodes
            ],
            dtype=np.int64,
        )
        maximum_lags = episode_ends - sources + 1
        if np.any(maximum_lags <= 0):
            raise RuntimeError("eligible replay source lost its physical future")
        lags = self._sample_lags(maximum_lags, gamma=gamma, rng=rng)
        futures = sources + lags - 1
        future_positions = np.fromiter(
            (self._position(int(value)) for value in futures),
            dtype=np.int64,
            count=batch_size,
        )
        if not np.array_equal(
            self.episode_ids[source_positions], self.episode_ids[future_positions]
        ):
            raise RuntimeError("replay future crossed an episode reset")
        goals = project_outcome(
            self.outcomes[future_positions],
            terminated=self.terminated[future_positions],
            task=task,
        ).astype(np.float32)
        return SampledSupervision(
            batch=GCSLBatch(
                states=self.states[source_positions].copy(),
                actions=self.actions[source_positions].copy(),
                goals=goals,
                lags=lags,
            ),
            source_outcomes=self.source_outcomes[source_positions].copy(),
            future_outcomes=self.outcomes[future_positions].copy(),
            future_terminated=self.terminated[future_positions].copy(),
            source_indices=sources,
            future_indices=futures,
            episode_ids=source_episodes.copy(),
            historical_rewards=self.historical_rewards[source_positions].copy(),
        )

    def retained_sequences(self) -> np.ndarray:
        return np.arange(self.first_sequence, self.total_appended, dtype=np.int64)

    def support_summary(self, task: TaskSpecification) -> dict[str, Any]:
        sequences = self.retained_sequences()
        positions = sequences % self.capacity
        if not len(positions):
            return {
                "retained_transitions": 0,
                "maximum_replay_progress": None,
                "minimum_goal_distance_to_canonical": None,
                "canonical_hindsight_outcome_rows": 0,
                "unique_projected_goals": 0,
                "timely_rows": 0,
                "safe_rows": 0,
            }
        goals = project_outcome(
            self.outcomes[positions],
            terminated=self.terminated[positions],
            task=task,
        )
        distances = np.linalg.norm(goals - np.ones(3, dtype=np.float64), axis=1)
        canonical = np.all(goals == 1.0, axis=1)
        return {
            "retained_transitions": int(len(goals)),
            "maximum_replay_progress": float(np.max(goals[:, 0])),
            "minimum_goal_distance_to_canonical": float(np.min(distances)),
            "canonical_hindsight_outcome_rows": int(np.sum(canonical)),
            "canonical_hindsight_outcome_fraction": float(np.mean(canonical)),
            "unique_projected_goals": int(len(np.unique(goals, axis=0))),
            "timely_rows": int(np.sum(goals[:, 1] == 1.0)),
            "safe_rows": int(np.sum(goals[:, 2] == 1.0)),
        }

    def integrity_report(self) -> dict[str, Any]:
        sequences = self.retained_sequences()
        positions = np.asarray([value % self.capacity for value in sequences])
        exact = np.array_equal(self.sequences[positions], sequences)
        ids = self.episode_ids[positions] if len(positions) else np.asarray([])
        steps = self.episode_steps[positions] if len(positions) else np.asarray([])
        boundaries_valid = True
        for index in range(1, len(ids)):
            same = ids[index] == ids[index - 1]
            if same and steps[index] != steps[index - 1] + 1:
                boundaries_valid = False
        return {
            "size": self.size,
            "capacity": self.capacity,
            "first_sequence": self.first_sequence,
            "total_appended": self.total_appended,
            "sequence_index_exact": bool(exact),
            "episode_boundaries_valid": boundaries_valid,
            "eligible_sources": self.eligible_size,
            "retained_episode_count": len(self._episode_counts),
        }

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
            self.sequences,
        )
        return sum(array.nbytes for array in arrays)

    def to_state(self) -> dict[str, Any]:
        sequences = self.retained_sequences()
        positions = np.asarray(
            [sequence % self.capacity for sequence in sequences], dtype=np.int64
        )
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
            "total_appended": self.total_appended,
            "sequences": sequences,
            **{field: getattr(self, field)[positions].copy() for field in fields},
            "closed_episode_ends": dict(self._closed_episode_ends),
            "last_episode_id": self._last_episode_id,
            "last_episode_step": self._last_episode_step,
            "last_row_ended": self._last_row_ended,
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
        replay.total_appended = int(state["total_appended"])
        sequences = np.asarray(state["sequences"], dtype=np.int64)
        if len(sequences) != replay.size or not np.array_equal(
            sequences, np.arange(replay.first_sequence, replay.total_appended)
        ):
            raise RuntimeError("checkpoint replay sequence range changed")
        positions = sequences % replay.capacity
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
            getattr(replay, field)[positions] = value
        replay.sequences[positions] = sequences
        replay._episode_counts = Counter(
            map(int, replay.episode_ids[positions].tolist())
        )
        replay._closed_episode_ends = {
            int(key): int(value)
            for key, value in state["closed_episode_ends"].items()
            if int(key) in replay._episode_counts
        }
        replay._last_episode_id = state["last_episode_id"]
        replay._last_episode_step = state["last_episode_step"]
        replay._last_row_ended = bool(state["last_row_ended"])
        report = replay.integrity_report()
        if not report["sequence_index_exact"] or not report["episode_boundaries_valid"]:
            raise RuntimeError("restored replay integrity failed")
        return replay
