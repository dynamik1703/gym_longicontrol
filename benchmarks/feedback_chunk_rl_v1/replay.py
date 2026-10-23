"""Fixed-capacity replay for completed residual SMDP transitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .macro_transition import MacroTransition


class MacroReplay:
    def __init__(
        self,
        capacity: int = 300_000,
        observation_dim: int = 16,
        action_dim: int = 3,
    ) -> None:
        if capacity <= 0 or observation_dim <= 0 or action_dim not in (1, 3):
            raise ValueError("replay dimensions are invalid")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.observation = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.speed_cost = np.zeros((capacity, 1), dtype=np.float32)
        self.terminal_cost = np.zeros((capacity, 1), dtype=np.float32)
        self.intervention_cost = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observation = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.continuation_discount = np.zeros((capacity, 1), dtype=np.float32)
        self.native_steps = np.zeros((capacity, 1), dtype=np.int16)
        self.terminated = np.zeros((capacity, 1), dtype=np.bool_)
        self.environment_truncated = np.zeros((capacity, 1), dtype=np.bool_)
        self.dataset_truncated = np.zeros((capacity, 1), dtype=np.bool_)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def append(self, transition: MacroTransition) -> None:
        if transition.action.shape != (self.action_dim,):
            raise ValueError("transition action dimension differs from replay")
        index = self.position
        self.observation[index] = transition.observation
        self.action[index] = transition.action
        self.reward[index, 0] = transition.reward
        self.speed_cost[index, 0] = transition.speed_cost
        self.terminal_cost[index, 0] = transition.terminal_cost
        self.intervention_cost[index, 0] = transition.intervention_cost
        self.next_observation[index] = transition.next_observation
        self.continuation_discount[index, 0] = transition.continuation_discount
        self.native_steps[index, 0] = transition.native_steps
        self.terminated[index, 0] = transition.terminated
        self.environment_truncated[index, 0] = transition.environment_truncated
        self.dataset_truncated[index, 0] = transition.dataset_truncated
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0 or self.size < batch_size:
            raise ValueError("replay does not contain the requested batch")
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            name: getattr(self, name)[indices].copy() for name in self._array_names()
        }

    @staticmethod
    def _array_names() -> tuple[str, ...]:
        return (
            "observation",
            "action",
            "reward",
            "speed_cost",
            "terminal_cost",
            "intervention_cost",
            "next_observation",
            "continuation_discount",
            "native_steps",
            "terminated",
            "environment_truncated",
            "dataset_truncated",
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "position": self.position,
            "size": self.size,
            "arrays": {
                name: getattr(self, name).copy() for name in self._array_names()
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> MacroReplay:
        replay = cls(
            int(state["capacity"]),
            int(state["observation_dim"]),
            int(state["action_dim"]),
        )
        replay.position = int(state["position"])
        replay.size = int(state["size"])
        if not 0 <= replay.position < replay.capacity:
            raise ValueError("invalid replay position")
        if not 0 <= replay.size <= replay.capacity:
            raise ValueError("invalid replay size")
        if set(state["arrays"]) != set(replay._array_names()):
            raise ValueError("replay array set changed")
        for name, value in state["arrays"].items():
            source = np.asarray(value)
            target = getattr(replay, name)
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"replay array contract changed: {name}")
            target[...] = source
        return replay
