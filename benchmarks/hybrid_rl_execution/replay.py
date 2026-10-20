"""Fixed-capacity replay for decision-boundary Hybrid SMDP transitions."""

from __future__ import annotations

from typing import Any

import numpy as np

from .macro_transition import MacroTransition


class MacroReplay:
    """Store proposed strategy actions, never executed native actuators."""

    def __init__(self, capacity: int = 300_000, observation_dim: int = 16) -> None:
        if capacity <= 0 or observation_dim <= 0:
            raise ValueError("replay dimensions must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.observation = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.action = np.zeros((capacity, 1), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.speed_cost = np.zeros((capacity, 1), dtype=np.float32)
        self.terminal_cost = np.zeros((capacity, 1), dtype=np.float32)
        self.intervention_cost = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observation = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.continuation_discount = np.zeros((capacity, 1), dtype=np.float32)
        self.native_steps = np.zeros((capacity, 1), dtype=np.int16)
        self.terminated = np.zeros((capacity, 1), dtype=np.bool_)
        self.truncated = np.zeros((capacity, 1), dtype=np.bool_)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def append(self, transition: MacroTransition) -> None:
        index = self.position
        self.observation[index] = transition.observation
        self.action[index, 0] = transition.action
        self.reward[index, 0] = transition.reward
        self.speed_cost[index, 0] = transition.speed_cost
        self.terminal_cost[index, 0] = transition.terminal_cost
        self.intervention_cost[index, 0] = transition.intervention_cost
        self.next_observation[index] = transition.next_observation
        self.continuation_discount[index, 0] = transition.continuation_discount
        self.native_steps[index, 0] = transition.native_steps
        self.terminated[index, 0] = transition.terminated
        self.truncated[index, 0] = transition.truncated
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0 or self.size < batch_size:
            raise ValueError("replay does not contain the requested batch")
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "observation": self.observation[indices].copy(),
            "action": self.action[indices].copy(),
            "reward": self.reward[indices].copy(),
            "speed_cost": self.speed_cost[indices].copy(),
            "terminal_cost": self.terminal_cost[indices].copy(),
            "intervention_cost": self.intervention_cost[indices].copy(),
            "next_observation": self.next_observation[indices].copy(),
            "continuation_discount": self.continuation_discount[indices].copy(),
            "native_steps": self.native_steps[indices].copy(),
            "terminated": self.terminated[indices].copy(),
            "truncated": self.truncated[indices].copy(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "position": self.position,
            "size": self.size,
            "arrays": {
                name: getattr(self, name).copy()
                for name in (
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
                    "truncated",
                )
            },
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> MacroReplay:
        replay = cls(int(state["capacity"]), int(state["observation_dim"]))
        replay.position = int(state["position"])
        replay.size = int(state["size"])
        if not 0 <= replay.position < replay.capacity:
            raise ValueError("invalid replay position")
        if not 0 <= replay.size <= replay.capacity:
            raise ValueError("invalid replay size")
        for name, target in state["arrays"].items():
            if not hasattr(replay, name):
                raise ValueError(f"unexpected replay array: {name}")
            value = np.asarray(target)
            existing = getattr(replay, name)
            if value.shape != existing.shape or value.dtype != existing.dtype:
                raise ValueError(f"replay array contract changed: {name}")
            existing[...] = value
        return replay
