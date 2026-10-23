"""Correct real-time SMDP transitions for one- and three-second actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose
from typing import Any

import numpy as np

GAMMA_DECISION = 0.99
GAMMA_NATIVE = GAMMA_DECISION ** (1.0 / 10.0)


@dataclass(frozen=True)
class MacroTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    speed_cost: float
    terminal_cost: float
    intervention_cost: float
    next_observation: np.ndarray
    continuation_discount: float
    native_steps: int
    maximum_native_steps: int
    terminated: bool
    environment_truncated: bool
    dataset_truncated: bool

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float64)
        next_observation = np.asarray(self.next_observation, dtype=np.float64)
        action = np.asarray(self.action, dtype=np.float64)
        if observation.shape != (16,) or next_observation.shape != (16,):
            raise ValueError("macro observations must be finite 16-vectors")
        if (
            not np.isfinite(observation).all()
            or not np.isfinite(next_observation).all()
        ):
            raise ValueError("macro observations must be finite")
        if action.shape not in ((1,), (3,)) or not np.isfinite(action).all():
            raise ValueError("macro action must be a finite 1- or 3-vector")
        if np.any(action < -1.0) or np.any(action > 1.0):
            raise ValueError("macro action must be in [-1, 1]")
        if self.maximum_native_steps not in (10, 30):
            raise ValueError("macro maximum duration must be 10 or 30")
        if action.size * 10 != self.maximum_native_steps:
            raise ValueError("action dimension and macro duration disagree")
        if not 1 <= self.native_steps <= self.maximum_native_steps:
            raise ValueError("macro transition duration is invalid")
        scalars = (
            self.reward,
            self.speed_cost,
            self.terminal_cost,
            self.intervention_cost,
            self.continuation_discount,
        )
        if not all(np.isfinite(value) for value in scalars):
            raise ValueError("macro values must be finite")
        if min(self.speed_cost, self.terminal_cost, self.intervention_cost) < 0.0:
            raise ValueError("macro costs must be nonnegative")
        ended = bool(
            self.terminated or self.environment_truncated or self.dataset_truncated
        )
        expected = 0.0 if ended else GAMMA_NATIVE**self.native_steps
        if not isclose(self.continuation_discount, expected, abs_tol=1e-12):
            raise ValueError("continuation discount is inconsistent with duration")
        object.__setattr__(self, "observation", observation.copy())
        object.__setattr__(self, "next_observation", next_observation.copy())
        object.__setattr__(self, "action", action.copy())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MacroAccumulator:
    def __init__(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        maximum_native_steps: int,
    ) -> None:
        value = np.asarray(observation, dtype=np.float64)
        actor_action = np.asarray(action, dtype=np.float64)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError("macro observation must be a finite 16-vector")
        if actor_action.shape not in ((1,), (3,)):
            raise ValueError("macro action dimension is invalid")
        if actor_action.size * 10 != maximum_native_steps:
            raise ValueError("macro action and duration disagree")
        if not np.isfinite(actor_action).all() or np.any(np.abs(actor_action) > 1.0):
            raise ValueError("macro action must be finite and in [-1, 1]")
        self.observation = value.copy()
        self.action = actor_action.copy()
        self.maximum_native_steps = int(maximum_native_steps)
        self.reward = 0.0
        self.speed_cost = 0.0
        self.terminal_cost = 0.0
        self.intervention_cost = 0.0
        self.native_steps = 0
        self.terminated = False
        self.environment_truncated = False
        self.dataset_truncated = False

    @property
    def complete(self) -> bool:
        return bool(
            self.native_steps == self.maximum_native_steps
            or self.terminated
            or self.environment_truncated
            or self.dataset_truncated
        )

    def add_native_step(
        self,
        *,
        reward: float,
        speed_cost: float,
        terminal_cost: float,
        intervention_cost: float,
        terminated: bool = False,
        environment_truncated: bool = False,
        dataset_truncated: bool = False,
    ) -> None:
        if self.complete:
            raise RuntimeError("completed macro transition cannot accept another step")
        values = (reward, speed_cost, terminal_cost, intervention_cost)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("native signals must be finite")
        if min(speed_cost, terminal_cost, intervention_cost) < 0.0:
            raise ValueError("native costs must be nonnegative")
        weight = GAMMA_NATIVE**self.native_steps
        self.reward += weight * float(reward)
        self.speed_cost += weight * float(speed_cost)
        self.terminal_cost += weight * float(terminal_cost)
        self.intervention_cost += weight * float(intervention_cost)
        self.native_steps += 1
        self.terminated = bool(terminated)
        self.environment_truncated = bool(environment_truncated)
        self.dataset_truncated = bool(dataset_truncated)

    def finish(self, next_observation: np.ndarray) -> MacroTransition:
        if not self.complete:
            raise RuntimeError("macro transition is not at a decision boundary")
        ended = bool(
            self.terminated or self.environment_truncated or self.dataset_truncated
        )
        discount = 0.0 if ended else GAMMA_NATIVE**self.native_steps
        return MacroTransition(
            observation=self.observation,
            action=self.action,
            reward=self.reward,
            speed_cost=self.speed_cost,
            terminal_cost=self.terminal_cost,
            intervention_cost=self.intervention_cost,
            next_observation=np.asarray(next_observation, dtype=np.float64),
            continuation_discount=discount,
            native_steps=self.native_steps,
            maximum_native_steps=self.maximum_native_steps,
            terminated=self.terminated,
            environment_truncated=self.environment_truncated,
            dataset_truncated=self.dataset_truncated,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.copy(),
            "action": self.action.copy(),
            "maximum_native_steps": self.maximum_native_steps,
            "reward": self.reward,
            "speed_cost": self.speed_cost,
            "terminal_cost": self.terminal_cost,
            "intervention_cost": self.intervention_cost,
            "native_steps": self.native_steps,
            "terminated": self.terminated,
            "environment_truncated": self.environment_truncated,
            "dataset_truncated": self.dataset_truncated,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> MacroAccumulator:
        result = cls(
            state["observation"],
            state["action"],
            int(state["maximum_native_steps"]),
        )
        for field in (
            "reward",
            "speed_cost",
            "terminal_cost",
            "intervention_cost",
        ):
            setattr(result, field, float(state[field]))
        result.native_steps = int(state["native_steps"])
        result.terminated = bool(state["terminated"])
        result.environment_truncated = bool(state["environment_truncated"])
        result.dataset_truncated = bool(state["dataset_truncated"])
        if not 0 <= result.native_steps <= result.maximum_native_steps:
            raise ValueError("restored macro duration is invalid")
        return result


def discounted_sum(values: np.ndarray) -> float:
    native = np.asarray(values, dtype=np.float64)
    if native.ndim != 1 or not np.isfinite(native).all():
        raise ValueError("native values must be one-dimensional and finite")
    return float(np.sum((GAMMA_NATIVE ** np.arange(native.size)) * native))


def reward_bellman_target(
    reward: np.ndarray,
    continuation_discount: np.ndarray,
    next_minimum_q: np.ndarray,
    next_log_probability: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return reward + continuation_discount * (
        next_minimum_q - float(alpha) * next_log_probability
    )


def cost_bellman_target(
    cost: np.ndarray,
    continuation_discount: np.ndarray,
    next_cost_q: np.ndarray,
) -> np.ndarray:
    return cost + continuation_discount * next_cost_q
