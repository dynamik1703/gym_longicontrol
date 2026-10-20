"""One-hertz semi-Markov transitions over the ten-hertz physical system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose
from typing import Any

import numpy as np

DECISION_INTERVAL_NATIVE_STEPS = 10
GAMMA_DECISION = 0.99
GAMMA_NATIVE = GAMMA_DECISION ** (1.0 / DECISION_INTERVAL_NATIVE_STEPS)


@dataclass(frozen=True)
class MacroTransition:
    """A strategy transition whose action is held for ``native_steps``."""

    observation: np.ndarray
    action: float
    reward: float
    speed_cost: float
    terminal_cost: float
    intervention_cost: float
    next_observation: np.ndarray
    continuation_discount: float
    native_steps: int
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float64)
        next_observation = np.asarray(self.next_observation, dtype=np.float64)
        if observation.shape != (16,) or next_observation.shape != (16,):
            raise ValueError("macro observations must be finite 16-vectors")
        if (
            not np.isfinite(observation).all()
            or not np.isfinite(next_observation).all()
        ):
            raise ValueError("macro observations must be finite 16-vectors")
        if not 1 <= self.native_steps <= DECISION_INTERVAL_NATIVE_STEPS:
            raise ValueError("macro transition must contain one to ten native steps")
        if not -1.0 <= self.action <= 1.0:
            raise ValueError("stored strategy action must be in [-1, 1]")
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
        ended = bool(self.terminated or self.truncated)
        expected = 0.0 if ended else GAMMA_NATIVE**self.native_steps
        if not isclose(self.continuation_discount, expected, abs_tol=1e-12):
            raise ValueError("continuation discount is inconsistent with duration")
        object.__setattr__(self, "observation", observation.copy())
        object.__setattr__(self, "next_observation", next_observation.copy())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MacroAccumulator:
    """Discount native signals until the next real strategy boundary."""

    def __init__(self, observation: np.ndarray, action: float) -> None:
        value = np.asarray(observation, dtype=np.float64)
        if value.shape != (16,) or not np.isfinite(value).all():
            raise ValueError("macro observation must be a finite 16-vector")
        if not np.isfinite(action) or not -1.0 <= action <= 1.0:
            raise ValueError("macro action must be finite and in [-1, 1]")
        self.observation = value.copy()
        self.action = float(action)
        self.reward = 0.0
        self.speed_cost = 0.0
        self.terminal_cost = 0.0
        self.intervention_cost = 0.0
        self.native_steps = 0
        self.terminated = False
        self.truncated = False

    @property
    def complete(self) -> bool:
        return bool(
            self.native_steps == DECISION_INTERVAL_NATIVE_STEPS
            or self.terminated
            or self.truncated
        )

    def add_native_step(
        self,
        *,
        reward: float,
        speed_cost: float,
        terminal_cost: float,
        intervention_cost: float,
        terminated: bool = False,
        truncated: bool = False,
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
        self.truncated = bool(truncated)

    def finish(self, next_observation: np.ndarray) -> MacroTransition:
        if not self.complete:
            raise RuntimeError("macro transition is not at a decision boundary")
        ended = self.terminated or self.truncated
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
            terminated=self.terminated,
            truncated=self.truncated,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.copy(),
            "action": self.action,
            "reward": self.reward,
            "speed_cost": self.speed_cost,
            "terminal_cost": self.terminal_cost,
            "intervention_cost": self.intervention_cost,
            "native_steps": self.native_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> MacroAccumulator:
        result = cls(state["observation"], float(state["action"]))
        result.reward = float(state["reward"])
        result.speed_cost = float(state["speed_cost"])
        result.terminal_cost = float(state["terminal_cost"])
        result.intervention_cost = float(state["intervention_cost"])
        result.native_steps = int(state["native_steps"])
        result.terminated = bool(state["terminated"])
        result.truncated = bool(state["truncated"])
        if not 0 <= result.native_steps <= DECISION_INTERVAL_NATIVE_STEPS:
            raise ValueError("invalid restored macro duration")
        return result


def reward_bellman_target(
    reward: np.ndarray,
    continuation_discount: np.ndarray,
    next_minimum_q: np.ndarray,
    next_log_probability: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Reference SAC reward target used by analytical tests."""

    return reward + continuation_discount * (
        next_minimum_q - float(alpha) * next_log_probability
    )


def cost_bellman_target(
    cost: np.ndarray,
    continuation_discount: np.ndarray,
    next_cost_q: np.ndarray,
) -> np.ndarray:
    """Reference constrained-critic target (costs have no entropy bonus)."""

    return cost + continuation_discount * next_cost_q
