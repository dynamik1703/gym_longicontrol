"""Pure supervised GCSL learner with no reward, cost, critic, or Bellman path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import GCSLConfig
from .policy import GCSLPolicy, parameter_count


@dataclass(frozen=True)
class GCSLBatch:
    states: np.ndarray
    actions: np.ndarray
    goals: np.ndarray
    lags: np.ndarray

    def validate(self, config: GCSLConfig) -> None:
        arrays = tuple(np.asarray(value) for value in self.as_arrays())
        if any(value.ndim != 2 for value in arrays[:3]):
            raise ValueError("states, actions, and goals must be rank-two")
        if not len(arrays[0]) == len(arrays[1]) == len(arrays[2]) == len(arrays[3]):
            raise ValueError("batch arrays must align")
        expected = (config.state_dim, config.action_dim, config.goal_dim)
        actual = tuple(value.shape[1] for value in arrays[:3])
        if actual != expected:
            raise ValueError(f"batch dimensions changed: {actual} != {expected}")
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("batch arrays must be finite")
        if np.any(np.abs(arrays[1]) > 1.0):
            raise ValueError("executed actions must lie in [-1, 1]")
        if np.any(arrays[3] <= 0):
            raise ValueError("future lags must be strictly positive")

    def as_arrays(self) -> tuple[np.ndarray, ...]:
        return self.states, self.actions, self.goals, self.lags


class GCSLLearner:
    def __init__(
        self,
        config: GCSLConfig | None = None,
        *,
        seed: int,
        device: str | torch.device = "cpu",
    ):
        self.config = config or GCSLConfig()
        self.device = torch.device(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.policy = GCSLPolicy(
                state_dim=self.config.state_dim,
                goal_dim=self.config.goal_dim,
                action_dim=self.config.action_dim,
                width=self.config.width,
                depth=self.config.depth,
                log_std_min=self.config.log_std_min,
                log_std_max=self.config.log_std_max,
                action_epsilon=self.config.action_epsilon,
            ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.learning_rate
        )

    def tensors(
        self, batch: GCSLBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch.validate(self.config)
        return tuple(
            torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for value in (batch.states, batch.actions, batch.goals)
        )

    def loss(self, batch: GCSLBatch) -> torch.Tensor:
        states, actions, goals = self.tensors(batch)
        return self.policy.nll(states, goals, actions).mean()

    def loss_and_gradients(
        self, batch: GCSLBatch
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
        loss = self.loss(batch)
        return loss, torch.autograd.grad(loss, tuple(self.policy.parameters()))

    def update(self, batch: GCSLBatch) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss(batch)
        loss.backward()
        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.detach().square().sum()
                for parameter in self.policy.parameters()
                if parameter.grad is not None
            )
        )
        self.optimizer.step()
        return {
            "action_nll": float(loss.detach().cpu()),
            "gradient_norm": float(gradient_norm.cpu()),
            "parameter_norm": self.parameter_norm(),
        }

    def deterministic_action(self, states, goals) -> np.ndarray:
        with torch.no_grad():
            action = self.policy.deterministic(
                torch.as_tensor(states, dtype=torch.float32, device=self.device),
                torch.as_tensor(goals, dtype=torch.float32, device=self.device),
            )
        return action.cpu().numpy()

    def sample_action(self, states, goals, *, generator: torch.Generator) -> np.ndarray:
        with torch.no_grad():
            action = self.policy.sample(
                torch.as_tensor(states, dtype=torch.float32, device=self.device),
                torch.as_tensor(goals, dtype=torch.float32, device=self.device),
                generator=generator,
            )
        return action.cpu().numpy()

    def parameter_norm(self) -> float:
        with torch.no_grad():
            total = sum(
                parameter.detach().square().sum()
                for parameter in self.policy.parameters()
            )
        return float(torch.sqrt(total).cpu())

    @property
    def parameter_count(self) -> int:
        return parameter_count(self.policy)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state) != {"policy", "optimizer"}:
            raise ValueError("GCSL learner checkpoint fields changed")
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
