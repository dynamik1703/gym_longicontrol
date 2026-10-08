"""Pure supervised learner: no rewards, critic, targets, or Bellman updates."""

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
    lags: np.ndarray | None = None

    def validate(self, config: GCSLConfig) -> None:
        arrays = (self.states, self.actions, self.goals)
        if any(np.asarray(value).ndim != 2 for value in arrays):
            raise ValueError("states, actions, and goals must be rank-two")
        if not len(self.states) == len(self.actions) == len(self.goals):
            raise ValueError("batch arrays must align")
        expected = (config.state_dim, config.action_dim, config.goal_dim)
        actual = tuple(np.asarray(value).shape[1] for value in arrays)
        if actual != expected:
            raise ValueError(f"batch dimensions changed: {actual} != {expected}")
        if not all(np.isfinite(np.asarray(value)).all() for value in arrays):
            raise ValueError("batch arrays must be finite")
        if np.any(np.abs(self.actions) > 1.0):
            raise ValueError("recorded actions must lie in [-1, 1]")
        if self.lags is not None and (
            len(self.lags) != len(self.states) or np.any(np.asarray(self.lags) <= 0)
        ):
            raise ValueError("future lags must align and be strictly positive")


class GCSLLearner:
    """Adam maximum-likelihood training for one goal-conditioned policy."""

    def __init__(
        self,
        config: GCSLConfig | None = None,
        *,
        seed: int = 0,
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
        gradients = torch.autograd.grad(loss, tuple(self.policy.parameters()))
        return loss, gradients

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
            state_tensor = torch.as_tensor(
                states, dtype=torch.float32, device=self.device
            )
            goal_tensor = torch.as_tensor(
                goals, dtype=torch.float32, device=self.device
            )
            action = self.policy.deterministic(state_tensor, goal_tensor)
        return action.cpu().numpy()

    def sample_action(
        self, states, goals, *, generator: torch.Generator
    ) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                states, dtype=torch.float32, device=self.device
            )
            goal_tensor = torch.as_tensor(
                goals, dtype=torch.float32, device=self.device
            )
            action, _ = self.policy.sample(
                state_tensor, goal_tensor, generator=generator
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
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
