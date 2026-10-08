"""Bounded stochastic continuous policy for the LongiControl GCSL adaptation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


class ResidualBlock(nn.Module):
    """Four Dense-LayerNorm-SiLU operations plus one residual addition."""

    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(4))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer, norm in zip(self.layers, self.norms, strict=True):
            hidden = functional.silu(norm(layer(hidden)))
        return inputs + hidden


class GCSLPolicy(nn.Module):
    """CRL-depth-4-matched actor with a tanh-transformed diagonal Gaussian."""

    def __init__(
        self,
        *,
        state_dim: int = 12,
        goal_dim: int = 3,
        action_dim: int = 1,
        width: int = 256,
        depth: int = 4,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        action_epsilon: float = 1e-6,
    ):
        super().__init__()
        if depth <= 0 or depth % 4:
            raise ValueError("depth must be a positive multiple of four")
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.action_epsilon = float(action_epsilon)
        self.input = nn.Linear(state_dim + goal_dim, width)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            ResidualBlock(width) for _ in range(depth // 4)
        )
        self.mean_head = nn.Linear(width, action_dim)
        self.log_std_head = nn.Linear(width, action_dim)

    def forward(
        self, states: torch.Tensor, goals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.shape[-1] != self.state_dim or goals.shape[-1] != self.goal_dim:
            raise ValueError("state or goal dimension changed")
        hidden = functional.silu(
            self.input_norm(self.input(torch.cat((states, goals), dim=-1)))
        )
        for block in self.blocks:
            hidden = block(hidden)
        mean = self.mean_head(hidden)
        bounded = torch.tanh(self.log_std_head(hidden))
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (bounded + 1.0)
        return mean, log_std

    def nll(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Exact transformed-density NLL, summed over action dimensions.

        Recorded endpoint actions are mapped to the nearest representable open
        interval point solely for the inverse-tanh density calculation.
        """

        mean, log_std = self(states, goals)
        clipped = actions.clamp(
            -1.0 + self.action_epsilon, 1.0 - self.action_epsilon
        )
        pre_tanh = torch.atanh(clipped)
        inv_std = torch.exp(-log_std)
        base_nll = (
            0.5 * ((pre_tanh - mean) * inv_std).square()
            + log_std
            + 0.5 * math.log(2.0 * math.pi)
        )
        log_abs_det_jacobian = 2.0 * (
            math.log(2.0)
            - pre_tanh
            - functional.softplus(-2.0 * pre_tanh)
        )
        return (base_nll + log_abs_det_jacobian).sum(dim=-1)

    def sample(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(states, goals)
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        pre_tanh = mean + torch.exp(log_std) * noise
        return torch.tanh(pre_tanh), pre_tanh

    def deterministic(self, states: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        mean, _ = self(states, goals)
        return torch.tanh(mean)

    def entropy_estimate(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        actions, _ = self.sample(states, goals, generator=generator)
        return self.nll(states, goals, actions)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
