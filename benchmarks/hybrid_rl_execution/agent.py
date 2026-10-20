"""Constrained Soft Actor-Critic for the one-hertz Hybrid strategy."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

OBSERVATION_DIM = 16
ACTION_DIM = 1
HIDDEN_SIZES = (256, 256)
TARGET_ENTROPY = -1.0
TAU = 0.005
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-4
ENTROPY_LEARNING_RATE = 3e-4
MULTIPLIER_LEARNING_RATE = 1e-3
INTERVENTION_REGULARIZER_WEIGHT = 0.05
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
TANH_EPSILON = 1e-6


def _mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_SIZES[0]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZES[1], output_dim),
    )


class GaussianActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBSERVATION_DIM, HIDDEN_SIZES[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1]),
            nn.ReLU(),
        )
        self.mean = nn.Linear(HIDDEN_SIZES[1], ACTION_DIM)
        self.log_std = nn.Linear(HIDDEN_SIZES[1], ACTION_DIM)

    def distribution_parameters(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observation)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        observation: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        standard_deviation = log_std.exp()
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        raw_action = mean + standard_deviation * noise
        action = torch.tanh(raw_action)
        gaussian_log_probability = -0.5 * (
            noise.square() + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_probability = gaussian_log_probability.sum(dim=-1, keepdim=True)
        log_probability -= torch.log(1.0 - action.square() + TANH_EPSILON).sum(
            dim=-1, keepdim=True
        )
        return action, log_probability

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(observation)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = _mlp(OBSERVATION_DIM + ACTION_DIM, 1)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=-1))


@dataclass(frozen=True)
class AgentConfig:
    observation_dim: int = OBSERVATION_DIM
    action_dim: int = ACTION_DIM
    hidden_sizes: tuple[int, int] = HIDDEN_SIZES
    tau: float = TAU
    target_entropy: float = TARGET_ENTROPY
    actor_learning_rate: float = ACTOR_LEARNING_RATE
    critic_learning_rate: float = CRITIC_LEARNING_RATE
    entropy_learning_rate: float = ENTROPY_LEARNING_RATE
    multiplier_learning_rate: float = MULTIPLIER_LEARNING_RATE
    intervention_regularizer_weight: float = INTERVENTION_REGULARIZER_WEIGHT


class ConstrainedSAC:
    """Twin reward critics plus speed, terminal and intervention critics.

    Speed and terminal critics have nonnegative learned Lagrange multipliers.
    Intervention deviation is a separate fixed-weight soft regularizer, not a
    feasibility constraint and not part of the physical energy reward.
    """

    def __init__(self, seed: int, device: str | torch.device = "cpu") -> None:
        self.seed = int(seed)
        self.device = torch.device(device)
        self.config = AgentConfig()
        if self.device.type != "cpu":
            raise ValueError("canonical Hybrid V1 is frozen to deterministic CPU")
        # Small fixed MLP updates are faster and reproducible without a large
        # host-dependent intra-op thread pool.
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.actor = GaussianActor().to(self.device)
            self.reward_critic_1 = Critic().to(self.device)
            self.reward_critic_2 = Critic().to(self.device)
            self.speed_critic = Critic().to(self.device)
            self.terminal_critic = Critic().to(self.device)
            self.intervention_critic = Critic().to(self.device)
        self.target_reward_critic_1 = copy.deepcopy(self.reward_critic_1)
        self.target_reward_critic_2 = copy.deepcopy(self.reward_critic_2)
        self.target_speed_critic = copy.deepcopy(self.speed_critic)
        self.target_terminal_critic = copy.deepcopy(self.terminal_critic)
        self.target_intervention_critic = copy.deepcopy(self.intervention_critic)
        self.actor_optimizer = Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.reward_optimizer = Adam(
            [
                *self.reward_critic_1.parameters(),
                *self.reward_critic_2.parameters(),
            ],
            lr=self.config.critic_learning_rate,
        )
        self.cost_optimizer = Adam(
            [
                *self.speed_critic.parameters(),
                *self.terminal_critic.parameters(),
                *self.intervention_critic.parameters(),
            ],
            lr=self.config.critic_learning_rate,
        )
        self.log_alpha = nn.Parameter(torch.zeros(1, device=self.device))
        self.alpha_optimizer = Adam(
            [self.log_alpha], lr=self.config.entropy_learning_rate
        )
        self.speed_multiplier = nn.Parameter(torch.zeros(1, device=self.device))
        self.terminal_multiplier = nn.Parameter(torch.zeros(1, device=self.device))
        self.multiplier_optimizer = Adam(
            [self.speed_multiplier, self.terminal_multiplier],
            lr=self.config.multiplier_learning_rate,
        )
        self.update_count = 0

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp().cpu().item())

    @property
    def parameter_count(self) -> int:
        modules = (
            self.actor,
            self.reward_critic_1,
            self.reward_critic_2,
            self.speed_critic,
            self.terminal_critic,
            self.intervention_critic,
        )
        return sum(
            parameter.numel() for module in modules for parameter in module.parameters()
        )

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> float:
        value = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        ).reshape(1, OBSERVATION_DIM)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(value)
            else:
                if generator is None:
                    raise ValueError("stochastic action requires an explicit RNG")
                action, _ = self.actor.sample(value, generator=generator)
        return float(action.cpu().item())

    @staticmethod
    def _set_gradient(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    @staticmethod
    def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - tau)
                target_parameter.add_(tau * source_parameter)

    def _tensor_batch(self, batch: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
        fields = (
            "observation",
            "action",
            "reward",
            "speed_cost",
            "terminal_cost",
            "intervention_cost",
            "next_observation",
            "continuation_discount",
        )
        return {
            name: torch.as_tensor(
                np.asarray(batch[name], dtype=np.float32), device=self.device
            )
            for name in fields
        }

    def update(
        self,
        batch: Mapping[str, np.ndarray],
        *,
        generator: torch.Generator,
    ) -> dict[str, float]:
        tensors = self._tensor_batch(batch)
        observation = tensors["observation"]
        action = tensors["action"]
        discount = tensors["continuation_discount"]
        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(
                tensors["next_observation"], generator=generator
            )
            next_reward_q = torch.minimum(
                self.target_reward_critic_1(tensors["next_observation"], next_action),
                self.target_reward_critic_2(tensors["next_observation"], next_action),
            )
            alpha = self.log_alpha.exp()
            reward_target = tensors["reward"] + discount * (
                next_reward_q - alpha * next_log_probability
            )
            speed_target = tensors["speed_cost"] + discount * self.target_speed_critic(
                tensors["next_observation"], next_action
            )
            terminal_target = tensors["terminal_cost"] + discount * (
                self.target_terminal_critic(tensors["next_observation"], next_action)
            )
            intervention_target = tensors["intervention_cost"] + discount * (
                self.target_intervention_critic(
                    tensors["next_observation"], next_action
                )
            )

        reward_prediction_1 = self.reward_critic_1(observation, action)
        reward_prediction_2 = self.reward_critic_2(observation, action)
        reward_loss = nn.functional.mse_loss(
            reward_prediction_1, reward_target
        ) + nn.functional.mse_loss(reward_prediction_2, reward_target)
        self.reward_optimizer.zero_grad(set_to_none=True)
        reward_loss.backward()
        self.reward_optimizer.step()

        speed_prediction = self.speed_critic(observation, action)
        terminal_prediction = self.terminal_critic(observation, action)
        intervention_prediction = self.intervention_critic(observation, action)
        cost_loss = (
            nn.functional.mse_loss(speed_prediction, speed_target)
            + nn.functional.mse_loss(terminal_prediction, terminal_target)
            + nn.functional.mse_loss(intervention_prediction, intervention_target)
        )
        self.cost_optimizer.zero_grad(set_to_none=True)
        cost_loss.backward()
        self.cost_optimizer.step()

        critics = (
            self.reward_critic_1,
            self.reward_critic_2,
            self.speed_critic,
            self.terminal_critic,
            self.intervention_critic,
        )
        for critic in critics:
            self._set_gradient(critic, False)
        policy_action, log_probability = self.actor.sample(
            observation, generator=generator
        )
        reward_q = torch.minimum(
            self.reward_critic_1(observation, policy_action),
            self.reward_critic_2(observation, policy_action),
        )
        speed_q = self.speed_critic(observation, policy_action)
        terminal_q = self.terminal_critic(observation, policy_action)
        intervention_q = self.intervention_critic(observation, policy_action)
        actor_loss = (
            self.log_alpha.detach().exp() * log_probability
            - reward_q
            + self.speed_multiplier.detach() * speed_q
            + self.terminal_multiplier.detach() * terminal_q
            + self.config.intervention_regularizer_weight * intervention_q
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for critic in critics:
            self._set_gradient(critic, True)

        alpha_loss = -(
            self.log_alpha * (log_probability + self.config.target_entropy).detach()
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        multiplier_loss = -(
            self.speed_multiplier * speed_q.detach().mean()
            + self.terminal_multiplier * terminal_q.detach().mean()
        )
        self.multiplier_optimizer.zero_grad(set_to_none=True)
        multiplier_loss.backward()
        self.multiplier_optimizer.step()
        with torch.no_grad():
            self.speed_multiplier.clamp_(min=0.0)
            self.terminal_multiplier.clamp_(min=0.0)

        pairs = (
            (self.reward_critic_1, self.target_reward_critic_1),
            (self.reward_critic_2, self.target_reward_critic_2),
            (self.speed_critic, self.target_speed_critic),
            (self.terminal_critic, self.target_terminal_critic),
            (self.intervention_critic, self.target_intervention_critic),
        )
        for source, target in pairs:
            self._soft_update(source, target, self.config.tau)
        self.update_count += 1
        values = {
            "reward_critic_loss": float(reward_loss.detach().cpu()),
            "cost_critic_loss": float(cost_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": self.alpha,
            "speed_multiplier": float(self.speed_multiplier.detach().cpu()),
            "terminal_multiplier": float(self.terminal_multiplier.detach().cpu()),
            "mean_reward_target": float(reward_target.mean().cpu()),
            "mean_speed_target": float(speed_target.mean().cpu()),
            "mean_terminal_target": float(terminal_target.mean().cpu()),
            "mean_intervention_target": float(intervention_target.mean().cpu()),
        }
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("nonfinite SAC update diagnostic")
        return values

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "config": self.config.__dict__.copy(),
            "update_count": self.update_count,
            "modules": {
                name: module.state_dict() for name, module in self._modules().items()
            },
            "optimizers": {
                "actor": self.actor_optimizer.state_dict(),
                "reward": self.reward_optimizer.state_dict(),
                "cost": self.cost_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
                "multiplier": self.multiplier_optimizer.state_dict(),
            },
            "log_alpha": self.log_alpha.detach().clone(),
            "speed_multiplier": self.speed_multiplier.detach().clone(),
            "terminal_multiplier": self.terminal_multiplier.detach().clone(),
        }

    def _modules(self) -> dict[str, nn.Module]:
        return {
            "actor": self.actor,
            "reward_critic_1": self.reward_critic_1,
            "reward_critic_2": self.reward_critic_2,
            "speed_critic": self.speed_critic,
            "terminal_critic": self.terminal_critic,
            "intervention_critic": self.intervention_critic,
            "target_reward_critic_1": self.target_reward_critic_1,
            "target_reward_critic_2": self.target_reward_critic_2,
            "target_speed_critic": self.target_speed_critic,
            "target_terminal_critic": self.target_terminal_critic,
            "target_intervention_critic": self.target_intervention_critic,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["seed"]) != self.seed:
            raise RuntimeError("agent seed changed across recovery")
        if dict(state["config"]) != self.config.__dict__:
            raise RuntimeError("agent configuration changed across recovery")
        for name, module_state in state["modules"].items():
            self._modules()[name].load_state_dict(module_state)
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
            self.speed_multiplier.copy_(state["speed_multiplier"].to(self.device))
            self.terminal_multiplier.copy_(state["terminal_multiplier"].to(self.device))
        self.actor_optimizer.load_state_dict(state["optimizers"]["actor"])
        self.reward_optimizer.load_state_dict(state["optimizers"]["reward"])
        self.cost_optimizer.load_state_dict(state["optimizers"]["cost"])
        self.alpha_optimizer.load_state_dict(state["optimizers"]["alpha"])
        self.multiplier_optimizer.load_state_dict(state["optimizers"]["multiplier"])
        self.update_count = int(state["update_count"])
