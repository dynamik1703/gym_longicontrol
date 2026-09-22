"""A small, import-safe Soft Actor-Critic implementation.

The implementation keeps the value-network SAC variant used by the original
project while updating its environment interaction to Gymnasium's API.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch.nn.functional import relu
from torch.optim import Adam


def seed_everything(seed: int, environment: Any | None = None) -> None:
    """Seed Python, NumPy, PyTorch and, when supplied, an environment's spaces."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if environment is not None:
        for space_name in ("action_space", "observation_space"):
            space = getattr(environment, space_name, None)
            if space is not None and hasattr(space, "seed"):
                space.seed(seed)


def reset_environment(
    environment: Any, seed: int | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reset a Gymnasium environment and validate its two-value response."""

    result = environment.reset(seed=seed) if seed is not None else environment.reset()
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(
            "The environment must implement Gymnasium reset() and return "
            "(observation, info)."
        )
    observation, info = result
    if not isinstance(info, dict):
        raise TypeError("Gymnasium reset() info must be a dictionary.")
    return np.asarray(observation, dtype=np.float32), info


def step_environment(
    environment: Any, action: np.ndarray
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    """Step a Gymnasium environment and validate terminated/truncated flags."""

    result = environment.step(action)
    if not isinstance(result, tuple) or len(result) != 5:
        raise TypeError(
            "The environment must implement Gymnasium step() and return "
            "(observation, reward, terminated, truncated, info)."
        )
    observation, reward, terminated, truncated, info = result
    if not isinstance(info, dict):
        raise TypeError("Gymnasium step() info must be a dictionary.")
    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        bool(terminated),
        bool(truncated),
        info,
    )


def numpy_to_torch(value: Any, device: torch.device | str = "cpu") -> Any:
    """Convert NumPy-like values recursively to float tensors."""

    if isinstance(value, tuple):
        return tuple(numpy_to_torch(element, device=device) for element in value)
    if isinstance(value, list):
        return [numpy_to_torch(element, device=device) for element in value]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def torch_to_numpy(value: Any) -> Any:
    """Convert tensors recursively without retaining graphs or device storage."""

    if isinstance(value, tuple):
        return tuple(torch_to_numpy(element) for element in value)
    if isinstance(value, list):
        return [torch_to_numpy(element) for element in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def weights_init_(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.1)


class MLP(torch.nn.Module):
    def __init__(self, hidden_sizes: Sequence[int], output_size: int, input_size: int):
        super().__init__()
        self.hidden_activation = relu
        self.fcs = torch.nn.ModuleList()
        previous_size = int(input_size)
        for next_size in hidden_sizes:
            self.fcs.append(torch.nn.Linear(previous_size, int(next_size)))
            previous_size = int(next_size)
        self.last_fc = torch.nn.Linear(previous_size, int(output_size))

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        hidden = net_input
        for layer in self.fcs:
            hidden = self.hidden_activation(layer(hidden))
        return self.last_fc(hidden)


class QNetwork(MLP):
    def __init__(self, hidden_sizes: Sequence[int], output_size: int, input_size: int):
        super().__init__(hidden_sizes, output_size, input_size)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return super().forward(torch.cat((state, action), dim=1))


class ValueNetwork(MLP):
    def __init__(self, hidden_sizes: Sequence[int], output_size: int, input_size: int):
        super().__init__(hidden_sizes, output_size, input_size)
        self.apply(weights_init_)


class PolicyNetwork(MLP):
    def __init__(self, hidden_sizes: Sequence[int], output_size: int, input_size: int):
        super().__init__(hidden_sizes, output_size, input_size)
        self.last_fc_mean = self.last_fc
        final_hidden_size = self.last_fc_mean.in_features
        self.last_fc_log_std = torch.nn.Linear(final_hidden_size, int(output_size))
        self.epsilon = 1e-6
        self.apply(weights_init_)

    def get_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        action, _, _, _ = self.forward(state.unsqueeze(0), deterministic=deterministic)
        return action.squeeze(0).detach()

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        hidden = state
        for layer in self.fcs:
            hidden = self.hidden_activation(layer(hidden))

        mean = self.last_fc_mean(hidden)
        if deterministic:
            return torch.tanh(mean), mean, None, None

        log_std = torch.clamp(self.last_fc_log_std(hidden), -20, 2)
        distribution = torch.distributions.Normal(mean, torch.exp(log_std))
        sample = distribution.rsample()
        action = torch.tanh(sample)
        log_prob = distribution.log_prob(sample)
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        return action, mean, log_std, log_prob.sum(1, keepdim=True)


class InitPolicy:
    """Uniform random initialization policy over a continuous action space."""

    def __init__(
        self,
        low: np.ndarray | float = -1.0,
        high: np.ndarray | float = 1.0,
        shape: tuple[int, ...] | None = None,
        seed: int | None = None,
    ):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = shape
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_action_space(
        cls, action_space: Any, seed: int | None = None
    ) -> InitPolicy:
        return cls(action_space.low, action_space.high, tuple(action_space.shape), seed)

    def get_action(self, *_: Any, **__: Any) -> torch.Tensor:
        sample = self.rng.uniform(self.low, self.high, size=self.shape)
        return torch.as_tensor(sample, dtype=torch.float32)


class ReplayBuffer:
    """Fixed-size replay buffer with its own reproducible random generator."""

    def __init__(
        self,
        buffer_capacity: int,
        batch_size: int,
        state_dim: int,
        action_dim: int,
        seed: int | None = None,
    ):
        self.capacity = int(buffer_capacity)
        self.batch_size = int(batch_size)
        if self.capacity <= 0:
            raise ValueError("buffer_capacity must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.current_size = 0
        self.index = 0
        self.rng = np.random.default_rng(seed)
        self.S = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)
        self.A = np.zeros((self.capacity, int(action_dim)), dtype=np.float32)
        self.R = np.zeros((self.capacity, 1), dtype=np.float32)
        self.S_prime = np.zeros((self.capacity, int(state_dim)), dtype=np.float32)
        # A truncation ends an episode but is not terminal for Bellman bootstrapping.
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

    def __len__(self) -> int:
        return self.current_size

    @property
    def can_sample(self) -> bool:
        return self.current_size >= self.batch_size

    def add_experience(self, experience: tuple[Any, Any, float, Any, bool]) -> None:
        state, action, reward, next_state, terminated = experience
        # Write before incrementing: index zero must be used for the first sample.
        self.S[self.index] = np.asarray(state, dtype=np.float32)
        self.A[self.index] = np.asarray(action, dtype=np.float32)
        self.R[self.index] = float(reward)
        self.S_prime[self.index] = np.asarray(next_state, dtype=np.float32)
        self.done[self.index] = float(terminated)

        self.index = (self.index + 1) % self.capacity
        self.current_size = min(self.current_size + 1, self.capacity)

    def sample_minibatch(self) -> tuple[np.ndarray, ...]:
        if not self.can_sample:
            raise ValueError(
                f"Cannot sample {self.batch_size} transitions from a buffer "
                f"containing {self.current_size}."
            )
        indices = self.rng.choice(self.current_size, self.batch_size, replace=False)
        return (
            self.S[indices],
            self.A[indices],
            self.R[indices],
            self.S_prime[indices],
            self.done[indices],
        )


class SAC:
    def __init__(
        self,
        environment: Any,
        policy_function: PolicyNetwork,
        q1_function: QNetwork,
        q2_function: QNetwork,
        value_function: ValueNetwork,
        replay_buffer: ReplayBuffer,
        adam_learning_rate: float,
        target_entropy: float,
        discount_factor_gamma: float,
        soft_update_factor_tau: float,
        evaluation_environment: Any | None = None,
        device: torch.device | str = "cpu",
    ):
        self.env = environment
        self.evaluation_env = evaluation_environment or environment
        self.device = torch.device(device)
        self.policy_function = policy_function.to(self.device)
        self.q1_function = q1_function.to(self.device)
        self.q2_function = q2_function.to(self.device)
        self.value_function = value_function.to(self.device)
        self.target_value_function = copy.deepcopy(self.value_function).to(self.device)
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, device=self.device))

        self.policy_function_optimizer = Adam(
            self.policy_function.parameters(), lr=adam_learning_rate
        )
        self.q1_function_optimizer = Adam(
            self.q1_function.parameters(), lr=adam_learning_rate
        )
        self.q2_function_optimizer = Adam(
            self.q2_function.parameters(), lr=adam_learning_rate
        )
        self.value_function_optimizer = Adam(
            self.value_function.parameters(), lr=adam_learning_rate
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=adam_learning_rate)

        self.replay_buffer = replay_buffer
        self.target_entropy = float(target_entropy)
        self.gamma = float(discount_factor_gamma)
        self.tau = float(soft_update_factor_tau)
        self.loss_criterion = torch.nn.MSELoss()
        self.losses = {
            "q1_loss": 0.0,
            "q2_loss": 0.0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "alpha_loss": 0.0,
        }

    def _tensor(self, value: Any) -> torch.Tensor:
        return numpy_to_torch(value, device=self.device)

    def _policy_action(self, state: np.ndarray, deterministic: bool) -> np.ndarray:
        state_tensor = self._tensor(state)
        with torch.no_grad():
            action = self.policy_function.get_action(
                state_tensor, deterministic=deterministic
            )
        return torch_to_numpy(action).reshape(self.env.action_space.shape)

    def do_training(self, num_steps_per_epoch: int, seed: int | None = None) -> None:
        state, _ = reset_environment(self.env, seed=seed)
        for _ in range(int(num_steps_per_epoch)):
            action = self._policy_action(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = step_environment(
                self.env, action
            )
            self.replay_buffer.add_experience(
                (state, action, reward, next_state, terminated)
            )

            if self.replay_buffer.can_sample:
                batch = numpy_to_torch(
                    self.replay_buffer.sample_minibatch(), device=self.device
                )
                self.update_parameters_for_batch(batch)
                self.update_value_function()

            state = next_state
            if terminated or truncated:
                state, _ = reset_environment(self.env)

    def do_evaluation(
        self,
        num_evaluation_episodes: int,
        record: bool = False,
        save_dname: str | Path | None = None,
        seed: int = 2,
    ) -> tuple[float, int]:
        if record:
            self.do_visualization(record=True, save_dname=save_dname, seed=seed)

        returns: list[float] = []
        env = self.evaluation_env
        for episode in range(int(num_evaluation_episodes)):
            state, _ = reset_environment(env, seed=seed + episode)
            episode_return = 0.0
            while True:
                state_tensor = self._tensor(state)
                with torch.no_grad():
                    action_tensor = self.policy_function.get_action(
                        state_tensor, deterministic=True
                    )
                action = torch_to_numpy(action_tensor).reshape(env.action_space.shape)
                state, reward, terminated, truncated, _ = step_environment(env, action)
                episode_return += reward
                if terminated or truncated:
                    returns.append(episode_return)
                    break
        return float(np.mean(returns)), len(returns)

    def do_visualization(
        self,
        record: bool = False,
        save_dname: str | Path | None = None,
        environment: Any | None = None,
        seed: int = 2,
    ) -> float:
        """Run one deterministic episode, optionally through ``RecordVideo``.

        A recording environment must have been created with
        ``render_mode='rgb_array'``.  Likewise, interactive visualization should
        receive an environment created with ``render_mode='human'``.
        """

        env = environment or self.evaluation_env
        wrapped_env = env
        if record:
            if save_dname is None:
                raise ValueError("save_dname is required when record=True")
            try:
                from gymnasium.wrappers import RecordVideo
            except ImportError as error:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "Video recording requires Gymnasium's video dependencies."
                ) from error
            video_folder = Path(save_dname) / "videos" / uuid4().hex
            video_folder.parent.mkdir(parents=True, exist_ok=True)
            wrapped_env = RecordVideo(
                env,
                video_folder=str(video_folder),
                episode_trigger=lambda _: True,
                name_prefix="sac-evaluation",
            )

        try:
            state, _ = reset_environment(wrapped_env, seed=seed)
            episode_return = 0.0
            while True:
                state_tensor = self._tensor(state)
                with torch.no_grad():
                    action_tensor = self.policy_function.get_action(
                        state_tensor, deterministic=True
                    )
                action = torch_to_numpy(action_tensor).reshape(
                    wrapped_env.action_space.shape
                )
                state, reward, terminated, truncated, _ = step_environment(
                    wrapped_env, action
                )
                episode_return += reward
                if terminated or truncated:
                    return episode_return
        finally:
            if wrapped_env is not env:
                wrapped_env.close()

    def init_replay_buffer(
        self,
        init_policy: InitPolicy | PolicyNetwork,
        fill_size: int,
        seed: int | None = None,
    ) -> None:
        state, _ = reset_environment(self.env, seed=seed)
        for _ in range(int(fill_size)):
            state_tensor = self._tensor(state)
            with torch.no_grad():
                action_tensor = init_policy.get_action(
                    state_tensor, deterministic=False
                )
            action = torch_to_numpy(action_tensor).reshape(self.env.action_space.shape)
            next_state, reward, terminated, truncated, _ = step_environment(
                self.env, action
            )
            self.replay_buffer.add_experience(
                (state, action, reward, next_state, terminated)
            )
            state = next_state
            if terminated or truncated:
                state, _ = reset_environment(self.env)

    def update_parameters_for_batch(self, batch: tuple[torch.Tensor, ...]) -> None:
        state, action, reward, next_state, terminated = batch

        q1_prediction = self.q1_function(state, action)
        q2_prediction = self.q2_function(state, action)
        value_prediction = self.value_function(state)
        new_action, _, _, log_pi = self.policy_function(state)
        assert log_pi is not None

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_value_prediction = self.target_value_function(next_state)
            q_target = reward + (1 - terminated) * self.gamma * next_value_prediction
        q1_loss = self.loss_criterion(q1_prediction, q_target)
        q2_loss = self.loss_criterion(q2_prediction, q_target)

        with torch.no_grad():
            q1_new_action = self.q1_function(state, new_action)
            q2_new_action = self.q2_function(state, new_action)
            value_target = torch.min(q1_new_action, q2_new_action) - alpha * log_pi
        value_loss = self.loss_criterion(value_prediction, value_target)

        self.q1_function_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_function_optimizer.step()

        self.q2_function_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_function_optimizer.step()

        self.value_function_optimizer.zero_grad()
        value_loss.backward()
        self.value_function_optimizer.step()

        for parameter in self.q1_function.parameters():
            parameter.requires_grad_(False)
        for parameter in self.q2_function.parameters():
            parameter.requires_grad_(False)
        try:
            policy_action, _, _, policy_log_pi = self.policy_function(state)
            assert policy_log_pi is not None
            q_policy = torch.min(
                self.q1_function(state, policy_action),
                self.q2_function(state, policy_action),
            )
            policy_loss = (alpha * policy_log_pi - q_policy).mean()
            self.policy_function_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_function_optimizer.step()
        finally:
            for parameter in self.q1_function.parameters():
                parameter.requires_grad_(True)
            for parameter in self.q2_function.parameters():
                parameter.requires_grad_(True)

        losses: Iterable[tuple[str, torch.Tensor]] = (
            ("q1_loss", q1_loss),
            ("q2_loss", q2_loss),
            ("value_loss", value_loss),
            ("policy_loss", policy_loss),
            ("alpha_loss", alpha_loss),
        )
        for name, loss in losses:
            self.losses[name] = float(loss.detach().cpu().item())

    def update_value_function(self) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_value_function.parameters(),
                self.value_function.parameters(),
            ):
                target_parameter.lerp_(parameter, self.tau)
