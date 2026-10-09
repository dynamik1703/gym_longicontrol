"""MBPO-style probabilistic ensemble for LongiControl vehicle dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gym_longicontrol.domain.state import SimulationConfig, VehicleState

from .config import ModelConfig
from .model_state import ModelState, VehiclePrediction


@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> Normalization:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or len(array) == 0 or not np.isfinite(array).all():
            raise ValueError("Normalization requires a finite non-empty matrix")
        return cls(array.mean(axis=0), np.maximum(array.std(axis=0), 1e-12))

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) / self.std

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.std + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - optional benchmark dependency
        raise RuntimeError("Learned MBRL requires the project 'train' extra") from error
    return torch, nn


def _member(config: ModelConfig):
    torch, nn = _torch()

    class Swish(nn.Module):
        def forward(self, value):
            return value * torch.sigmoid(value)

    layers: list[Any] = []
    width = len(config.input_fields)
    for hidden in config.hidden_sizes:
        linear = nn.Linear(width, hidden)
        nn.init.trunc_normal_(linear.weight, std=1.0 / (2.0 * width**0.5))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, Swish()))
        width = hidden
    layers.append(nn.Linear(width, 2 * len(config.target_fields)))
    return nn.Sequential(*layers)


class ProbabilisticEnsemble:
    condition = "learned"

    def __init__(
        self,
        config: ModelConfig,
        simulation_config: SimulationConfig,
        *,
        seed: int,
        device: str = "cpu",
    ):
        torch, nn = _torch()
        self.config = config
        self.simulation_config = simulation_config
        self.seed = int(seed)
        self.device = torch.device(device)
        torch.manual_seed(self.seed)
        self.members = nn.ModuleList(
            [_member(config) for _ in range(config.ensemble_size)]
        ).to(self.device)
        self.optimizers = [
            torch.optim.Adam(member.parameters(), lr=config.learning_rate)
            for member in self.members
        ]
        self.input_normalization: Normalization | None = None
        self.target_normalization: Normalization | None = None
        self.elite_indices: tuple[int, ...] = ()
        self.version = 0
        self.training_examples_seen = 0

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.members.parameters())

    def _distribution(self, member, inputs):
        torch, _nn = _torch()
        output = member(inputs)
        mean, raw_logvar = output.chunk(2, dim=-1)
        maximum = self.config.maximum_log_variance
        minimum = self.config.minimum_log_variance
        logvar = maximum - torch.nn.functional.softplus(maximum - raw_logvar)
        logvar = minimum + torch.nn.functional.softplus(logvar - minimum)
        return mean, logvar

    def negative_log_likelihood(self, member, inputs, targets):
        torch, _nn = _torch()
        mean, logvar = self._distribution(member, inputs)
        inverse_variance = torch.exp(-logvar)
        return ((mean - targets) ** 2 * inverse_variance + logvar).mean()

    def require_ready(self) -> None:
        if (
            self.version <= 0
            or not self.elite_indices
            or self.input_normalization is None
            or self.target_normalization is None
        ):
            raise RuntimeError("The learned model has not completed a verified refresh")

    def predict(
        self, state: ModelState, action: float, *, rng: np.random.Generator
    ) -> VehiclePrediction:
        torch, _nn = _torch()
        self.require_ready()
        assert self.input_normalization is not None
        assert self.target_normalization is not None
        member_index = int(rng.choice(self.elite_indices))
        normalized = self.input_normalization.normalize(
            state.model_input(action)[None, :]
        )
        inputs = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean_tensor, logvar_tensor = self._distribution(
                self.members[member_index], inputs
            )
        mean_n = mean_tensor.cpu().numpy()[0].astype(np.float64)
        variance_n = np.exp(logvar_tensor.cpu().numpy()[0].astype(np.float64))
        sample_n = mean_n + np.sqrt(variance_n) * rng.normal(size=mean_n.shape)
        sample = self.target_normalization.denormalize(sample_n)
        physical_variance = variance_n * self.target_normalization.std**2
        delta_position, delta_velocity, next_acceleration, step_energy = sample
        dt = self.simulation_config.dt_s
        next_vehicle = VehicleState(
            position_m=state.vehicle.position_m + float(delta_position),
            velocity_m_s=state.vehicle.velocity_m_s + float(delta_velocity),
            acceleration_m_s2=float(next_acceleration),
            jerk_m_s3=abs(next_acceleration - state.vehicle.acceleration_m_s2) / dt,
            elapsed_time_s=state.vehicle.elapsed_time_s + dt,
            total_energy_kwh=state.vehicle.total_energy_kwh + float(step_energy),
        )
        return VehiclePrediction(
            next_vehicle=next_vehicle,
            signed_step_energy_kwh=float(step_energy),
            ensemble_member=member_index,
            predictive_variance=tuple(float(value) for value in physical_variance),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "members": self.members.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "input_normalization": (
                None
                if self.input_normalization is None
                else self.input_normalization.state_dict()
            ),
            "target_normalization": (
                None
                if self.target_normalization is None
                else self.target_normalization.state_dict()
            ),
            "elite_indices": self.elite_indices,
            "version": self.version,
            "training_examples_seen": self.training_examples_seen,
            "seed": self.seed,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.members.load_state_dict(payload["members"])
        for optimizer, state in zip(self.optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
        for name in ("input_normalization", "target_normalization"):
            value = payload[name]
            setattr(
                self,
                name,
                None
                if value is None
                else Normalization(
                    np.asarray(value["mean"], dtype=np.float64),
                    np.asarray(value["std"], dtype=np.float64),
                ),
            )
        self.elite_indices = tuple(int(value) for value in payload["elite_indices"])
        self.version = int(payload["version"])
        self.training_examples_seen = int(payload["training_examples_seen"])
