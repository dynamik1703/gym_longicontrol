"""One-step real-state branching and exact real/model batch composition."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from gym_longicontrol.domain.state import SimulationConfig
from gym_longicontrol.domain.vehicle import VehicleModel

from .adapter import OneStepModel, model_step
from .config import ImaginationConfig
from .model_state import ModelState, ProjectedTransition, TrackContext


@dataclass(frozen=True)
class RealReplaySource:
    transition_id: int
    state: ModelState
    context: TrackContext
    observation: np.ndarray


class RealSourceReplay:
    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.items: list[RealReplaySource] = []

    def append(self, value: RealReplaySource) -> None:
        self.items.append(value)
        if len(self.items) > self.capacity:
            del self.items[: len(self.items) - self.capacity]

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> list[RealReplaySource]:
        if not self.items:
            raise ValueError("Real replay is empty")
        indices = rng.integers(0, len(self.items), size=count)
        return [self.items[int(index)] for index in indices]


class SyntheticReplay:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: list[ProjectedTransition] = []
        self.generated_total = 0
        self.sampled_total = 0

    def extend(self, values: list[ProjectedTransition]) -> None:
        if any(value.source != "model" for value in values):
            raise ValueError("Synthetic replay accepts model transitions only")
        self.items.extend(values)
        self.generated_total += len(values)
        if len(self.items) > self.capacity:
            del self.items[: len(self.items) - self.capacity]

    def sample(
        self, count: int, rng: np.random.Generator
    ) -> list[ProjectedTransition]:
        if len(self.items) < count:
            raise ValueError("Not enough synthetic transitions for mixed update")
        ids = rng.integers(0, len(self.items), size=count)
        self.sampled_total += count
        return [self.items[int(index)] for index in ids]


def refresh_due(real_transitions: int, config: ImaginationConfig) -> bool:
    return bool(
        real_transitions >= config.warmup_real_transitions
        and (real_transitions - config.warmup_real_transitions)
        % config.refresh_real_transitions
        == 0
    )


def generate_synthetic_transitions(
    *,
    model: OneStepModel,
    real_replay: RealSourceReplay,
    actor: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    count: int,
    rng: np.random.Generator,
    vehicle: VehicleModel,
    simulation_config: SimulationConfig,
    energy_scale_kwh: float,
    deadline_s: float,
    max_episode_steps: int,
) -> list[ProjectedTransition]:
    """Generate exactly H=1 samples from real states with current actor actions."""

    generated: list[ProjectedTransition] = []
    for source in real_replay.sample(count, rng):
        action_array = np.asarray(
            actor(source.observation.copy(), rng), dtype=np.float64
        )
        if action_array.shape != (1,) or not np.isfinite(action_array).all():
            raise ValueError("Current stochastic actor returned an invalid action")
        generated.append(
            model_step(
                model,
                source.state,
                float(np.clip(action_array[0], -1.0, 1.0)),
                context=source.context,
                vehicle=vehicle,
                config=simulation_config,
                energy_scale_kwh=energy_scale_kwh,
                deadline_s=deadline_s,
                max_episode_steps=max_episode_steps,
                source_real_transition_id=source.transition_id,
                rng=rng,
            )
        )
    return generated


def exact_mixed_batch(
    real_items: list[object],
    synthetic_replay: SyntheticReplay,
    *,
    config: ImaginationConfig,
    rng: np.random.Generator,
) -> tuple[list[object], list[ProjectedTransition]]:
    if len(real_items) != config.real_batch_size:
        raise ValueError("Mixed batch must contain exactly 128 real samples")
    model_items = synthetic_replay.sample(config.model_batch_size, rng)
    return real_items, model_items


class RealEpisodePIDGate:
    """Forbid any model event from driving SACLag's episodic PID update."""

    def __init__(self):
        self.real_episode_updates = 0

    def update(self, policy, *, source: str, stats: dict, **kwargs) -> None:
        if source != "real":
            raise ValueError("Synthetic episodes must never update PID multipliers")
        policy.pre_update_fn(stats_train=stats, **kwargs)
        self.real_episode_updates += 1
