"""Target decomposition, anticipation, and stability diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .adapter import ModelState, RecordedTransition, TrackContext
from .physics_rollout import batched_step, observations, pack_states
from .value_expansion import TargetDecomposition, _policy_draw

CRITIC_NAMES = ("objective", "speed_cost", "deadline_cost")


def _summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_decomposition(value: TargetDecomposition) -> dict[str, Any]:
    result: dict[str, Any] = {}
    denominator = np.maximum(np.abs(value.targets), 1e-12)
    fraction = np.abs(value.model_expansion) / denominator
    for index, name in enumerate(CRITIC_NAMES):
        result[name] = {
            "target": _summary(value.targets[:, index]),
            "real_two_step": _summary(value.real_prefix[:, index]),
            "model_expansion": _summary(value.model_expansion[:, index]),
            "terminal_or_bootstrap": _summary(value.bootstrap[:, index]),
            "model_expansion_fraction_of_target_magnitude": _summary(
                fraction[:, index]
            ),
        }
    result["rollout"] = {
        "model_steps_mean": float(value.model_steps.mean()),
        "model_steps_min": int(value.model_steps.min()),
        "model_steps_max": int(value.model_steps.max()),
        "true_terminal_count": int(value.true_terminal.sum()),
        "time_limit_bootstrap_count": int(value.time_limit_bootstrap.sum()),
    }
    return result


@dataclass
class TargetAccumulator:
    """Bounded-memory aggregation for checkpoint target diagnostics."""

    count: np.ndarray = field(default_factory=lambda: np.zeros((5, 3), dtype=np.int64))
    total: np.ndarray = field(
        default_factory=lambda: np.zeros((5, 3), dtype=np.float64)
    )
    total_square: np.ndarray = field(
        default_factory=lambda: np.zeros((5, 3), dtype=np.float64)
    )
    minimum: np.ndarray = field(
        default_factory=lambda: np.full((5, 3), np.inf, dtype=np.float64)
    )
    maximum: np.ndarray = field(
        default_factory=lambda: np.full((5, 3), -np.inf, dtype=np.float64)
    )
    model_step_count: int = 0
    model_step_total: int = 0
    model_step_minimum: int = np.iinfo(np.int64).max
    model_step_maximum: int = 0
    true_terminal_count: int = 0
    time_limit_bootstrap_count: int = 0
    update_count: int = 0

    def add(self, decomposition: TargetDecomposition) -> None:
        denominator = np.maximum(np.abs(decomposition.targets), 1e-12)
        components = np.stack(
            (
                decomposition.targets,
                decomposition.real_prefix,
                decomposition.model_expansion,
                decomposition.bootstrap,
                np.abs(decomposition.model_expansion) / denominator,
            )
        )
        batch_size = components.shape[1]
        self.count += batch_size
        self.total += components.sum(axis=1)
        self.total_square += np.square(components).sum(axis=1)
        self.minimum = np.minimum(self.minimum, components.min(axis=1))
        self.maximum = np.maximum(self.maximum, components.max(axis=1))
        steps = np.asarray(decomposition.model_steps, dtype=np.int64)
        self.model_step_count += steps.size
        self.model_step_total += int(steps.sum())
        self.model_step_minimum = min(self.model_step_minimum, int(steps.min()))
        self.model_step_maximum = max(self.model_step_maximum, int(steps.max()))
        self.true_terminal_count += int(decomposition.true_terminal.sum())
        self.time_limit_bootstrap_count += int(
            decomposition.time_limit_bootstrap.sum()
        )
        self.update_count += 1

    def drain(self, *, real_transitions: int) -> dict[str, Any]:
        component_names = (
            "target",
            "real_two_step",
            "model_expansion",
            "terminal_or_bootstrap",
            "model_expansion_fraction_of_target_magnitude",
        )
        critics: dict[str, Any] = {}
        for critic_index, critic_name in enumerate(CRITIC_NAMES):
            critics[critic_name] = {}
            for component_index, component_name in enumerate(component_names):
                count = int(self.count[component_index, critic_index])
                if count == 0:
                    statistics = {key: None for key in ("mean", "std", "min", "max")}
                else:
                    mean = self.total[component_index, critic_index] / count
                    variance = max(
                        self.total_square[component_index, critic_index] / count
                        - mean * mean,
                        0.0,
                    )
                    statistics = {
                        "mean": float(mean),
                        "std": float(np.sqrt(variance)),
                        "min": float(self.minimum[component_index, critic_index]),
                        "max": float(self.maximum[component_index, critic_index]),
                    }
                critics[critic_name][component_name] = statistics
        payload = {
            "record_type": "target_decomposition",
            "real_transitions": int(real_transitions),
            "updates": self.update_count,
            "critics": critics,
            "rollout": {
                "model_steps_mean": (
                    self.model_step_total / self.model_step_count
                    if self.model_step_count
                    else None
                ),
                "model_steps_min": (
                    self.model_step_minimum if self.model_step_count else None
                ),
                "model_steps_max": (
                    self.model_step_maximum if self.model_step_count else None
                ),
                "true_terminal_count": self.true_terminal_count,
                "time_limit_bootstrap_count": self.time_limit_bootstrap_count,
            },
        }
        self.__dict__.update(TargetAccumulator().__dict__)
        return payload


def anticipatory_speed_diagnostic(
    *,
    policy: Any,
    state: ModelState,
    context: TrackContext,
    diagnostic_actions: Sequence[float] = (1.0, 0.0, -1.0),
    horizon: int,
    rollout_rng: np.random.Generator,
    vehicle: Any,
    simulation_config: Any,
    gamma: float = 0.99,
) -> dict[str, float]:
    """Compare first actions without feeding the diagnostic back into control."""

    labels = {1.0: "accelerate_now", 0.0: "coast", -1.0: "brake_now"}
    actions = np.asarray(diagnostic_actions, dtype=np.float64)
    states = pack_states([state] * len(actions))
    contexts = [context] * len(actions)
    returns = np.zeros(len(actions), dtype=np.float64)
    active = np.ones(len(actions), dtype=bool)
    for step_index in range(horizon):
        if step_index == 0:
            step_actions = actions
        else:
            obs = observations(states, contexts, vehicle, simulation_config)
            step_actions, _log_probability, _critic_action = _policy_draw(
                policy, obs, rollout_rng
            )
        transition = batched_step(
            states,
            step_actions,
            contexts=contexts,
            vehicle=vehicle,
            config=simulation_config,
        )
        returns += active * gamma**step_index * transition.costs[:, 0]
        active &= ~(transition.terminated | transition.truncated)
        states = transition.state
        if not active.any():
            break
    return {
        labels.get(float(action), f"action_{float(action):g}"): float(value)
        for action, value in zip(actions, returns)
    }


def downward_limit_probe_metadata(
    transition: RecordedTransition,
) -> dict[str, float] | None:
    """Describe the first lower limit visible from a stored real source state."""

    position = transition.state.vehicle.position_m
    track = transition.context.track
    segment = int(np.searchsorted(track.positions_m, position, side="right") - 1)
    segment = int(np.clip(segment, 0, len(track.positions_m) - 1))
    current_limit = float(track.limits_m_s[segment])
    for target in range(segment + 1, len(track.positions_m)):
        target_limit = float(track.limits_m_s[target])
        if target_limit >= current_limit:
            continue
        distance = float(track.positions_m[target] - position)
        if 0.0 < distance <= transition.context.sensor_range_m:
            return {
                "position_m": float(position),
                "velocity_m_s": float(transition.state.vehicle.velocity_m_s),
                "current_limit_m_s": current_limit,
                "future_limit_m_s": target_limit,
                "transition_distance_m": distance,
            }
        return None
    return None


def late_collapse(
    development_rsr: Sequence[int], *, minimum_peak_to_final_drop: int = 3
) -> dict[str, Any]:
    if not development_rsr:
        raise ValueError("At least one Development RSR count is required")
    values = tuple(int(item) for item in development_rsr)
    peak = max(values)
    drop = peak - values[-1]
    return {
        "peak_rsr": peak,
        "final_rsr": values[-1],
        "peak_to_final_drop": drop,
        "late_collapse": drop >= minimum_peak_to_final_drop,
        "threshold": minimum_peak_to_final_drop,
    }


def numerical_instability(
    *,
    lambda_speed: float,
    lambda_deadline: float,
    critic_losses: Sequence[float],
    target_magnitudes: Sequence[float],
    multiplier_threshold: float = 1000.0,
    critic_loss_threshold: float = 1_000_000.0,
    target_threshold: float = 1_000_000.0,
) -> bool:
    values = np.asarray(
        [lambda_speed, lambda_deadline, *critic_losses, *target_magnitudes],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        return True
    return bool(
        max(lambda_speed, lambda_deadline) >= multiplier_threshold
        or max(critic_losses, default=0.0) >= critic_loss_threshold
        or max(target_magnitudes, default=0.0) >= target_threshold
    )
