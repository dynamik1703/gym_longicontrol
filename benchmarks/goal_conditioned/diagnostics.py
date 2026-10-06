"""Observational replay diagnostics that do not alter sampled training data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .goal import (
    CURRENT_POSITION,
    ELAPSED_TIME,
    MAX_SPEED_VIOLATION,
)
from .replay_buffer import GoalReplayBuffer


@dataclass
class ReplayDiagnosticCounters:
    """Counts over replay rows already sampled by SAC for optimization."""

    total_sampled_rows: int = 0
    real_rows: int = 0
    virtual_rows: int = 0
    eligible_virtual_rows: int = 0
    relabeled_virtual_rows: int = 0
    fallback_virtual_rows: int = 0
    eligible_original_target_rows: int = 0
    positive_real_reward_rows: int = 0
    positive_virtual_reward_rows: int = 0
    virtual_terminal_rows: int = 0
    relabel_added_terminal_rows: int = 0
    zero_virtual_deadline_failure_rows: int = 0
    zero_virtual_violation_failure_rows: int = 0
    virtual_target_distance_m_sum: float = 0.0
    virtual_target_distance_m_min: float | None = None
    virtual_target_distance_m_max: float | None = None

    def observe_target_distances(self, distances_m: np.ndarray) -> None:
        if distances_m.size == 0:
            return
        self.virtual_target_distance_m_sum += float(distances_m.sum())
        minimum = float(distances_m.min())
        maximum = float(distances_m.max())
        self.virtual_target_distance_m_min = (
            minimum
            if self.virtual_target_distance_m_min is None
            else min(self.virtual_target_distance_m_min, minimum)
        )
        self.virtual_target_distance_m_max = (
            maximum
            if self.virtual_target_distance_m_max is None
            else max(self.virtual_target_distance_m_max, maximum)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["denominators"] = {
            "replay_row_rates": "total_sampled_rows",
            "real_reward_rate": "real_rows",
            "virtual_row_rates": "virtual_rows",
            "virtual_target_distance": "virtual_rows",
        }
        payload["rates"] = {
            "real_row_fraction": _ratio(self.real_rows, self.total_sampled_rows),
            "virtual_row_fraction": _ratio(
                self.virtual_rows, self.total_sampled_rows
            ),
            "positive_real_reward_rate": _ratio(
                self.positive_real_reward_rows, self.real_rows
            ),
            "eligible_virtual_rate": _ratio(
                self.eligible_virtual_rows, self.virtual_rows
            ),
            "relabeled_virtual_rate": _ratio(
                self.relabeled_virtual_rows, self.virtual_rows
            ),
            "fallback_virtual_rate": _ratio(
                self.fallback_virtual_rows, self.virtual_rows
            ),
            "positive_virtual_reward_rate": _ratio(
                self.positive_virtual_reward_rows, self.virtual_rows
            ),
            "virtual_terminal_rate": _ratio(
                self.virtual_terminal_rows, self.virtual_rows
            ),
            "relabel_added_terminal_rate": _ratio(
                self.relabel_added_terminal_rows, self.virtual_rows
            ),
            "zero_virtual_deadline_failure_rate": _ratio(
                self.zero_virtual_deadline_failure_rows, self.virtual_rows
            ),
            "zero_virtual_violation_failure_rate": _ratio(
                self.zero_virtual_violation_failure_rows, self.virtual_rows
            ),
        }
        payload["mean_virtual_target_distance_m"] = (
            self.virtual_target_distance_m_sum / self.virtual_rows
            if self.virtual_rows
            else None
        )
        return payload


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _tensor_array(value) -> np.ndarray:
    return value.detach().cpu().numpy()


class DiagnosticGoalReplayBuffer(GoalReplayBuffer):
    """Add counters after normal replay sampling, without another RNG draw."""

    def __init__(self, *args, diagnostics_enabled: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.replay_diagnostics = ReplayDiagnosticCounters()

    def _get_real_samples(self, batch_indices, env_indices, env=None):
        samples = super()._get_real_samples(batch_indices, env_indices, env)
        if self.diagnostics_enabled:
            rewards = _tensor_array(samples.rewards).reshape(-1)
            count = int(rewards.size)
            self.replay_diagnostics.total_sampled_rows += count
            self.replay_diagnostics.real_rows += count
            self.replay_diagnostics.positive_real_reward_rows += int(
                np.count_nonzero(rewards > 0.0)
            )
        return samples

    def _get_virtual_samples(self, batch_indices, env_indices, env=None):
        samples = super()._get_virtual_samples(batch_indices, env_indices, env)
        if not self.diagnostics_enabled:
            return samples

        rewards = _tensor_array(samples.rewards).reshape(-1)
        dones = _tensor_array(samples.dones).reshape(-1).astype(bool)
        count = int(rewards.size)
        self.replay_diagnostics.total_sampled_rows += count
        self.replay_diagnostics.virtual_rows += count
        if count == 0:
            return samples

        desired = _tensor_array(samples.observations["desired_goal"])
        achieved = _tensor_array(samples.observations["achieved_goal"])
        next_achieved = _tensor_array(samples.next_observations["achieved_goal"])
        original_desired = self.observations["desired_goal"][
            batch_indices, env_indices
        ]
        target_changed = desired[:, CURRENT_POSITION] != original_desired[
            :, CURRENT_POSITION
        ]

        eligible = np.zeros(count, dtype=bool)
        for row, (batch_index, env_index) in enumerate(
            zip(batch_indices, env_indices, strict=True)
        ):
            episode_start = self.ep_start[batch_index, env_index]
            episode_length = self.ep_length[batch_index, env_index]
            current_in_episode = (
                batch_index - episode_start
            ) % self.buffer_size
            future_indices = (
                episode_start + np.arange(current_in_episode, episode_length)
            ) % self.buffer_size
            future_positions = self.next_observations["achieved_goal"][
                future_indices, env_index, CURRENT_POSITION
            ]
            current_position = self.observations["achieved_goal"][
                batch_index, env_index, CURRENT_POSITION
            ]
            eligible[row] = bool(np.any(future_positions > current_position))

        positive = rewards > 0.0
        zero = ~positive
        original_done = self.dones[batch_indices, env_indices].astype(bool)
        distance_m = np.maximum(
            desired[:, CURRENT_POSITION] - achieved[:, CURRENT_POSITION], 0.0
        ) * 1000.0

        counters = self.replay_diagnostics
        counters.eligible_virtual_rows += int(np.count_nonzero(eligible))
        counters.relabeled_virtual_rows += int(np.count_nonzero(target_changed))
        counters.fallback_virtual_rows += int(np.count_nonzero(~eligible))
        counters.eligible_original_target_rows += int(
            np.count_nonzero(eligible & ~target_changed)
        )
        counters.positive_virtual_reward_rows += int(np.count_nonzero(positive))
        counters.virtual_terminal_rows += int(np.count_nonzero(dones))
        counters.relabel_added_terminal_rows += int(
            np.count_nonzero(dones & ~original_done)
        )
        counters.zero_virtual_deadline_failure_rows += int(
            np.count_nonzero(
                zero
                & (
                    next_achieved[:, ELAPSED_TIME]
                    > desired[:, ELAPSED_TIME]
                )
            )
        )
        counters.zero_virtual_violation_failure_rows += int(
            np.count_nonzero(
                zero
                & (
                    next_achieved[:, MAX_SPEED_VIOLATION]
                    > desired[:, MAX_SPEED_VIOLATION]
                )
            )
        )
        counters.observe_target_distances(distance_m)
        return samples
