"""Minimal HER adapter for fixed constraints and consistent terminal masks."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from .goal import (
    CURRENT_POSITION,
    PREVIOUS_POSITION,
    goal_transition_reward,
    relabeled_terminal_mask,
)


class GoalReplayBuffer(HerReplayBuffer):
    """Episode-complete replay with optional constraint-preserving HER.

    Stock SB3 HER intentionally leaves ``done`` unchanged and relabels the
    complete desired-goal vector. Neither behavior matches this benchmark.
    This adapter changes only those two mechanics and computes reward directly
    from stored goal arrays rather than a live environment or ``info``.
    """

    def __init__(
        self,
        *args,
        handle_timeout_termination: bool = False,
        n_sampled_goal: int = 4,
        goal_selection_strategy: GoalSelectionStrategy | str = "future",
        copy_info_dict: bool = False,
        **kwargs,
    ):
        if handle_timeout_termination:
            raise ValueError("The 180-second task horizon must remain terminal")
        if n_sampled_goal not in (0, 4):
            raise ValueError("Replay is frozen to zero or four virtual goals")
        if isinstance(goal_selection_strategy, GoalSelectionStrategy):
            is_future = goal_selection_strategy is GoalSelectionStrategy.FUTURE
        else:
            is_future = str(goal_selection_strategy).lower() == "future"
        if not is_future:
            raise ValueError("This protocol freezes HER goal selection to future")
        if copy_info_dict:
            raise ValueError("Relabeled reward must not depend on stored info")
        super().__init__(
            *args,
            handle_timeout_termination=False,
            n_sampled_goal=n_sampled_goal,
            goal_selection_strategy=GoalSelectionStrategy.FUTURE,
            copy_info_dict=False,
            **kwargs,
        )

    def _sample_goals(self, batch_indices, env_indices):
        """Sample a future position that was not already reached at ``s_t``.

        A stationary prefix can have no admissible future achieved position.
        Such a row falls back to its original canonical goal instead of creating
        a counterfactual transition that should have terminated before ``s_t``.
        """

        original_desired = np.array(
            self.observations["desired_goal"][batch_indices, env_indices],
            dtype=np.float64,
            copy=True,
        )
        sampled = original_desired.copy()
        for row, (batch_index, env_index) in enumerate(
            zip(batch_indices, env_indices, strict=True)
        ):
            episode_start = self.ep_start[batch_index, env_index]
            episode_length = self.ep_length[batch_index, env_index]
            current_in_episode = (
                batch_index - episode_start
            ) % self.buffer_size
            future_in_episode = np.arange(current_in_episode, episode_length)
            future_indices = (
                episode_start + future_in_episode
            ) % self.buffer_size
            positions = self.next_observations["achieved_goal"][
                future_indices, env_index, CURRENT_POSITION
            ]
            current_position = self.observations["achieved_goal"][
                batch_index, env_index, CURRENT_POSITION
            ]
            eligible = future_indices[positions > current_position]
            if eligible.size == 0:
                continue
            selected = int(np.random.choice(eligible))
            target = self.next_observations["achieved_goal"][
                selected, env_index, CURRENT_POSITION
            ]
            sampled[row, CURRENT_POSITION] = target
            sampled[row, PREVIOUS_POSITION] = target
        return sampled

    def _get_virtual_samples(self, batch_indices, env_indices, env=None):
        obs = {
            key: value[batch_indices, env_indices, :]
            for key, value in self.observations.items()
        }
        next_obs = {
            key: value[batch_indices, env_indices, :]
            for key, value in self.next_observations.items()
        }
        new_goals = self._sample_goals(batch_indices, env_indices)
        obs["desired_goal"] = new_goals
        next_obs["desired_goal"] = new_goals

        rewards = np.asarray(
            goal_transition_reward(next_obs["achieved_goal"], new_goals),
            dtype=np.float32,
        )
        dones = np.asarray(
            relabeled_terminal_mask(
                self.dones[batch_indices, env_indices],
                next_obs["achieved_goal"],
                new_goals,
            ),
            dtype=np.float32,
        )

        normalized_obs = self._normalize_obs(obs, env)
        normalized_next_obs = self._normalize_obs(next_obs, env)
        normalized_rewards = self._normalize_reward(rewards.reshape(-1, 1), env)
        return DictReplayBufferSamples(
            observations={
                key: self.to_torch(value)
                for key, value in normalized_obs.items()
            },
            actions=self.to_torch(self.actions[batch_indices, env_indices]),
            next_observations={
                key: self.to_torch(value)
                for key, value in normalized_next_obs.items()
            },
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(normalized_rewards),
        )
