"""Factories for the preregistered comparison; no training is started here."""

from __future__ import annotations

from benchmarks.scalar_sac.experiment import _base_environment

from .environment import GoalConditionedTask
from .replay_buffer import GoalReplayBuffer


def make_environment(configuration):
    """Return the shared real-rollout environment with the canonical goal."""

    return GoalConditionedTask(
        _base_environment(configuration),
        task=configuration.task,
        scales=configuration.goal_scales,
    )


def sac_model_kwargs(configuration, condition_id: str) -> dict:
    """Build matched SAC arguments; only the replay mechanism may differ."""

    condition = configuration.condition(condition_id)
    replay_buffer_kwargs = {
        "handle_timeout_termination": False,
        "n_sampled_goal": (
            configuration.her.n_sampled_goal
            if condition.hindsight_relabeling
            else 0
        ),
        "goal_selection_strategy": configuration.her.goal_selection_strategy,
        "copy_info_dict": configuration.her.copy_info_dict,
    }
    values = configuration.sac
    return {
        "policy": values.policy,
        "learning_rate": values.learning_rate,
        "buffer_size": values.buffer_size,
        "learning_starts": values.learning_starts,
        "batch_size": values.batch_size,
        "tau": values.tau,
        "gamma": values.gamma,
        "train_freq": values.train_freq,
        "gradient_steps": values.gradient_steps,
        "ent_coef": values.ent_coef,
        "policy_kwargs": {"net_arch": list(values.policy_network)},
        "replay_buffer_class": GoalReplayBuffer,
        "replay_buffer_kwargs": replay_buffer_kwargs,
    }


def build_model(configuration, condition_id, environment, seed, device="auto"):
    """Instantiate one preregistered learner without calling ``learn``."""

    from stable_baselines3 import SAC

    return SAC(
        env=environment,
        seed=seed,
        device=device,
        verbose=0,
        **sac_model_kwargs(configuration, condition_id),
    )
