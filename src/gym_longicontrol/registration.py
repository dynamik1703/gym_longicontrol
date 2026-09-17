"""Idempotent registration of the versioned Gymnasium environments."""

from gymnasium.envs.registration import register, registry


def register_envs():
    for name in (
        "DeterministicTrack",
        "StochasticTrack",
        "MODeterministicTrack",
        "MOStochasticTrack",
    ):
        env_id = f"{name}-v1"
        entry_point = f"gym_longicontrol.envs:{name}"
        if env_id not in registry:
            register(
                id=env_id,
                entry_point=entry_point,
                max_episode_steps=1800,
                # Gymnasium's scalar-reward checker is not an MO checker.
                disable_env_checker=name.startswith("MO"),
            )
        elif registry[env_id].entry_point != entry_point:
            raise ValueError(
                f"Environment ID {env_id} is already owned by another package"
            )
