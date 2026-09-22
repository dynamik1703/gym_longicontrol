"""Evaluate maximization objectives on identical tracks, without training."""

import gymnasium as gym
import numpy as np

from gym_longicontrol.envs.multi_objective import REWARD_NAMES

ENV_IDS = ("MOStochasticTrack-v1", "MODeterministicTrack-v1")
# Interior preferences also suit CAPQL's upstream near-uniform weight sampler.
DEFAULT_WEIGHTS = (
    (0.25, 0.25, 0.25, 0.25),
    (0.4, 0.2, 0.2, 0.2),
    (0.2, 0.4, 0.2, 0.2),
    (0.2, 0.2, 0.4, 0.2),
    (0.2, 0.2, 0.2, 0.4),
)
# Fixed before training, in undiscounted episode-return units. Never fitted to
# an observed front. Scaling affects reporting only, not training rewards.
DEFAULT_REFERENCE = (-12000.0, -2000.0, -2000.0, -2000.0)
DEFAULT_SCALE = (1800.0, 1800.0, 1800.0, 1800.0)


def validate_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.ndim != 2
        or weights.shape[0] == 0
        or weights.shape[1] != len(REWARD_NAMES)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-8)
    ):
        raise ValueError(
            "Each preference must have four finite nonnegative weights summing to 1"
        )
    if len(np.unique(weights, axis=0)) != len(weights):
        raise ValueError("Preference vectors must be distinct")
    return weights.copy()


def make_env(env_id=ENV_IDS[0], *, max_episode_steps=1800):
    if env_id not in ENV_IDS:
        raise ValueError(f"Unknown MO environment: {env_id}")
    if not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be a positive integer")
    return gym.make(f"gym_longicontrol:{env_id}", max_episode_steps=max_episode_steps)


def nondominated(points):
    """Unique nondominated points; all signed objectives are maximized."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Expected a finite matrix of return vectors")
    points = np.unique(points, axis=0)
    return (
        points[
            [
                not np.any(
                    np.all(points >= point, axis=1) & np.any(points > point, axis=1)
                )
                for point in points
            ]
        ]
        if len(points)
        else points
    )


def front_metrics(points, *, reference=DEFAULT_REFERENCE, scale=DEFAULT_SCALE):
    """Hypervolume of expected returns, not a union across training seeds."""
    from morl_baselines.common.performance_indicators import hypervolume

    points = np.asarray(points, dtype=np.float64)
    reference, scale = np.asarray(reference), np.asarray(scale)
    if (
        points.ndim != 2
        or points.shape[1] != len(REWARD_NAMES)
        or not np.isfinite(points).all()
        or reference.shape != (len(REWARD_NAMES),)
        or scale.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "Returns, reference and positive scales must have four finite objectives"
        )
    if np.any(points < reference):
        raise ValueError(
            "A return is below the fixed hypervolume reference. Choose and document "
            "a common worse reference before rerunning all compared methods."
        )
    front = nondominated(points)
    hv = hypervolume(reference / scale, front / scale) if len(front) else 0.0
    return {
        "pareto_front": front.tolist(),
        "hypervolume": float(hv),
        "reference_point": reference.tolist(),
        "objective_scale": scale.tolist(),
        "return_type": "undiscounted; per-policy mean across evaluation tracks",
    }


def rollout(policy, *, env_id=ENV_IDS[0], seed=1001, max_episode_steps=1800):
    """Run a deterministic policy to termination or the registered time limit."""
    env = make_env(env_id, max_episode_steps=max_episode_steps)
    try:
        observation, _ = env.reset(seed=seed)
        total = np.zeros(len(REWARD_NAMES))
        jerk_sum = 0.0
        overspeed_steps = steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total += reward
            jerk_sum += abs(info["jerk_m_s3"])
            overspeed_steps += info["velocity_m_s"] > info["speed_limit_m_s"]
            steps += 1
        return {
            "seed": int(seed),
            "return_vector": total.tolist(),
            "completed": bool(terminated),
            "truncated": bool(truncated),
            "steps": steps,
            "distance_m": float(info["position_m"]),
            "elapsed_time_s": float(info["elapsed_time_s"]),
            "energy_kwh": float(info["total_energy_kwh"]),
            "mean_abs_jerk_m_s3": float(jerk_sum / steps),
            "overspeed_time_s": float(overspeed_steps * env.unwrapped.config.dt_s),
        }
    finally:
        env.close()


def evaluate(
    policies,
    *,
    weights=DEFAULT_WEIGHTS,
    seeds=(1001, 1002, 1003),
    env_id=ENV_IDS[0],
    max_episode_steps=1800,
    reference=DEFAULT_REFERENCE,
    scale=DEFAULT_SCALE,
):
    weights = validate_weights(weights)
    seeds = tuple(seeds)
    if not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("At least one nonnegative integer evaluation seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be distinct")
    if len(policies) != len(weights):
        raise ValueError("One policy is required for each preference")
    results = []
    for policy, weight in zip(policies, weights):
        episodes = [
            rollout(
                policy, env_id=env_id, seed=seed, max_episode_steps=max_episode_steps
            )
            for seed in seeds
        ]
        mean_return = np.mean(
            [episode["return_vector"] for episode in episodes], axis=0
        )
        results.append(
            {
                "weight": weight.tolist(),
                "episodes": episodes,
                "mean_return": mean_return.tolist(),
                "mean_scalarized_return": float(weight @ mean_return),
                "completion_rate": float(np.mean([e["completed"] for e in episodes])),
                "mean_overspeed_time_s": float(
                    np.mean([e["overspeed_time_s"] for e in episodes])
                ),
            }
        )
    return {
        "policies": results,
        **front_metrics(
            [row["mean_return"] for row in results], reference=reference, scale=scale
        ),
    }
