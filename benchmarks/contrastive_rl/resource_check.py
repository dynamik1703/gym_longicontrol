"""Bounded synthetic/resource probe; this module cannot run policy training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import psutil

from .config import ReferenceCoreConfig
from .learner import ContrastiveBatch, ReferenceLearner, tree_parameter_count


def _wait(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves:
        leaves[0].block_until_ready()


def _synthetic_batch(config: ReferenceCoreConfig, batch_size: int):
    rng = np.random.default_rng(20261007)
    return ContrastiveBatch(
        states=jnp.asarray(rng.normal(size=(batch_size, config.state_dim))),
        actions=jnp.asarray(rng.uniform(-1, 1, size=(batch_size, config.action_dim))),
        critic_goals=jnp.asarray(rng.normal(size=(batch_size, config.goal_dim))),
        actor_goals=jnp.asarray(rng.normal(size=(batch_size, config.goal_dim))),
        historical_reward=jnp.asarray(rng.normal(size=batch_size)),
    )


def profile_depth(depth: int, *, batch_size: int, iterations: int) -> dict:
    if iterations > 100:
        raise ValueError("preparation limit is 100 synthetic updates per depth")
    config = ReferenceCoreConfig(depth=depth)
    rss_before = psutil.Process().memory_info().rss
    learner, state = ReferenceLearner.create(config, seed=123)
    batch = _synthetic_batch(config, batch_size)

    start = time.perf_counter()
    batch = jax.device_put(batch)
    _wait(batch)
    transfer_seconds = time.perf_counter() - start

    action_fn = jax.jit(learner.deterministic_action)
    states = batch.states
    goals = batch.actor_goals
    start = time.perf_counter()
    actions = action_fn(state.actor.params, states, goals)
    actions.block_until_ready()
    actor_compile_seconds = time.perf_counter() - start
    actor_iterations = 100
    start = time.perf_counter()
    for _ in range(actor_iterations):
        actions = action_fn(state.actor.params, states, goals)
    actions.block_until_ready()
    actor_steady_seconds = time.perf_counter() - start

    update_fn = jax.jit(learner.update)
    start = time.perf_counter()
    state, metrics = update_fn(state, batch, jax.random.PRNGKey(99))
    _wait(metrics)
    update_compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for index in range(iterations):
        state, metrics = update_fn(
            state, batch, jax.random.fold_in(jax.random.PRNGKey(100), index)
        )
    _wait(metrics)
    update_steady_seconds = time.perf_counter() - start
    rss_after = psutil.Process().memory_info().rss

    return {
        "depth_inside_residual_blocks": depth,
        "residual_blocks_per_network": depth // 4,
        "input_projection_dense_layers_per_network": 1,
        "actor_output_dense_layers": 2,
        "critic_output_dense_layers_per_encoder": 1,
        "batch_size": batch_size,
        "synthetic_update_iterations_steady": iterations,
        "synthetic_update_iterations_total_including_compile": iterations + 1,
        "parameters": {
            "actor": tree_parameter_count(state.actor.params),
            "state_action_encoder": tree_parameter_count(
                state.critic.params["state_action"]
            ),
            "goal_encoder": tree_parameter_count(state.critic.params["goal"]),
            "total_trainable": (
                tree_parameter_count(state.actor.params)
                + tree_parameter_count(state.critic.params)
                + tree_parameter_count(state.alpha.params)
            ),
        },
        "timing_seconds": {
            "host_to_device_batch": transfer_seconds,
            "actor_compile_and_first_call": actor_compile_seconds,
            "actor_steady_total": actor_steady_seconds,
            "actor_steady_per_batch": actor_steady_seconds / actor_iterations,
            "update_compile_and_first_call": update_compile_seconds,
            "update_steady_total": update_steady_seconds,
            "update_steady_per_iteration": update_steady_seconds / iterations,
        },
        "resident_memory_bytes": {
            "before": rss_before,
            "after": rss_after,
            "observed_increase": max(0, rss_after - rss_before),
        },
        "finite_final_metrics": bool(
            all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
        ),
    }


def profile_simulator(steps: int) -> dict:
    if steps < 0 or steps > 2000:
        raise ValueError("preparation limit is 2,000 simulator transitions")
    if not steps:
        return {"transitions": 0, "reason": "disabled"}
    import gymnasium as gym

    import gym_longicontrol  # noqa: F401

    env = gym.make("StochasticTrack-v1")
    observation, _ = env.reset(seed=2000)
    start = time.perf_counter()
    resets = 0
    for _ in range(steps):
        observation, _, terminated, truncated, _ = env.step(np.array([0.0]))
        if terminated or truncated:
            resets += 1
            observation, _ = env.reset(seed=2000 + resets)
    elapsed = time.perf_counter() - start
    env.close()
    return {
        "transitions": steps,
        "development_seed_start": 2000,
        "fixed_action": 0.0,
        "resets": resets,
        "elapsed_seconds": elapsed,
        "transitions_per_second": steps / elapsed,
        "final_observation_finite": bool(np.isfinite(observation).all()),
        "used_for_learning": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--simulator-steps", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")

    report = {
        "kind": "preparation_only_synthetic_resource_check",
        "generated_at": "2026-10-07",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpu_count": psutil.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
        },
        "backend": {
            "jax": importlib.metadata.version("jax"),
            "jaxlib": importlib.metadata.version("jaxlib"),
            "flax": importlib.metadata.version("flax"),
            "optax": importlib.metadata.version("optax"),
            "devices": [str(device) for device in jax.devices()],
            "external_tracking": False,
        },
        "depths": [
            profile_depth(
                depth, batch_size=args.batch_size, iterations=args.iterations
            )
            for depth in (4, 16)
        ],
        "simulator": profile_simulator(args.simulator_steps),
        "training_performed": False,
        "validation_tracks_used": False,
        "paper_tracks_used": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
