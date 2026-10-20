"""Bounded synthetic and fixture-only Hybrid resource benchmark."""

from __future__ import annotations

import resource
import time
from typing import Any

import gymnasium as gym
import torch
from torch import nn

from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from .baseline import TimeAwareEnergyStrategy
from .controller import STRATEGY_OBSERVATION_DIMENSIONS, HybridTargetSpeedController

SYNTHETIC_UPDATES = 100
REPLAY_CAPACITY = 300_000
UPDATES_PER_POLICY = 30_000
TRANSITIONS_PER_POLICY = 300_000
POLICY_COUNT = 3


class _MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def _synthetic_update_benchmark() -> dict[str, Any]:
    torch.manual_seed(20261020)
    torch.set_num_threads(1)
    batch = 256
    observation = torch.randn(batch, STRATEGY_OBSERVATION_DIMENSIONS)
    next_observation = torch.randn(batch, STRATEGY_OBSERVATION_DIMENSIONS)
    action = torch.tanh(torch.randn(batch, 1))
    reward = torch.randn(batch, 1)
    actor = _MLP(STRATEGY_OBSERVATION_DIMENSIONS, 2)
    critic_1 = _MLP(STRATEGY_OBSERVATION_DIMENSIONS + 1, 1)
    critic_2 = _MLP(STRATEGY_OBSERVATION_DIMENSIONS + 1, 1)
    parameters = [
        *actor.parameters(),
        *critic_1.parameters(),
        *critic_2.parameters(),
    ]
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    started = time.perf_counter()
    for _ in range(SYNTHETIC_UPDATES):
        optimizer.zero_grad(set_to_none=True)
        actor_output = actor(observation)
        proposed = torch.tanh(actor_output[:, :1])
        critic_input = torch.cat((observation, action), dim=1)
        proposed_input = torch.cat((next_observation, proposed), dim=1)
        loss = (
            critic_1(critic_input).square().mean()
            + critic_2(critic_input).square().mean()
            - critic_1(proposed_input).mean()
            - critic_2(proposed_input).mean()
            - 0.01 * reward.mean()
        )
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    parameter_count = sum(parameter.numel() for parameter in parameters)
    return {
        "scope": "synthetic tensors only; no environment data and no retained model",
        "updates": SYNTHETIC_UPDATES,
        "batch_size": batch,
        "wall_time_s": elapsed,
        "mean_update_time_s": elapsed / SYNTHETIC_UPDATES,
        "parameter_count_actor_plus_two_critics": parameter_count,
    }


def _shield_benchmark() -> dict[str, Any]:
    track = physical_fixtures()["consecutive_downward_changes"]
    environment = gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=lambda _rng: track),
        max_episode_steps=1_800,
    )
    controller = HybridTargetSpeedController(strategy_enabled=True)
    strategy = TimeAwareEnergyStrategy()
    actions = 0
    prediction_calls = 0
    action_time_s = 0.0
    full_started = time.perf_counter()
    try:
        observation, _ = environment.reset(seed=123)
        controller.reset()
        strategy.reset()
        ended = False
        while not ended:
            proposal = strategy.action(
                controller.state,
                actions,
                route_length_m=controller.config.track_length_m,
            )
            started = time.perf_counter()
            action = controller.action(observation, proposal)
            action_time_s += time.perf_counter() - started
            assert controller.last_decision is not None
            prediction_calls += controller.last_decision.physics_prediction_calls
            observation, _, terminated, truncated, _ = environment.step([action])
            controller.observe_executed_action(action)
            actions += 1
            ended = bool(terminated or truncated)
    finally:
        environment.close()
    wall = time.perf_counter() - full_started
    return {
        "fixture": "consecutive_downward_changes",
        "actions": actions,
        "rollout_wall_time_s": wall,
        "mean_full_environment_step_time_s": wall / actions,
        "shield_action_time_s": action_time_s,
        "mean_shield_time_per_action_s": action_time_s / actions,
        "physics_prediction_calls": prediction_calls,
        "mean_physics_prediction_calls_per_action": prediction_calls / actions,
    }


def benchmark_resources() -> dict[str, Any]:
    update = _synthetic_update_benchmark()
    shield = _shield_benchmark()
    collection_s = (
        shield["mean_full_environment_step_time_s"]
        * TRANSITIONS_PER_POLICY
        * POLICY_COUNT
    )
    update_s = update["mean_update_time_s"] * UPDATES_PER_POLICY * POLICY_COUNT
    diagnostic_episodes = 18 * 6 * POLICY_COUNT
    final_episodes = 36 * (POLICY_COUNT + 2)
    assumed_steps_per_evaluation_episode = 1_200
    evaluation_s = (
        (diagnostic_episodes + final_episodes)
        * assumed_steps_per_evaluation_episode
        * shield["mean_full_environment_step_time_s"]
    )
    raw_projection_s = collection_s + update_s + evaluation_s
    projected_s = raw_projection_s * 3.0
    float_fields = STRATEGY_OBSERVATION_DIMENSIONS * 2 + 6
    replay_bytes_per_policy = REPLAY_CAPACITY * float_fields * 4
    retained_replay_bytes = replay_bytes_per_policy * POLICY_COUNT
    checkpoints_bytes = (
        update["parameter_count_actor_plus_two_critics"] * 4 * 7 * POLICY_COUNT * 4
    )
    retained_overhead_bytes = 2_000_000_000
    projected_disk_bytes = (
        retained_replay_bytes + checkpoints_bytes + retained_overhead_bytes
    )
    return {
        "schema_version": 1,
        "bounded_preparation_only": True,
        "model_free_update_benchmark": update,
        "causal_shield_benchmark": shield,
        "future_study": {
            "policy_count": POLICY_COUNT,
            "real_transitions_per_policy": TRANSITIONS_PER_POLICY,
            "gradient_updates_per_policy": UPDATES_PER_POLICY,
            "diagnostic_episode_count": diagnostic_episodes,
            "final_evaluation_episode_count": final_episodes,
            "collection_projection_s": collection_s,
            "update_projection_s": update_s,
            "evaluation_projection_s": evaluation_s,
            "uncertainty_factor": 3.0,
            "projected_runtime_s": projected_s,
            "projected_runtime_hours": projected_s / 3600.0,
            "projected_retained_replay_bytes": retained_replay_bytes,
            "projected_checkpoint_bytes": checkpoints_bytes,
            "projected_other_artifact_bytes": retained_overhead_bytes,
            "projected_retained_disk_bytes": projected_disk_bytes,
            "projected_retained_disk_gb": projected_disk_bytes / 1e9,
        },
        "limits": {"runtime_hours": 24.0, "retained_disk_gb": 12.0},
        "runtime_gate_passed": projected_s <= 24.0 * 3600.0,
        "disk_gate_passed": projected_disk_bytes <= 12_000_000_000,
        "maximum_rss_platform_units": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "no_rl_training_occurred": True,
    }
