"""Seed-paired byte-stable learner and replay initialization checks."""

from __future__ import annotations

import hashlib
import pickle
from typing import Any

from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_v2_repro_audit.harness import _module_sha256
from benchmarks.constrained_v2_seed11_factorial.stream_preflight import (
    stable_state_sha256,
)

from .config import TRAINING_SEEDS, StudyConfiguration
from .streams import build_training_environment


def empty_replay_sha256(capacity: int) -> str:
    from tianshou.data import ReplayBuffer

    payload = pickle.dumps(ReplayBuffer(capacity), protocol=5)
    return hashlib.sha256(payload).hexdigest()


def policy_initial_hashes(policy: Any, *, replay_capacity: int) -> dict[str, Any]:
    critics = [_module_sha256(item) for item in policy.critics]
    targets = [_module_sha256(item) for item in policy.critics_old]
    row = {
        "actor_sha256": _module_sha256(policy.actor),
        "reward_critic_sha256": critics[0],
        "speed_cost_critic_sha256": critics[1],
        "deadline_cost_critic_sha256": critics[2],
        "target_reward_critic_sha256": targets[0],
        "target_speed_cost_critic_sha256": targets[1],
        "target_deadline_cost_critic_sha256": targets[2],
        "actor_optimizer_sha256": stable_state_sha256(policy.actor_optim.state_dict()),
        "critic_optimizer_sha256": stable_state_sha256(
            policy.critics_optim.state_dict()
        ),
        "alpha": {
            "effective_alpha": float(policy._alpha.detach().cpu()),  # noqa: SLF001
            "log_alpha_sha256": stable_state_sha256(policy._log_alpha),  # noqa: SLF001
            "target_entropy": float(policy._target_entropy),  # noqa: SLF001
            "optimizer_sha256": stable_state_sha256(policy._alpha_optim.state_dict()),  # noqa: SLF001
        },
        "pid_state_sha256": stable_state_sha256(
            [item.state_dict() for item in policy.lag_optims]
        ),
        "pid_multipliers": [float(item.get_lag()) for item in policy.lag_optims],
        "empty_replay_sha256": empty_replay_sha256(replay_capacity),
    }
    row["combined_learner_and_replay_sha256"] = stable_state_sha256(row)
    return row


def verify_initialization(
    configuration: StudyConfiguration, *, device: str = "cpu", threads: int = 4
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for condition in configuration.conditions:
        environment = build_training_environment(
            configuration,
            training_seed=condition.training_seed,
            max_simulator_steps=configuration.raw["real_transition_budget"],
        )
        try:
            agent, _logger = build_agent(
                configuration.v2,
                environment,
                training_seed=condition.training_seed,
                device=device,
                threads=threads,
            )
            rows[condition.name] = policy_initial_hashes(
                agent.policy,
                replay_capacity=configuration.v2.algorithm.buffer_size,
            )
        finally:
            environment.close()
    by_seed: dict[str, Any] = {}
    pair_pass = True
    for seed in TRAINING_SEEDS:
        pair = [
            condition
            for condition in configuration.conditions
            if condition.training_seed == seed
        ]
        left, right = (rows[condition.name] for condition in pair)
        identical = left == right
        pair_pass = pair_pass and identical
        by_seed[str(seed)] = {
            "episodic_condition": pair[0].name,
            "transitional_condition": pair[1].name,
            "byte_hashes_identical": identical,
            "combined_learner_and_replay_sha256": left[
                "combined_learner_and_replay_sha256"
            ]
            if identical
            else None,
        }
    distinct_across_seeds = len(
        {
            by_seed[str(seed)]["combined_learner_and_replay_sha256"]
            for seed in TRAINING_SEEDS
        }
    ) == len(TRAINING_SEEDS)
    alpha_pid = all(
        row["alpha"]["effective_alpha"] == 1.0 and row["pid_multipliers"] == [0.0, 0.0]
        for row in rows.values()
    )
    return {
        "by_condition": rows,
        "by_seed_pair": by_seed,
        "all_pairs_byte_identical": pair_pass,
        "seeds_initialize_differently": distinct_across_seeds,
        "alpha_and_pid_expected": alpha_pid,
        "passed": pair_pass and distinct_across_seeds and alpha_pid,
    }
