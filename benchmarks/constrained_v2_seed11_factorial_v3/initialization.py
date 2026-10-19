"""Byte-stable learner initialization verification for all four treatments."""

from __future__ import annotations

from typing import Any

from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_v2_repro_audit.harness import _module_sha256
from benchmarks.constrained_v2_seed11_factorial.stream_preflight import (
    stable_state_sha256,
)

from .config import StudyConfiguration
from .streams import build_training_environment

EXPECTED = {
    "actor_sha256": "72e9207137ecd2f8e1c6506915f1348eb59f47133a3b181cbf09098888e928c5",
    "combined_learner_sha256": (
        "beffa5734886fc714e4eeca5ddae17fa51e12af9839a2999af7743dc5287d87a"
    ),
    "reward_critic_sha256": (
        "b3b6f1dbeaa139a78715d7aa3671036046da9b32e222ee9c9c13bddbf2511649"
    ),
    "speed_cost_critic_sha256": (
        "0586555124d0e8d553f3fd59be4776a74380906ebe0fa3aa88ab443a5dd07353"
    ),
    "deadline_cost_critic_sha256": (
        "844a8d1d8c272f9270030f2c9fcf06a575d6d36708fa89d8428eb47ce864eb4d"
    ),
}


def policy_initial_hashes(policy: Any) -> dict[str, Any]:
    critics = [_module_sha256(item) for item in policy.critics]
    target_critics = [_module_sha256(item) for item in policy.critics_old]
    row = {
        "actor_sha256": _module_sha256(policy.actor),
        "reward_critic_sha256": critics[0],
        "speed_cost_critic_sha256": critics[1],
        "deadline_cost_critic_sha256": critics[2],
        "target_reward_critic_sha256": target_critics[0],
        "target_speed_cost_critic_sha256": target_critics[1],
        "target_deadline_cost_critic_sha256": target_critics[2],
        "actor_optimizer_sha256": stable_state_sha256(
            policy.actor_optim.state_dict()
        ),
        "critic_optimizer_sha256": stable_state_sha256(
            policy.critics_optim.state_dict()
        ),
        "alpha": {
            "effective_alpha": float(policy._alpha.detach().cpu()),  # noqa: SLF001
            "log_alpha_sha256": stable_state_sha256(
                policy._log_alpha  # noqa: SLF001
            ),
            "target_entropy": float(policy._target_entropy),  # noqa: SLF001
            "optimizer_sha256": stable_state_sha256(
                policy._alpha_optim.state_dict()  # noqa: SLF001
            ),
        },
        "pid_state_sha256": stable_state_sha256(
            [item.state_dict() for item in policy.lag_optims]
        ),
        "pid_multipliers": [float(item.get_lag()) for item in policy.lag_optims],
    }
    row["combined_learner_sha256"] = stable_state_sha256(row)
    return row


def verify_initialization(
    configuration: StudyConfiguration, *, device: str = "cpu", threads: int = 4
) -> dict[str, Any]:
    rows = {}
    for condition in configuration.conditions:
        environment = build_training_environment(
            configuration,
            stream_name=condition.track_stream,
            training_seed=configuration.raw["training_seed"],
            max_simulator_steps=configuration.raw["real_transition_budget"],
        )
        try:
            agent, _logger = build_agent(
                configuration.v2,
                environment,
                training_seed=configuration.raw["training_seed"],
                device=device,
                threads=threads,
            )
            rows[condition.name] = policy_initial_hashes(agent.policy)
        finally:
            environment.close()
    values = list(rows.values())
    identical = all(row == values[0] for row in values[1:])
    expected = all(
        all(row[key] == value for key, value in EXPECTED.items())
        for row in values
    )
    alpha_pid = all(
        row["alpha"]["effective_alpha"] == 1.0
        and row["pid_multipliers"] == [0.0, 0.0]
        for row in values
    )
    return {
        "by_condition": rows,
        "all_identical": identical,
        "all_expected_primary_hashes": expected,
        "alpha_and_pid_expected": alpha_pid,
        "passed": identical and expected and alpha_pid,
    }
