"""Historical Constrained-to-new-Scalar initialization pairing."""

from __future__ import annotations

from typing import Any

from benchmarks.constrained_rl_v2.adapter import build_agent as build_constrained_agent
from benchmarks.constrained_v2_repro_audit.harness import _module_sha256
from benchmarks.scalar_sac.experiment import _base_environment

from .agent import build_scalar_agent, scalar_policy_invariants
from .config import TRAINING_SEEDS, StudyConfiguration


def verify_initialization(
    configuration: StudyConfiguration, *, device: str = "cpu", threads: int = 4
) -> dict[str, Any]:
    rows = {}
    v2 = configuration.constrained_v2
    for seed in TRAINING_SEEDS:
        environment = _base_environment(v2)
        try:
            constrained, _ = build_constrained_agent(
                v2,
                environment,
                training_seed=seed,
                device=device,
                threads=threads,
            )
            scalar, _ = build_scalar_agent(
                environment,
                training_seed=seed,
                device=device,
                threads=threads,
            )
            actor_c = _module_sha256(constrained.policy.actor)
            actor_s = _module_sha256(scalar.policy.actor)
            critic_c = _module_sha256(constrained.policy.critics[0])
            critic_s = _module_sha256(scalar.policy.critics[0])
            expected = configuration.raw["reference_initialization"][str(seed)]
            passed = (
                actor_c == actor_s == expected["actor_sha256"]
                and critic_c == critic_s == expected["reward_critic_sha256"]
            )
            rows[str(seed)] = {
                "actor_constrained_sha256": actor_c,
                "actor_scalar_sha256": actor_s,
                "reward_critic_constrained_sha256": critic_c,
                "reward_critic_scalar_sha256": critic_s,
                "byte_identical": passed,
                "scalar_invariants": scalar_policy_invariants(scalar.policy),
            }
        finally:
            environment.close()
    return {
        "strategy": (
            "Pinned historical three-critic construction is reproduced because FSRL "
            "allocates all networks before orthogonal initialization. The actor and "
            "first double-Q reward critic are retained byte-for-byte; two temporary "
            "cost critics are discarded before the Scalar policy is constructed."
        ),
        "by_seed": rows,
        "passed": all(row["byte_identical"] for row in rows.values()),
    }
