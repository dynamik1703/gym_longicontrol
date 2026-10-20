from __future__ import annotations

import numpy as np
import torch

from benchmarks.hybrid_rl_design.preflight import (
    fallback_parity_preflight,
    shield_preflight,
)
from benchmarks.hybrid_rl_execution.agent import ConstrainedSAC
from benchmarks.hybrid_rl_execution.safety import run_safety_audit
from benchmarks.hybrid_rl_execution.verification import causal_information_boundary


def _batch(size: int = 8):
    rng = np.random.default_rng(20261020)
    return {
        "observation": rng.normal(size=(size, 16)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(size, 1)).astype(np.float32),
        "reward": rng.normal(size=(size, 1)).astype(np.float32),
        "speed_cost": rng.uniform(0, 0.1, size=(size, 1)).astype(np.float32),
        "terminal_cost": np.zeros((size, 1), dtype=np.float32),
        "intervention_cost": rng.uniform(0, 0.5, size=(size, 1)).astype(np.float32),
        "next_observation": rng.normal(size=(size, 16)).astype(np.float32),
        "continuation_discount": np.full((size, 1), 0.99, dtype=np.float32),
    }


def test_actual_actor_critics_entropy_and_multiplier_update():
    agent = ConstrainedSAC(11)
    before = {name: tensor.clone() for name, tensor in agent.actor.state_dict().items()}
    diagnostics = agent.update(_batch(), generator=torch.Generator().manual_seed(123))
    assert agent.update_count == 1
    assert any(
        not torch.equal(before[name], tensor)
        for name, tensor in agent.actor.state_dict().items()
    )
    assert diagnostics["alpha"] > 0.0
    assert diagnostics["speed_multiplier"] >= 0.0
    assert diagnostics["terminal_multiplier"] >= 0.0


def test_frozen_fallback_exact_parity_and_shield_influence_reproduced():
    parity = fallback_parity_preflight()
    shield = shield_preflight()
    assert parity["all_exact"]
    assert shield["endpoint_safety_passed"]
    assert shield["deadline_fixture_passed"]
    assert not shield["viability_falsifications"]
    assert not shield["SHIELD_DOMINATES_POLICY"]
    assert abs(shield["override_fraction"] - 0.17597581811013274) < 1e-15
    assert abs(shield["influence_fraction"] - 0.6338546458141674) < 1e-15
    assert shield["influenced_fixture_count"] == 6


def test_extended_adversarial_viability_and_information_boundary():
    safety = run_safety_audit()
    assert safety["exact_right_endpoint_speed_passed"]
    assert safety["eventual_completion_passed"]
    assert safety["deadline_passed"]
    assert not safety["viability_falsifications"]
    assert safety["passed"]
    assert causal_information_boundary()["passed"]
