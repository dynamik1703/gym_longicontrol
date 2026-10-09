"""Isolated external evaluation with explicit split gates and no model planning."""

from __future__ import annotations

from typing import Any

from benchmarks.constrained_rl.adapter import deterministic_policy_adapter
from benchmarks.scalar_sac.evaluation import evaluate_policy
from benchmarks.scalar_sac.experiment import _base_environment

from .config import MBRLConfiguration
from .execution import assert_paper_test_blocked, assert_validation_authorized


def evaluate_development(policy: Any, configuration: MBRLConfiguration):
    environment = _base_environment(configuration.v2)
    policy.eval()
    try:
        return evaluate_policy(
            deterministic_policy_adapter(policy),
            environment,
            task=configuration.v2.task,
            evaluation_seeds=configuration.raw["track_splits"]["development"],
        )
    finally:
        environment.close()
        policy.train()


def evaluate_validation(policy: Any, configuration: MBRLConfiguration, manifest):
    assert_validation_authorized(manifest)
    environment = _base_environment(configuration.v2)
    policy.eval()
    try:
        return evaluate_policy(
            deterministic_policy_adapter(policy),
            environment,
            task=configuration.v2.task,
            evaluation_seeds=configuration.raw["track_splits"]["validation"],
        )
    finally:
        environment.close()
        policy.train()


def evaluate_paper_test(*_args, **_kwargs):
    assert_paper_test_blocked()
