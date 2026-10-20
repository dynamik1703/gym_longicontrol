"""Deterministic actor-only adapters with Validation and Paper fail-closed."""

from __future__ import annotations

from typing import Any

from benchmarks.constrained_v2_collector_replication.evaluation import (
    evaluate_actor_on_seeds,
)

from .config import forbid_paper, require_validation_authorized
from .split import diagnostic_seeds, validation_seeds


class _ReferenceConfigurationView:
    def __init__(self, configuration):
        self.v2 = configuration.constrained_v2
        self.raw = configuration.raw


def evaluate_diagnostic(policy: Any, configuration: Any, *, real_transitions: int):
    return evaluate_actor_on_seeds(
        policy,
        _ReferenceConfigurationView(configuration),
        evaluation_seeds=diagnostic_seeds(),
        real_transitions=real_transitions,
        evaluation_id="SAMEINFO_DIAGNOSTIC_V1",
    )


def evaluate_validation(policy: Any, configuration: Any):
    require_validation_authorized(configuration)
    return evaluate_actor_on_seeds(
        policy,
        _ReferenceConfigurationView(configuration),
        evaluation_seeds=validation_seeds(),
        real_transitions=configuration.raw["real_transition_budget"],
        evaluation_id="SAMEINFO_VALIDATION_V1",
    )


evaluate_paper = forbid_paper
