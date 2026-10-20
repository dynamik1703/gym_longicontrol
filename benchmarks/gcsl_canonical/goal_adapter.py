"""Frozen projected-goal semantics reused from the audited CRL implementation."""

from benchmarks.contrastive_rl.goal_adapter import (
    CANONICAL_MAX_SPEED_VIOLATION_M_S,
    CANONICAL_MAX_TIME_S,
    CANONICAL_ROUTE_LENGTH_M,
    OUTCOME_DIM,
    PROJECTED_GOAL_DIM,
    OutcomeScales,
    augment_policy_state,
    canonical_command,
    canonical_goal_set_membership,
    normalize_outcome,
    physical_outcome,
    project_outcome,
    projected_goal_is_canonical,
)

__all__ = [
    "CANONICAL_MAX_SPEED_VIOLATION_M_S",
    "CANONICAL_MAX_TIME_S",
    "CANONICAL_ROUTE_LENGTH_M",
    "OUTCOME_DIM",
    "PROJECTED_GOAL_DIM",
    "OutcomeScales",
    "augment_policy_state",
    "canonical_command",
    "canonical_goal_set_membership",
    "normalize_outcome",
    "physical_outcome",
    "project_outcome",
    "projected_goal_is_canonical",
]
