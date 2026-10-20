"""Frozen descriptive interpretation gates (not significance tests)."""

from __future__ import annotations


def classify(
    paired_seed_differences: tuple[int, int, int], pooled_difference: int
) -> str:
    """Classify Constrained-minus-Scalar RSR counts."""

    deltas = tuple(int(value) for value in paired_seed_differences)
    pooled = int(pooled_difference)
    constrained = (
        pooled >= 6 and min(deltas) >= -2 and sum(value >= 2 for value in deltas) >= 2
    )
    scalar = (
        pooled <= -6 and max(deltas) <= 2 and sum(value <= -2 for value in deltas) >= 2
    )
    if constrained:
        return "CONSTRAINED_ROBUST_ADVANTAGE"
    if scalar:
        return "SCALAR_ROBUST_ADVANTAGE"
    severe_contradiction = max(deltas) >= 3 and min(deltas) <= -3
    if abs(pooled) < 6 and not severe_contradiction:
        return "NO_MATERIAL_DIFFERENCE"
    return "SEED_HETEROGENEOUS"
