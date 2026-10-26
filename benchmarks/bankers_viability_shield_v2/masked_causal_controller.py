"""Fixed finite-grid projection; contains no learner or adaptation."""

from .inherited_shield_adapter import ACTION_GRID, SafetyIntegrityError


def closest_certified(mask, proposal):
    if len(mask) != 21 or not any(mask) or not mask[0]:
        raise SafetyIntegrityError("EMPTY_MASK_OR_BACKUP_UNAVAILABLE")
    return min(
        (i for i, valid in enumerate(mask) if valid),
        key=lambda i: (abs(ACTION_GRID[i] - proposal), ACTION_GRID[i]),
    )
