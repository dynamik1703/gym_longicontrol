"""One full grid per decision, never a stale-state mask cache."""

from benchmarks.bankers_viability_shield_v1.backup_policy import SafetyIntegrityError
from benchmarks.bankers_viability_shield_v1.certificate import digest
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID


def safe_action_mask(shield, ordinary, memory):
    identity = digest(
        {"observation": [float(x) for x in ordinary], "memory": memory.snapshot()}
    )
    return [
        shield.certify(ordinary, memory, a, input_hash=identity) for a in ACTION_GRID
    ]


def closest_certified(mask, proposal):
    if len(mask) != 21 or not any(mask):
        raise SafetyIntegrityError("empty or malformed mask")
    return min(
        (i for i, valid in enumerate(mask) if valid),
        key=lambda i: (abs(ACTION_GRID[i] - proposal), ACTION_GRID[i]),
    )


def policy_input(ordinary, memory, certificates):
    """Future interface only. No seed, hidden route, learner, or training loop."""
    return dict(
        observation=list(ordinary),
        physical_memory=memory.snapshot(),
        remaining_distance_m=max(0.0, 1000 - memory.state.position_m),
        remaining_time_s=140 - memory.state.elapsed_time_s,
        mask=[c["accepted"] for c in certificates],
        causal_identity=certificates[0]["input_sha256"],
    )


def masked_log_probabilities(logits, mask):
    import numpy as np

    if len(logits) != 21 or len(mask) != 21 or not any(mask):
        raise SafetyIntegrityError("invalid masked distribution")
    result = np.full(21, -np.inf)
    valid = np.asarray(mask, dtype=bool)
    values = np.asarray(logits, dtype=np.float64)[valid]
    if not np.isfinite(values).all():
        raise SafetyIntegrityError("nonfinite valid logits")
    shifted = values - max(values)
    result[valid] = shifted - np.log(np.exp(shifted).sum())
    probabilities = np.zeros(21)
    probabilities[valid] = np.exp(result[valid])
    positive = probabilities > 0
    entropy = -float(np.sum(probabilities[positive] * result[positive]))
    return probabilities, result, entropy
