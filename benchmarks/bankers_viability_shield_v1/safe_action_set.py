"""The fixed V1 menu, with every admitted point individually certified."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backup_policy import SafetyIntegrityError
from .certificate import BankersShield, SpeedCertificate, digest

ACTION_GRID = tuple(i / 10 for i in range(-10, 11))


@dataclass(frozen=True)
class SafeActionSet:
    certificates: tuple[SpeedCertificate, ...]

    @property
    def mask(self):
        return tuple(c.accepted for c in self.certificates)

    @property
    def actions(self):
        return tuple(a for a, valid in zip(ACTION_GRID, self.mask) if valid)

    @property
    def sha256(self):
        return digest([c.sha256 for c in self.certificates])

    @property
    def physics_prediction_count(self):
        return sum(c.physics_prediction_count for c in self.certificates)

    @property
    def distinct_successors(self):
        return len(
            {
                (
                    c.predicted_next_state["position_m"],
                    c.predicted_next_state["velocity_m_s"],
                )
                for c in self.certificates
                if c.accepted
            }
        )

    @property
    def velocity_span_m_s(self):
        values = [
            c.predicted_next_state["velocity_m_s"]
            for c in self.certificates
            if c.accepted
        ]
        return max(values) - min(values) if values else 0.0


def safe_action_set(shield: BankersShield, observation, memory) -> SafeActionSet:
    result = SafeActionSet(
        tuple(shield.certify(observation, memory, a) for a in ACTION_GRID)
    )
    if not any(result.mask) or not result.mask[0]:
        raise SafetyIntegrityError("EMPTY_MASK_OR_BACKUP_UNAVAILABLE")
    return result


def masked_distribution(logits, mask):
    """Interface utility only: no learner or gradient updates.

    Invalid choices have zero mass, zero entropy contribution and zero target
    contribution. A future learner must apply this to each relevant target mask.
    """
    values = np.asarray(logits, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.shape != (21,) or valid.shape != (21,) or not valid.any():
        raise SafetyIntegrityError("invalid masked distribution")
    if not np.isfinite(values[valid]).all():
        raise SafetyIntegrityError("nonfinite valid logits")
    weights = np.exp(values[valid] - np.max(values[valid]))
    probabilities = np.zeros(21)
    probabilities[valid] = weights / weights.sum()
    entropy = -float(np.sum(probabilities[valid] * np.log(probabilities[valid])))
    return probabilities, entropy
