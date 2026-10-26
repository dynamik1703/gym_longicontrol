"""Three admission mechanisms over the unchanged V4 physical contract."""

import numpy as np

from benchmarks.bankers_viability_shield_v1.certificate import BankersShield, digest
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
from benchmarks.bankers_viability_shield_v3.fast_certificate import safety_view
from benchmarks.bankers_viability_shield_v3.safe_action_mask import closest_certified
from benchmarks.bankers_viability_shield_v4.compact_certificate import (
    CompactShield, safe_action_mask,
)

METHODS = ('RAW_DISCRETE_SAC', 'BANKER_POST_SAC', 'BANKER_PRE_SAC')


def causal_binding(ordinary, memory):
    return digest(dict(observation=[float(x) for x in ordinary], memory=memory.snapshot()))


class Treatment:
    def __init__(self, method, identity):
        if method not in METHODS:
            raise ValueError('unknown treatment')
        self.method = method
        self.shield = CompactShield(identity)
        self.final = BankersShield()

    def mask(self, ordinary, memory):
        if self.method == METHODS[0]:
            return [True] * 21, [], causal_binding(ordinary, memory)
        certificates = safe_action_mask(self.shield, ordinary, memory)
        mask = [c['accepted'] for c in certificates]
        if not mask[0]:
            raise RuntimeError('EMPTY_MASK_OR_BACKUP_UNAVAILABLE')
        return mask, certificates, causal_binding(ordinary, memory)

    def policy_support(self, physical_mask):
        return physical_mask if self.method == METHODS[2] else [True] * 21

    def execute(self, proposal, ordinary, memory, mask, certificates, binding):
        if binding != causal_binding(ordinary, memory):
            raise RuntimeError('STALE_CURRENT_MASK')
        if type(proposal) is not int or not 0 <= proposal < 21:
            raise ValueError('invalid proposed action ID')
        selected = proposal
        final = None
        reason = 'RAW_DIRECT' if self.method == METHODS[0] else 'PROPOSAL_CERTIFIED'
        if self.method == METHODS[1] and not mask[proposal]:
            selected = closest_certified(mask, ACTION_GRID[proposal])
            reason = 'PROPOSAL_NOT_CERTIFIED'
        if self.method != METHODS[0]:
            if not mask[selected]:
                raise RuntimeError('PRE_INVALID_SAMPLE')
            final = self.final.certify(ordinary, memory, ACTION_GRID[selected]).as_dict()
            if not final['accepted'] or safety_view(final) != safety_view(certificates[selected]):
                raise RuntimeError('MASK_FINAL_CERTIFICATE_DISAGREEMENT')
        return selected, final, reason


def features(ordinary, memory):
    # Same partially observed causal base for every arm. Shield memory stays
    # external to the actor; PRE additionally restricts categorical support.
    return np.asarray([*ordinary, memory.state.position_m / 1000,
                       memory.state.elapsed_time_s / 140], dtype=np.float32)
