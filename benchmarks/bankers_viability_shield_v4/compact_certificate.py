"""Storage projection around the unchanged, frozen V3 online certifier."""

import base64
import struct

from benchmarks.bankers_viability_shield_v1.backup_policy import VELOCITY_MARGIN
from benchmarks.bankers_viability_shield_v3.fast_certificate import (
    FastBankersShield,
    safety_view,
)

SCHEMA = "bankers-speed-certificate-v4.0"


def compact(certificate, memory, identity, action_index=None):
    # The V3 algorithm is called once. Its transient transcript is never journaled.
    path = list(
        struct.iter_unpack(
            "!dd", base64.b64decode(certificate["backup_transcript_b64"], validate=True)
        )
    )
    terminal = certificate["backup_terminal_state"]
    if terminal is None:
        prefix = None
        status = (
            "ABSENT_FAIL_CLOSED"
            if certificate["rejection_reason"].startswith("FAIL_CLOSED")
            else "NOT_APPLICABLE_ENDPOINT_REJECTED"
        )
    else:
        prefix = 0
        for x, v in path:
            if v > memory.bound(x) - VELOCITY_MARGIN:
                break
            prefix += 1
        status = "VALID" if certificate["backup_feasible"] else "REJECTED"
    return dict(
        schema=SCHEMA,
        **safety_view(certificate),
        action_index=action_index,
        backup_record_status=status,
        backup_valid_prefix_steps=prefix,
        backup_trajectory_sha256=certificate["backup_trajectory_sha256"],
        source_config_identity=identity,
        reference_kernel=certificate["kernel"],
    )


class CompactShield:
    def __init__(self, identity, vehicle=None):
        self.reference = FastBankersShield(vehicle)
        self.vehicle = self.reference.vehicle
        self.identity = identity

    def certify(self, ordinary, memory, action, *, input_hash=None, action_index=None):
        return compact(
            self.reference.certify(ordinary, memory, action, input_hash=input_hash),
            memory,
            self.identity,
            action_index,
        )


def safe_action_mask(shield, ordinary, memory):
    from benchmarks.bankers_viability_shield_v1.certificate import digest
    from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID

    identity = digest(
        {"observation": [float(x) for x in ordinary], "memory": memory.snapshot()}
    )
    return [
        shield.certify(ordinary, memory, action, input_hash=identity, action_index=i)
        for i, action in enumerate(ACTION_GRID)
    ]
