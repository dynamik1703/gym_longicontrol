"""Versioned safety-only certificates with exact brake transcripts."""

import base64
import hashlib
import struct
from math import isfinite

from benchmarks.bankers_viability_shield_v1.backup_policy import (
    BACKUP_ID,
    MIN_LIMIT,
    POSITION_MARGIN,
    VELOCITY_MARGIN,
    SafetyIntegrityError,
    checked_advance,
)
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield, digest
from gym_longicontrol.domain.vehicle import VehicleModel

from .fast_kinematics import (
    fast_kinematic_successor,
    model_finiteness_audit,
    reduction_domain,
)

SCHEMA = "bankers-speed-certificate-v3.0"
SAFETY_FIELDS = (
    "proposed_action",
    "accepted",
    "right_endpoint_compliant",
    "known_transition_braking_feasible",
    "unknown_preview_recoverable",
    "certified_backup_identity",
    "backup_feasible",
    "backup_steps",
    "minimum_velocity_slack_m_s",
    "recovery_distance_m",
    "preview_available_after_action_m",
    "distance_to_active_constraint_m",
    "numerical_margins",
    "rejection_reason",
    "classification",
    "input_sha256",
    "physics_prediction_count",
)


def point(state):
    if state is None:
        return None
    value = state if isinstance(state, dict) else vars(state)
    return {k: value[k] for k in ("position_m", "velocity_m_s")}


def safety_view(certificate):
    return {
        **{k: certificate[k] for k in SAFETY_FIELDS},
        "predicted_next_state": point(certificate["predicted_next_state"]),
        "backup_terminal_state": point(certificate["backup_terminal_state"]),
    }


def transcript_fields(path):
    raw = b"".join(struct.pack("!dd", x, v) for x, v in path)
    return dict(
        backup_transcript_b64=base64.b64encode(raw).decode("ascii"),
        backup_trajectory_sha256=hashlib.sha256(raw).hexdigest(),
    )


def fast_backup_witness(successor, memory, vehicle):
    cursor = successor
    path = []
    known = unknown = True
    slack = memory.bound(cursor.position_m) - cursor.velocity_m_s
    while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
        cursor = fast_kinematic_successor(cursor, -1.0, vehicle)
        path.append((cursor.position_m, cursor.velocity_m_s))
        bound = memory.bound(cursor.position_m)
        slack = min(slack, bound - cursor.velocity_m_s)
        if cursor.velocity_m_s > bound - VELOCITY_MARGIN:
            if cursor.position_m >= memory.frontier_m:
                unknown = False
            else:
                known = False
        if len(path) > 400:
            raise SafetyIntegrityError("backup failed finite bound")
    return cursor, path, known, unknown, slack


class FastBankersShield:
    def __init__(self, vehicle=None):
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.finiteness = model_finiteness_audit(self.vehicle)
        self.frozen = BankersShield(self.vehicle)

    def certify(self, ordinary, memory, action, *, input_hash=None):
        # The frozen implementation owns invalid/out-of-domain fail-closed behavior.
        try:
            valid = (
                self.finiteness is not None
                and reduction_domain(memory.state)
                and isfinite(float(action))
                and -1 <= float(action) <= 1
                and len(ordinary) == 8
                and all(isfinite(float(x)) for x in ordinary)
                and memory.last_observation == [float(x) for x in ordinary]
            )
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            full = self.frozen.certify(ordinary, memory, action).as_dict()
            result = safety_view(full)
            path = []
            if full["backup_terminal_state"] is not None:
                cursor, _, _ = checked_advance(
                    memory.state, float(action), self.vehicle
                )
                for _ in range(full["backup_steps"]):
                    cursor, _, _ = checked_advance(cursor, -1.0, self.vehicle)
                    path.append((cursor.position_m, cursor.velocity_m_s))
            return dict(
                schema=SCHEMA,
                **result,
                **transcript_fields(path),
                kernel="FROZEN_FALLBACK",
            )
        action = float(action)
        if input_hash is None:
            input_hash = digest(
                {
                    "observation": [float(x) for x in ordinary],
                    "memory": memory.snapshot(),
                }
            )
        successor = fast_kinematic_successor(memory.state, action, self.vehicle)
        available = memory.frontier_m - successor.position_m
        active = min(
            [
                p - successor.position_m
                for p, limit in memory.envelope
                if p >= successor.position_m and limit < successor.velocity_m_s
            ]
            or [max(0.0, available)]
        )
        endpoint = (
            successor.velocity_m_s
            <= memory.bound(successor.position_m) - VELOCITY_MARGIN
        )
        slack = memory.bound(successor.position_m) - successor.velocity_m_s
        terminal = None
        path = []
        known = unknown = True
        backup = False
        distance = None
        reason = "RIGHT_ENDPOINT_SPEED"
        if endpoint:
            terminal, path, known, unknown, slack = fast_backup_witness(
                successor, memory, self.vehicle
            )
            distance = terminal.position_m - successor.position_m
            backup = known and unknown
            reason = (
                "ACCEPTED"
                if backup
                else "KNOWN_TRANSITION_BACKUP"
                if not known
                else "UNKNOWN_PREVIEW_BACKUP"
            )
        return dict(
            schema=SCHEMA,
            proposed_action=action,
            accepted=bool(endpoint and backup),
            predicted_next_state=point(successor),
            right_endpoint_compliant=bool(endpoint),
            known_transition_braking_feasible=bool(known),
            unknown_preview_recoverable=bool(unknown),
            certified_backup_identity=BACKUP_ID,
            backup_feasible=bool(backup),
            backup_steps=len(path),
            backup_terminal_state=point(terminal),
            minimum_velocity_slack_m_s=slack,
            recovery_distance_m=distance,
            preview_available_after_action_m=available,
            distance_to_active_constraint_m=active,
            numerical_margins=dict(
                velocity_margin_m_s=VELOCITY_MARGIN,
                position_enclosure_m=POSITION_MARGIN,
            ),
            rejection_reason=reason,
            classification="FORMALLY_SUPPORTED_UNDER_STATED_MODEL",
            input_sha256=input_hash,
            physics_prediction_count=1 + len(path),
            **transcript_fields(path),
            kernel="KINEMATICS_ONLY",
        )
