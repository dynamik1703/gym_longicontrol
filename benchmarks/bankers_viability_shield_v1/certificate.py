"""Finite brake-witness certificates over exact production transitions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from gym_longicontrol.domain.vehicle import VehicleModel

from .backup_policy import (
    BACKUP_ID,
    MIN_LIMIT,
    POSITION_MARGIN,
    VELOCITY_MARGIN,
    CausalMemory,
    SafetyIntegrityError,
    checked_advance,
)

HERE = Path(__file__).resolve().parent
CLASSIFICATION = "FORMALLY_SUPPORTED_UNDER_STATED_MODEL"


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def identities() -> dict:
    audit = json.loads((HERE / "source_audit.json").read_text())
    return {
        "production_and_inherited_sha256": audit["ordered_source_sha256"],
        "configuration_sha256": hashlib.sha256(
            (HERE / "canonical.json").read_bytes()
        ).hexdigest(),
        "implementation_sha256": digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(HERE.glob("*.py"))
            }
        ),
    }


@dataclass(frozen=True)
class SpeedCertificate:
    proposed_action: float | None
    accepted: bool
    predicted_next_state: dict | None
    right_endpoint_compliant: bool
    known_transition_braking_feasible: bool
    unknown_preview_recoverable: bool
    certified_backup_identity: str
    backup_feasible: bool
    backup_steps: int
    backup_terminal_state: dict | None
    backup_trajectory_sha256: str | None
    minimum_velocity_slack_m_s: float | None
    recovery_distance_m: float | None
    preview_available_after_action_m: float | None
    distance_to_active_constraint_m: float | None
    numerical_margins: dict
    rejection_reason: str
    classification: str
    source_config_identity: dict
    input_sha256: str
    physics_prediction_count: int

    def as_dict(self):
        return asdict(self)

    @property
    def sha256(self):
        return digest(self.as_dict())


class BankersShield:
    def __init__(self, vehicle=None):
        self.vehicle = vehicle if vehicle is not None else VehicleModel()
        self.identity = identities()

    def certify(self, causal_observation, causal_controller_memory, proposed_action):
        memory: CausalMemory = causal_controller_memory
        action = None
        successor = terminal = None
        endpoint = known = unknown = backup = False
        path = []
        calls = 0
        min_slack = None
        distance = available = active_distance = None
        reason = "INVALID_INPUT"
        input_hash = ""
        try:
            action = float(proposed_action)
            if not isfinite(action) or not -1 <= action <= 1:
                action = None
                raise SafetyIntegrityError("invalid native action")
            ordinary = [float(v) for v in causal_observation]
            if memory.last_observation != ordinary:
                raise SafetyIntegrityError("observation not incorporated or stale")
            input_hash = digest({"observation": ordinary, "memory": memory.snapshot()})
            cursor, _, _ = checked_advance(memory.state, action, self.vehicle)
            calls += 1
            successor = asdict(cursor)
            available = memory.frontier_m - cursor.position_m
            active_distance = min(
                [
                    p - cursor.position_m
                    for p, limit in memory.envelope
                    if p >= cursor.position_m and limit < cursor.velocity_m_s
                ]
                or [max(0.0, available)]
            )
            endpoint = cursor.velocity_m_s <= (
                memory.bound(cursor.position_m) - VELOCITY_MARGIN
            )
            known = unknown = True
            min_slack = memory.bound(cursor.position_m) - cursor.velocity_m_s
            if not endpoint:
                reason = "RIGHT_ENDPOINT_SPEED"
            else:
                origin = cursor.position_m
                while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
                    cursor, _, _ = checked_advance(cursor, -1.0, self.vehicle)
                    calls += 1
                    path.append(asdict(cursor))
                    slack = memory.bound(cursor.position_m) - cursor.velocity_m_s
                    min_slack = min(min_slack, slack)
                    if (
                        cursor.velocity_m_s
                        > memory.bound(cursor.position_m) - VELOCITY_MARGIN
                    ):
                        if cursor.position_m >= memory.frontier_m:
                            unknown = False
                        else:
                            known = False
                    if len(path) > 400:
                        raise SafetyIntegrityError("backup failed finite bound")
                terminal = asdict(cursor)
                distance = cursor.position_m - origin
                backup = known and unknown
                reason = (
                    "ACCEPTED"
                    if backup
                    else (
                        "KNOWN_TRANSITION_BACKUP"
                        if not known
                        else "UNKNOWN_PREVIEW_BACKUP"
                    )
                )
        except (ValueError, TypeError, OverflowError, SafetyIntegrityError) as error:
            reason = "FAIL_CLOSED:" + str(error)
            backup = endpoint = known = unknown = False
        return SpeedCertificate(
            action,
            bool(endpoint and backup),
            successor,
            endpoint,
            known,
            unknown,
            BACKUP_ID,
            backup,
            len(path),
            terminal,
            digest(path) if terminal is not None else None,
            min_slack,
            distance,
            available,
            active_distance,
            {
                "velocity_margin_m_s": VELOCITY_MARGIN,
                "position_enclosure_m": POSITION_MARGIN,
            },
            reason,
            CLASSIFICATION,
            self.identity,
            input_hash,
            calls,
        )
