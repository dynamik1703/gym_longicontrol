"""Disk replay using production dynamics, independently rebuilt witnesses."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from benchmarks.bankers_viability_shield_v1.certificate import (
    identities as v1_identities,
)
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.domain.vehicle import VehicleModel

from .inherited_shield_adapter import (
    ACTION_GRID,
    BACKUP_ID,
    BANKERS,
    CONFIG,
    MIN_LIMIT,
    POSITION_MARGIN,
    VELOCITY_MARGIN,
    CausalMemory,
    MeteredVehicle,
    _ExternalExecutionAdapter,
    inherited_test,
)
from .masked_causal_controller import closest_certified
from .serialization import canonical_bytes, digest, require, validate_records


def lower(memory, x):
    value = MIN_LIMIT
    for start, bound in memory.envelope:
        if start > x:
            break
        value = bound
    return value


def certificate_from_physics(ordinary, memory, action, vehicle, identity):
    """Does not call BankersShield.certify or CausalMemory.bound."""
    successor, _, _ = advance(memory.state, action, vehicle, CONFIG)
    require(
        all(math.isfinite(v) for v in asdict(successor).values()), "finite successor"
    )
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
        successor.velocity_m_s <= lower(memory, successor.position_m) - VELOCITY_MARGIN
    )
    slack = lower(memory, successor.position_m) - successor.velocity_m_s
    path = []
    terminal = distance = None
    known = unknown = backup = False
    if endpoint:
        known = unknown = True
        cursor = successor
        while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
            cursor, _, _ = advance(cursor, -1.0, vehicle, CONFIG)
            require(
                all(math.isfinite(v) for v in asdict(cursor).values()), "finite backup"
            )
            path.append(asdict(cursor))
            slack = min(slack, lower(memory, cursor.position_m) - cursor.velocity_m_s)
            if cursor.velocity_m_s > lower(memory, cursor.position_m) - VELOCITY_MARGIN:
                if cursor.position_m >= memory.frontier_m:
                    unknown = False
                else:
                    known = False
            require(len(path) <= 400, "finite backup convergence")
        terminal = asdict(cursor)
        distance = cursor.position_m - successor.position_m
        backup = known and unknown
    reason = (
        "RIGHT_ENDPOINT_SPEED"
        if not endpoint
        else "ACCEPTED"
        if backup
        else "KNOWN_TRANSITION_BACKUP"
        if not known
        else "UNKNOWN_PREVIEW_BACKUP"
    )
    # In V1 known/unknown flags are initialized true even for a rejected endpoint.
    if not endpoint:
        known = unknown = True
    return {
        "proposed_action": float(action),
        "accepted": bool(endpoint and backup),
        "predicted_next_state": asdict(successor),
        "right_endpoint_compliant": bool(endpoint),
        "known_transition_braking_feasible": bool(known),
        "unknown_preview_recoverable": bool(unknown),
        "certified_backup_identity": BACKUP_ID,
        "backup_feasible": bool(backup),
        "backup_steps": len(path),
        "backup_terminal_state": terminal,
        "backup_trajectory_sha256": digest(path) if terminal is not None else None,
        "minimum_velocity_slack_m_s": slack,
        "recovery_distance_m": distance,
        "preview_available_after_action_m": available,
        "distance_to_active_constraint_m": active,
        "numerical_margins": {
            "velocity_margin_m_s": VELOCITY_MARGIN,
            "position_enclosure_m": POSITION_MARGIN,
        },
        "rejection_reason": reason,
        "classification": "FORMALLY_SUPPORTED_UNDER_STATED_MODEL",
        "source_config_identity": identity,
        "input_sha256": digest(
            {"observation": list(ordinary), "memory": memory.snapshot()}
        ),
        "physics_prediction_count": 1 + len(path),
    }


def verify_envelope(memory, previous, track):
    # Check the retained bound against actual infrastructure at every discontinuity
    # and its adjacent representable endpoints. Also check monotone information.
    points = {memory.state.position_m, memory.frontier_m}
    points.update(p for p, _ in memory.envelope if p >= memory.state.position_m)
    points.update(float(p) for p in track.positions_m if p >= memory.state.position_m)
    points.update(p for p, _ in previous.envelope if p >= memory.state.position_m)
    for x in points:
        for p in (x, float(np.nextafter(x, -np.inf)), float(np.nextafter(x, np.inf))):
            if p < memory.state.position_m:
                continue
            require(
                lower(memory, p) <= track.sense(p, 150).current_limit_m_s,
                "lower envelope exceeds physical limit",
            )
            require(
                lower(memory, p) >= lower(previous, p), "information lost from envelope"
            )


def authority(certificates, state):
    admitted = [c for c in certificates if c["accepted"]]
    velocities = {c["predicted_next_state"]["velocity_m_s"] for c in admitted}
    actions = [c["proposed_action"] for c in admitted]
    reasons = [c["rejection_reason"] for c in certificates]
    return {
        "count": len(admitted),
        "distinct_velocities": len(velocities),
        "velocity_span_m_s": max(velocities) - min(velocities) if velocities else 0.0,
        "positive_acceleration_choices": sum(
            c["predicted_next_state"]["velocity_m_s"] > state.velocity_m_s
            for c in admitted
        ),
        "negative_acceleration_choices": sum(
            c["predicted_next_state"]["velocity_m_s"] < state.velocity_m_s
            for c in admitted
        ),
        "highest_action": max(actions) if actions else None,
        "lowest_action": min(actions) if actions else None,
        "known_transition_rejections": reasons.count("KNOWN_TRANSITION_BACKUP"),
        "unknown_preview_rejections": reasons.count("UNKNOWN_PREVIEW_BACKUP"),
        "endpoint_rejections": reasons.count("RIGHT_ENDPOINT_SPEED"),
        "numeric_integrity_rejections": sum(
            r.startswith("FAIL_CLOSED") for r in reasons
        ),
    }


def replay(rows, track, method, expected=None, *, complete=True, initial_state=None):
    validate_records(rows)
    vehicle = VehicleModel()
    metered = MeteredVehicle()
    causal = _ExternalExecutionAdapter(vehicle=metered)
    causal.reset()
    state = initial_state or VehicleState()
    causal.state = state
    memory = CausalMemory(state=state)
    maximum = integral = previous_excess = 0.0
    events = 0
    identity = v1_identities()
    for step, row in enumerate(rows):
        require(row["method"] == method, "method identity")
        require(
            canonical_bytes(row["causal_pre_state"]) == canonical_bytes(asdict(state)),
            "pre-state",
        )
        ordinary = observation(
            state, track.sense(state.position_m, 150), vehicle.specs, CONFIG, 1.0
        )
        require(
            np.array_equal(ordinary, row["observation"]), "observation reconstruction"
        )
        previous_memory = CausalMemory.restore(memory.snapshot())
        memory.ingest(ordinary)
        verify_envelope(memory, previous_memory, track)
        require(
            canonical_bytes(memory.snapshot()) == canonical_bytes(row["memory"]),
            "memory reconstruction",
        )
        fallback = causal.action(ordinary)
        require(fallback == row["causal60_proposal"], "frozen CAUSAL60 law")
        causal_snapshot = {
            "state": asdict(causal.state),
            "known_transitions": sorted(causal._known_transitions.items()),
            "last_action": causal._last_action,
        }
        require(
            canonical_bytes(causal_snapshot) == canonical_bytes(row["causal60_memory"]),
            "CAUSAL60 memory",
        )
        rebuilt = [
            certificate_from_physics(ordinary, memory, a, vehicle, identity)
            for a in ACTION_GRID
        ]
        require(
            canonical_bytes(rebuilt) == canonical_bytes(row["certificates"]),
            "independent complete certificates",
        )
        mask = [c["accepted"] for c in rebuilt]
        require(mask == row["safe_action_mask"], "independent mask")
        require(authority(rebuilt, state) == row["authority"], "physical authority")
        transitions = tuple(
            Transition(p, v) for p, v in sorted(causal._known_transitions.items())
        )
        old = (
            [
                inherited_test(
                    state, a, float(ordinary[2] * 37), transitions, fallback, metered
                )
                for a in ACTION_GRID
            ]
            if method != "CAUSAL60"
            else []
        )
        require(
            canonical_bytes(old) == canonical_bytes(row["inherited_comparison"]),
            "legacy predicates",
        )
        diag = (
            {
                "state": "CALCULATED",
                "applicable": True,
                "completion_bound_s": old[-1]["completion_bound_s"],
                "viable": old[-1]["deadline_viable"],
            }
            if old
            else {
                "state": "NOT_APPLICABLE",
                "applicable": False,
                "completion_bound_s": None,
                "viable": None,
            }
        )
        require(diag == row["deadline_diagnostic"], "deadline reconstruction")
        proposal = (
            fallback if method in ("CAUSAL60", "BANKERS_MASKED_CAUSAL60") else 1.0
        )
        require(row["proposal_action"] == proposal, "proposal law")
        selected = None
        if method == "CAUSAL60":
            action = fallback
        elif method == "LEGACY_NATIVE_POST_GREEDY":
            action = 1.0 if old[-1]["accepted"] else fallback
        elif method == "V2_SYNTHETIC_MINIMAL":
            require(mask[0], "empty mask/backup")
            selected, action = 0, -1.0
        else:
            require(mask[0] and any(mask), "empty mask/backup")
            selected = (
                max(i for i, valid in enumerate(mask) if valid)
                if method == "BANKERS_GREEDY"
                else closest_certified(mask, fallback)
            )
            action = ACTION_GRID[selected]
        require(
            row["selected_index"] == selected and row["executed_action"] == action,
            "selection law",
        )
        cert = certificate_from_physics(ordinary, memory, action, vehicle, identity)
        require(
            canonical_bytes(cert) == canonical_bytes(row["selected_certificate"]),
            "executed certificate",
        )
        if method in BANKERS or method == "V2_SYNTHETIC_MINIMAL":
            require(cert["accepted"], "selected certificate rejected")
        successor, _, energy = advance(state, action, vehicle, CONFIG)
        require(row["predicted_next_state"] == asdict(successor), "predicted successor")
        require(row["actual_next_state"] == asdict(successor), "actual successor")
        limit = track.sense(successor.position_m, 150).current_limit_m_s
        excess = max(0.0, successor.velocity_m_s - limit)
        require(
            row["actual_speed_limit_m_s"] == limit
            and row["speed_excess_m_s"] == excess,
            "exact speed meaning",
        )
        require(row["signed_energy_increment_kwh"] == energy, "signed energy meaning")
        if method in BANKERS or method == "V2_SYNTHETIC_MINIMAL":
            require(excess == 0, "BANKERS_V2_COUNTEREXAMPLE_FOUND")
        maximum = max(maximum, excess)
        integral += excess * CONFIG.dt_s
        events += int(excess > 0 and previous_excess == 0)
        previous_excess = excess
        terminated = successor.position_m >= 1000
        truncated = step + 1 == 1800
        require(
            type(row["terminated"]) is bool and type(row["truncated"]) is bool,
            "lifecycle type",
        )
        require(
            row["terminated"] == terminated and row["truncated"] == truncated,
            "original lifecycle",
        )
        if terminated or truncated:
            require(step == len(rows) - 1, "evidence after episode end")
        memory.execute(action, vehicle)
        causal.observe_external_action(action)
        state = successor
    if complete:
        require(
            state.position_m >= 1000 or len(rows) == 1800, "incomplete episode evidence"
        )
    metrics = EpisodeMetrics(
        state.position_m >= 1000,
        state.elapsed_time_s,
        state.total_energy_kwh,
        events,
        maximum,
        integral,
    )
    result = {
        "steps": len(rows),
        "metrics": asdict(metrics),
        "feasible": is_feasible(metrics, TaskSpecification(140, 0)),
    }
    if expected is not None:
        for k, v in result.items():
            require(expected[k] == v, "summary " + k)
    return result
