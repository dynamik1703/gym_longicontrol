"""Independent full-production reconstruction. Never invokes the fast classifier."""

import hashlib
import math
import struct
from dataclasses import asdict

import numpy as np

from benchmarks.bankers_viability_shield_v1.backup_policy import (
    BACKUP_ID,
    MIN_LIMIT,
    POSITION_MARGIN,
    VELOCITY_MARGIN,
)
from benchmarks.bankers_viability_shield_v1.certificate import (
    BankersShield,
)
from benchmarks.bankers_viability_shield_v1.certificate import (
    identities as original_identities,
)
from benchmarks.bankers_viability_shield_v2.independent_verification import (
    authority,
    lower,
    verify_envelope,
)
from benchmarks.bankers_viability_shield_v3.fast_certificate import safety_view
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.domain.vehicle import VehicleModel

from .controller_adapters import (
    ACTION_GRID,
    BANKERS,
    CONFIG,
    DIAGNOSTIC_STRIDE,
    CausalMemory,
    MeteredVehicle,
    _ExternalExecutionAdapter,
    inherited_test,
    select,
)
from .serialization import canonical_bytes, digest, require, validate_records


def physical_certificate(ordinary, memory, action, vehicle, identity):
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
    }, path


def verify_compact(
    c,
    ordinary,
    memory,
    vehicle,
    identity,
    evidence_identity,
    *,
    action_index=None,
    raw_action=None,
):
    action = c["proposed_action"] if raw_action is None else raw_action
    try:
        domain = (
            all(
                math.isfinite(v) and abs(v) <= 1e100
                for v in vars(memory.state).values()
            )
            and 0 <= memory.state.velocity_m_s <= 37
            and math.isfinite(float(action))
            and -1 <= float(action) <= 1
            and len(ordinary) == 8
            and all(math.isfinite(float(x)) for x in ordinary)
            and memory.last_observation == [float(x) for x in ordinary]
        )
    except (ValueError, TypeError, OverflowError):
        domain = False
    if domain:
        rebuilt, path = physical_certificate(
            ordinary, memory, action, vehicle, identity
        )
    else:
        # Complete frozen physical certifier for invalid/out-of-domain inputs.
        # It is not the V3/V4 kinematic classifier.
        rebuilt = BankersShield(vehicle).certify(ordinary, memory, action).as_dict()
        path = []
        if rebuilt["backup_terminal_state"] is not None:
            cursor, _, _ = advance(memory.state, action, vehicle, CONFIG)
            for _ in range(rebuilt["backup_steps"]):
                cursor, _, _ = advance(cursor, -1.0, vehicle, CONFIG)
                path.append(asdict(cursor))
    # Endpoint, both preview predicates, all margins/reasons and terminal state
    # are checked independently of the trajectory hash.
    require(
        canonical_bytes(safety_view(c)) == canonical_bytes(safety_view(rebuilt)),
        "independent safety predicates/margins",
    )
    raw = b"".join(struct.pack("!dd", v["position_m"], v["velocity_m_s"]) for v in path)
    require(
        c["backup_trajectory_sha256"] == hashlib.sha256(raw).hexdigest(),
        "independent trajectory SHA-256",
    )
    if rebuilt["backup_terminal_state"] is None:
        prefix = None
        status = (
            "ABSENT_FAIL_CLOSED"
            if rebuilt["rejection_reason"].startswith("FAIL_CLOSED")
            else "NOT_APPLICABLE_ENDPOINT_REJECTED"
        )
    else:
        prefix = 0
        for v in path:
            if v["velocity_m_s"] > lower(memory, v["position_m"]) - VELOCITY_MARGIN:
                break
            prefix += 1
        status = "VALID" if rebuilt["backup_feasible"] else "REJECTED"
    require(
        c["backup_valid_prefix_steps"] == prefix
        and c["backup_record_status"] == status,
        "independent valid backup prefix/status",
    )
    require(
        c["source_config_identity"] == evidence_identity
        and c["action_index"] == action_index,
        "candidate source/action identity",
    )
    require(c["schema"] == "bankers-speed-certificate-v4.0", "candidate schema")
    require(
        c["reference_kernel"] == ("KINEMATICS_ONLY" if domain else "FROZEN_FALLBACK"),
        "kernel domain",
    )
    return rebuilt


def replay(
    rows,
    track,
    method,
    expected=None,
    *,
    complete=True,
    initial_state=None,
    evidence_identity=None,
    attempt=None,
):
    validate_records(rows)
    from .integrity import source_identity

    evidence_identity = evidence_identity or source_identity()
    attempt = attempt or rows[0]["attempt"]
    vehicle = VehicleModel()
    metered = MeteredVehicle()
    causal = _ExternalExecutionAdapter(vehicle=metered)
    causal.reset()
    state = initial_state or VehicleState()
    causal.state = state
    memory = CausalMemory(state=state)
    identity = original_identities()
    maximum = integral = previous = 0.0
    events = 0
    for step, row in enumerate(rows):
        require(
            row["evidence_identity"] == evidence_identity and row["attempt"] == attempt,
            "independent source/attempt identity",
        )
        require(
            row["method"] == method and row["causal_pre_state"] == asdict(state),
            "state/method reconstruction",
        )
        ordinary = observation(
            state, track.sense(state.position_m, 150), vehicle.specs, CONFIG, 1.0
        )
        require(
            np.array_equal(ordinary, row["observation"]), "observation reconstruction"
        )
        old_memory = CausalMemory.restore(memory.snapshot())
        memory.ingest(ordinary)
        verify_envelope(memory, old_memory, track)
        require(
            canonical_bytes(memory.snapshot()) == canonical_bytes(row["memory"]),
            "causal memory",
        )
        fallback = causal.action(ordinary)
        require(
            canonical_bytes(row["causal60_memory"])
            == canonical_bytes(
                dict(
                    state=asdict(causal.state),
                    known_transitions=sorted(causal._known_transitions.items()),
                    last_action=causal._last_action,
                )
            ),
            "causal60 memory",
        )
        require(fallback == row["causal60_proposal"], "CAUSAL60 law")
        transitions = tuple(
            Transition(p, v) for p, v in sorted(causal._known_transitions.items())
        )
        is_banker = method in BANKERS or method == "V4_SYNTHETIC_MINIMAL"
        require(
            len(row["certificates"]) == (21 if is_banker else 0),
            "method-specific grid scope",
        )
        rebuilt = [
            verify_compact(
                c,
                ordinary,
                memory,
                vehicle,
                identity,
                evidence_identity,
                action_index=i,
            )
            for i, c in enumerate(row["certificates"])
        ]
        require(
            [c["proposed_action"] for c in rebuilt]
            == (list(ACTION_GRID) if is_banker else []),
            "complete grid",
        )
        mask = [c["accepted"] for c in rebuilt]
        require(mask == row["safe_action_mask"], "independent mask")
        if method == "BANKERS_MASKED_CAUSAL60":
            require(
                row["causal60_proposal_certificate"]["proposed_action"] == fallback,
                "causal proposal identity",
            )
            verify_compact(
                row["causal60_proposal_certificate"],
                ordinary,
                memory,
                vehicle,
                identity,
                evidence_identity,
                action_index=(
                    ACTION_GRID.index(fallback) if fallback in ACTION_GRID else None
                ),
            )
        else:
            require(
                row["causal60_proposal_certificate"] is None,
                "proposal certificate scope",
            )
        if is_banker:
            require(mask[0] and any(mask), "certified backup")
            require(authority(rebuilt, state) == row["authority"], "authority")
        old = (
            inherited_test(
                state, 1.0, float(ordinary[2] * 37), transitions, fallback, metered
            )
            if method == "LEGACY_NATIVE_POST_GREEDY"
            else None
        )
        require(old == row["legacy_proposal_check"], "legacy proposal predicate")
        sampled = is_banker and step % DIAGNOSTIC_STRIDE == 0
        require(row["diagnostic_sampled"] == sampled, "diagnostic sampling rule")
        diagnostic = (
            [
                inherited_test(
                    state, a, float(ordinary[2] * 37), transitions, fallback, metered
                )
                for a in ACTION_GRID
            ]
            if sampled
            else []
        )
        require(
            canonical_bytes(diagnostic) == canonical_bytes(row["inherited_comparison"]),
            "sampled inherited predicates",
        )
        value = old or (diagnostic[-1] if diagnostic else None)
        diag = (
            dict(
                state="CALCULATED",
                applicable=True,
                completion_bound_s=value["completion_bound_s"],
                viable=value["deadline_viable"],
            )
            if value
            else dict(
                state="NOT_APPLICABLE",
                applicable=False,
                completion_bound_s=None,
                viable=None,
            )
        )
        require(
            canonical_bytes(diag) == canonical_bytes(row["deadline_diagnostic"]),
            "deadline diagnostic",
        )
        index, action = select(method, fallback, mask, old)
        require(
            row["selected_index"] == index and row["executed_action"] == action,
            "selection law",
        )
        require(
            row["proposal_action"]
            == (fallback if method in ("CAUSAL60", "BANKERS_MASKED_CAUSAL60") else 1.0),
            "proposal law",
        )
        if is_banker:
            final, _ = physical_certificate(ordinary, memory, action, vehicle, identity)
            require(
                final == row["selected_certificate"] and final["accepted"],
                "full selected certificate",
            )
        else:
            require(row["selected_certificate"] is None, "certificate scope")
        successor, power, energy = advance(state, action, vehicle, CONFIG)
        require(
            row["predicted_next_state"]
            == asdict(successor)
            == row["actual_next_state"],
            "complete production state",
        )
        require(
            row["actual_power_kw"] == power
            and row["signed_energy_increment_kwh"] == energy,
            "signed native energy",
        )
        require(row["native_time_s"] == successor.elapsed_time_s, "native timestamp")
        require(row["completed"] == (successor.position_m >= 1000), "completion flag")
        limit = track.sense(successor.position_m, 150).current_limit_m_s
        excess = max(0.0, successor.velocity_m_s - limit)
        require(
            row["actual_speed_limit_m_s"] == limit
            and row["speed_excess_m_s"] == excess,
            "exact speed bound",
        )
        if is_banker:
            require(excess == 0, "BANKERS_V4_COUNTEREXAMPLE_FOUND")
        maximum = max(maximum, excess)
        integral += excess * CONFIG.dt_s
        events += int(excess > 0 and previous == 0)
        previous = excess
        require(
            row["terminated"] == (successor.position_m >= 1000)
            and row["truncated"] == (step + 1 == 1800),
            "native lifecycle",
        )
        if row["terminated"] or row["truncated"]:
            require(step == len(rows) - 1, "evidence after termination")
        memory.execute(action, vehicle)
        causal.observe_external_action(action)
        state = successor
    if complete:
        require(
            state.position_m >= 1000 or len(rows) == 1800, "complete native episode"
        )
    metrics = EpisodeMetrics(
        state.position_m >= 1000,
        state.elapsed_time_s,
        state.total_energy_kwh,
        events,
        maximum,
        integral,
    )
    result = dict(
        steps=len(rows),
        metrics=asdict(metrics),
        feasible=is_feasible(metrics, TaskSpecification(140, 0)),
    )
    if expected:
        for k, v in result.items():
            require(expected[k] == v, "summary " + k)
    return result
