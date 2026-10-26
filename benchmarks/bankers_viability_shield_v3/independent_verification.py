"""Rebuild every applicable admission with complete original production physics.

This module never calls the fast kernel or FastBankersShield.
"""

from dataclasses import asdict

import numpy as np

from benchmarks.bankers_viability_shield_v1.certificate import (
    identities as original_identities,
)
from benchmarks.bankers_viability_shield_v2.independent_verification import (
    authority,
    certificate_from_physics,
    verify_envelope,
)
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
from .fast_certificate import safety_view, transcript_fields
from .serialization import canonical_bytes, require, validate_records


def verify_compact(c, ordinary, memory, vehicle, identity):
    rebuilt = certificate_from_physics(
        ordinary, memory, c["proposed_action"], vehicle, identity
    )
    require(safety_view(c) == safety_view(rebuilt), "independent safety predicate")
    path = []
    if rebuilt["backup_terminal_state"] is not None:
        cursor, _, _ = advance(memory.state, c["proposed_action"], vehicle, CONFIG)
        for _ in range(rebuilt["backup_steps"]):
            cursor, _, _ = advance(cursor, -1.0, vehicle, CONFIG)
            path.append((cursor.position_m, cursor.velocity_m_s))
    for k, v in transcript_fields(path).items():
        require(c[k] == v, "independent backup transcript")
    return rebuilt


def replay(rows, track, method, expected=None, *, complete=True, initial_state=None):
    validate_records(rows)
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
        require(fallback == row["causal60_proposal"], "CAUSAL60 law")
        transitions = tuple(
            Transition(p, v) for p, v in sorted(causal._known_transitions.items())
        )
        is_banker = method in BANKERS or method == "V3_SYNTHETIC_MINIMAL"
        require(
            len(row["certificates"]) == (21 if is_banker else 0),
            "method-specific grid scope",
        )
        rebuilt = [
            verify_compact(c, ordinary, memory, vehicle, identity)
            for c in row["certificates"]
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
            final = certificate_from_physics(
                ordinary, memory, action, vehicle, identity
            )
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
        limit = track.sense(successor.position_m, 150).current_limit_m_s
        excess = max(0.0, successor.velocity_m_s - limit)
        require(
            row["actual_speed_limit_m_s"] == limit
            and row["speed_excess_m_s"] == excess,
            "exact speed bound",
        )
        if is_banker:
            require(excess == 0, "BANKERS_V3_COUNTEREXAMPLE_FOUND")
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
