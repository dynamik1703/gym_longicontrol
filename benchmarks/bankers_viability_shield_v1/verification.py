"""Independent scalar witness verification and complete native-record replay."""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite

import numpy as np

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.domain.vehicle import VehicleModel

from .backup_policy import (
    BACKUP_ID,
    CONFIG,
    MIN_LIMIT,
    VELOCITY_MARGIN,
    CausalMemory,
    SafetyIntegrityError,
)
from .certificate import digest
from .safe_action_set import ACTION_GRID


def require(condition, reason):
    if not condition:
        raise SafetyIntegrityError("verification: " + reason)


def verify_certificate(certificate, ordinary, memory, vehicle=None):
    """Rebuild the witness without calling the admission implementation."""
    vehicle = vehicle if vehicle is not None else VehicleModel()
    c = certificate.as_dict() if hasattr(certificate, "as_dict") else certificate
    require(
        c["input_sha256"]
        == digest({"observation": list(ordinary), "memory": memory.snapshot()}),
        "input digest",
    )
    require(c["certified_backup_identity"] == BACKUP_ID, "backup identity")
    successor, _, _ = advance(memory.state, c["proposed_action"], vehicle, CONFIG)
    require(asdict(successor) == c["predicted_next_state"], "production successor")

    # Deliberately use a linear lower-envelope scan, independent of bound().
    def lower(position):
        result = MIN_LIMIT
        for start, value in memory.envelope:
            if start > position:
                break
            result = value
        return result

    endpoint = successor.velocity_m_s <= lower(successor.position_m) - VELOCITY_MARGIN
    accepted = endpoint
    path = []
    if endpoint:
        cursor = successor
        while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
            cursor, _, _ = advance(cursor, -1.0, vehicle, CONFIG)
            require(all(isfinite(v) for v in asdict(cursor).values()), "finite witness")
            path.append(asdict(cursor))
            accepted &= (
                cursor.velocity_m_s <= lower(cursor.position_m) - VELOCITY_MARGIN
            )
            require(len(path) <= 400, "finite backup bound")
        require(c["backup_steps"] == len(path), "backup step count")
        require(c["backup_terminal_state"] == asdict(cursor), "backup endpoint")
        require(c["backup_trajectory_sha256"] == digest(path), "witness digest")
    require(c["accepted"] == bool(accepted), "admission result")
    require(c["right_endpoint_compliant"] == bool(endpoint), "endpoint result")
    return True


def replay_records(records, track, method, expected_summary=None, full_masks=True):
    """Require every native record; reconstruct physics, sensor and task metrics.

    Route access is confined to this offline verifier. Online controller inputs
    are reconstructed anew from the retained observation/actions, not trusted.
    """
    vehicle = VehicleModel()
    state = VehicleState()
    memory = CausalMemory()
    maximum = integral = previous_excess = 0.0
    events = 0
    count = 0
    for step, row in enumerate(records):
        require(row["step"] == step, "missing/duplicate native record")
        require(row["method"] == method, "method identity")
        require(row["causal_pre_state"] == asdict(state), "causal pre-state")
        sensor = track.sense(state.position_m, 150)
        ordinary = observation(state, sensor, vehicle.specs, CONFIG, 1.0)
        require(np.array_equal(ordinary, row["observation"]), "ordinary observation")
        memory.ingest(ordinary)
        require(
            digest(memory.snapshot()) == digest(row["memory"]),
            "causal memory reconstruction",
        )
        certificates = row["certificate_summaries"]
        require(
            len(certificates) == 21 and len(row["safe_action_mask"]) == 21,
            "complete native menu",
        )
        if full_masks:
            # Independent full check of all 21 classifications, with minimal
            # transcript construction; selected certificate is checked fully.
            expected_mask = []
            for action in ACTION_GRID:
                cursor, _, _ = advance(state, action, vehicle, CONFIG)
                valid = (
                    cursor.velocity_m_s
                    <= memory.bound(cursor.position_m) - VELOCITY_MARGIN
                )
                if valid:
                    while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
                        cursor, _, _ = advance(cursor, -1.0, vehicle, CONFIG)
                        if (
                            cursor.velocity_m_s
                            > memory.bound(cursor.position_m) - VELOCITY_MARGIN
                        ):
                            valid = False
                        require(
                            cursor.elapsed_time_s < state.elapsed_time_s + 41,
                            "witness convergence",
                        )
                expected_mask.append(bool(valid))
            require(expected_mask == row["safe_action_mask"], "independent full mask")
        require(
            [c["accepted"] for c in certificates] == row["safe_action_mask"],
            "mask summaries",
        )
        require(
            digest([c["sha256"] for c in certificates]) == row["mask_sha256"],
            "mask certificate digest",
        )
        if method == "BANKERS_PRE":
            index = row["selected_index"]
            require(
                type(index) is int and row["safe_action_mask"][index], "valid selection"
            )
            require(row["executed_action"] == ACTION_GRID[index], "action index")
            require(
                row["selected_certificate"] is not None, "selected certificate exists"
            )
            verify_certificate(row["selected_certificate"], ordinary, memory, vehicle)
            require(
                digest(row["selected_certificate"]) == certificates[index]["sha256"],
                "selected mask certificate identity",
            )
            require(row["backup_available"], "backup availability")
        action = row["executed_action"]
        require(isfinite(action) and -1 <= action <= 1, "executed native action")
        successor, _, energy = advance(state, action, vehicle, CONFIG)
        limit = track.sense(successor.position_m, 150).current_limit_m_s
        excess = max(0.0, successor.velocity_m_s - limit)
        require(row["actual_next_state"] == asdict(successor), "actual successor")
        require(row["signed_energy_increment_kwh"] == energy, "signed energy")
        require(row["actual_speed_limit_m_s"] == limit, "actual limit")
        require(row["speed_excess_m_s"] == excess, "exact speed excess")
        if row["deadline_diagnostic_applicable"]:
            require(row["completion_bound_s"] is not None, "missing deadline bound")
            predicted, _, _ = advance(state, 1.0, vehicle, CONFIG)
            floor = MIN_LIMIT - VELOCITY_MARGIN
            while predicted.velocity_m_s < floor and predicted.position_m < 1000:
                predicted, _, _ = advance(predicted, 1.0, vehicle, CONFIG)
            bound = (
                predicted.elapsed_time_s + max(0, 1000 - predicted.position_m) / floor
            )
            require(row["completion_bound_s"] == bound, "independent deadline bound")
        else:
            require(row["completion_bound_s"] is None, "inapplicable deadline bound")
        maximum = max(maximum, excess)
        integral += excess * 0.1
        if excess > 0 and previous_excess == 0:
            events += 1
        previous_excess = excess
        state = successor
        memory.execute(action, vehicle)
        count += 1
        terminated = state.position_m >= 1000
        truncated = step + 1 == 1800 and not terminated
        require(
            row["terminated"] == terminated and row["truncated"] == truncated,
            "original lifecycle",
        )
        if terminated or truncated:
            require(step == len(records) - 1, "records after end")
    require(
        count > 0 and (state.position_m >= 1000 or count == 1800), "complete evidence"
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
        "steps": count,
        "metrics": asdict(metrics),
        "feasible": is_feasible(metrics, TaskSpecification(140, 0)),
    }
    if expected_summary:
        for key in result:
            require(result[key] == expected_summary[key], "summary " + key)
    return result
