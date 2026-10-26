"""Offline falsification search; synthetic routes never enter the online API."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .backup_policy import CONFIG, MIN_LIMIT, VELOCITY_MARGIN, CausalMemory
from .certificate import HERE, BankersShield
from .fixtures import fixtures, make_track
from .integrity import atomic_json
from .safe_action_set import ACTION_GRID
from .verification import require


def search():
    vehicle = VehicleModel()
    shield = BankersShield(vehicle)
    rng = np.random.default_rng(20261026)
    tested = admitted = suffixes = 0
    for name, track in fixtures().items():
        for trial in range(40):
            x = float(rng.uniform(0, 999))
            current = track.sense(x, 150).current_limit_m_s
            velocity = float(rng.uniform(0, max(0, current - 2e-6)))
            state = VehicleState(position_m=x, velocity_m_s=velocity)
            ordinary = observation(state, track.sense(x, 150), vehicle.specs, CONFIG, 1)
            memory = CausalMemory(state=state)
            memory.ingest(ordinary)
            for action in ACTION_GRID:
                tested += 1
                certificate = shield.certify(ordinary, memory, action)
                if not certificate.accepted:
                    continue
                admitted += 1
                cursor, _, _ = advance(state, action, vehicle, CONFIG)
                actual_path = [asdict(cursor)]
                while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
                    cursor, _, _ = advance(cursor, -1, vehicle, CONFIG)
                    actual_path.append(asdict(cursor))
                    require(len(actual_path) < 402, "search brake convergence")
                bad = [
                    row
                    for row in actual_path
                    if row["velocity_m_s"]
                    > track.sense(row["position_m"], 150).current_limit_m_s
                ]
                if bad:
                    atomic_json(
                        HERE / "invariant_counterexample.json",
                        {
                            "fixture": name,
                            "trial": trial,
                            "state": asdict(state),
                            "action": action,
                            "observation": ordinary.tolist(),
                            "memory": memory.snapshot(),
                            "certificate": certificate.as_dict(),
                            "actual_path": actual_path,
                            "first_violation": bad[0],
                            "classification": "COUNTEREXAMPLE_FOUND",
                        },
                    )
                    raise AssertionError("admitted witness physically unsafe")
                next_memory = CausalMemory.restore(memory.snapshot())
                next_memory.execute(action, vehicle)
                next_obs = observation(
                    next_memory.state,
                    track.sense(next_memory.state.position_m, 150),
                    vehicle.specs,
                    CONFIG,
                    1,
                )
                next_memory.ingest(next_obs)
                require(
                    shield.certify(next_obs, next_memory, -1).accepted,
                    "counterexample to inductive backup availability",
                )
                suffixes += 1
    # Separate negative control: the unsound full-150-m successor shortcut.
    # This is not a counterexample to the proposed envelope invariant or the
    # inherited complete 60-km/h controller's reachable-state domain.
    latency_example = None
    track = make_track([0, 150.000001], [100, 20])
    for velocity in np.linspace(24, 27, 1201):
        state = VehicleState(velocity_m_s=float(velocity))
        next_state, _, _ = advance(state, 1, vehicle, CONFIG)
        cursor = next_state
        violating = None
        while cursor.velocity_m_s > MIN_LIMIT - VELOCITY_MARGIN:
            cursor, _, _ = advance(cursor, -1, vehicle, CONFIG)
            if cursor.position_m >= 150.000001 and cursor.velocity_m_s > MIN_LIMIT:
                violating = asdict(cursor)
        if cursor.position_m - next_state.position_m + 0.5 <= 150 and violating:
            ordinary = observation(state, track.sense(0, 150), vehicle.specs, CONFIG, 1)
            memory = CausalMemory()
            memory.state = state
            memory.ingest(ordinary)
            new = shield.certify(ordinary, memory, 1)
            require(
                not new.accepted, "corrected frontier rejects latency negative control"
            )
            latency_example = {
                "pre_state": asdict(state),
                "proposed_action": 1.0,
                "braking_distance_from_successor_m": cursor.position_m
                - next_state.position_m,
                "actual_first_bad_endpoint": violating,
                "new_certificate": new.as_dict(),
                "scope": (
                    "naive refreshed-preview rule outside old 60-km/h proposal cap"
                ),
            }
            break
    result = {
        "synthetic_states": 320,
        "candidate_certificates": tested,
        "admitted_witnesses_independently_executed": admitted,
        "inductive_successor_backups_checked": suffixes,
        "core_invariant_counterexamples": 0,
        "negative_control_latency_counterexample": latency_example,
        "classification": "EMPIRICAL_FALSIFICATION_SEARCH_ONLY",
    }
    atomic_json(HERE / "counterexample_search.json", result)
    return result


if __name__ == "__main__":
    print(search())
