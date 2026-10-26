"""Additional inherited boundary scenarios with fresh V3 identities."""

import time
from dataclasses import asdict

from benchmarks.bankers_viability_shield_v1.backup_policy import (
    CONFIG,
    MIN_LIMIT,
    CausalMemory,
)
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState

from .fast_certificate import FastBankersShield, safety_view
from .independent_verification import verify_compact
from .integrity import HERE
from .serialization import atomic_json, require


def supplemental():
    start = time.process_time()
    fast = FastBankersShield()
    old = BankersShield()
    constant = make_track([0], [100])

    def context(track, state):
        o = observation(
            state, track.sense(state.position_m, 150), fast.vehicle.specs, CONFIG, 1.0
        )
        m = CausalMemory(state=state)
        m.ingest(o)
        return o, m

    def boundary(action):
        low, high = 25.0, 27.0
        for _ in range(50):
            middle = (low + high) / 2
            o, m = context(constant, VehicleState(velocity_m_s=middle))
            if old.certify(o, m, action).accepted:
                low = middle
            else:
                high = middle
        return low

    one_speed = (boundary(-1.0) + boundary(-0.9)) / 2
    cases = [("V3_ONLY_ONE_CERTIFIED", constant, VehicleState(velocity_m_s=one_speed))]
    for p in (149.999999999, 150.0, 150.000000001):
        cases.append(
            ("V3_EXACT_SIGN_FRONTIER", make_track([0, p], [60, 20]), VehicleState())
        )
    import numpy as np

    for v in (
        MIN_LIMIT,
        np.nextafter(MIN_LIMIT, -np.inf),
        np.nextafter(MIN_LIMIT, np.inf),
        MIN_LIMIT - 1e-6,
        np.nextafter(MIN_LIMIT - 1e-6, np.inf),
        0.5,
    ):
        cases.append(("V3_ROUNDING", constant, VehicleState(velocity_m_s=float(v))))
    cases.extend(
        [
            (
                "V3_WITHIN_STEP",
                make_track([0, 100], [60, 20]),
                VehicleState(position_m=99.9, velocity_m_s=10.0),
            ),
            (
                "V3_KNOWN_BACKUP",
                make_track([0, 300], [60, 20]),
                VehicleState(position_m=260.0, velocity_m_s=16.0),
            ),
            (
                "V3_BACKUP_SUFFIX",
                make_track([0, 300, 400], [60, 40, 20]),
                VehicleState(position_m=200.0, velocity_m_s=15.0),
            ),
        ]
    )
    results = []
    for name, track, state in cases:
        ordinary, memory = context(track, state)
        certs = []
        for action in ACTION_GRID:
            frozen = old.certify(ordinary, memory, action).as_dict()
            current = fast.certify(ordinary, memory, action)
            if safety_view(frozen) != safety_view(current):
                atomic_json(
                    HERE / "first_equivalence_counterexample.json",
                    dict(
                        identity=name,
                        state=asdict(state),
                        action=action,
                        observation=ordinary.tolist(),
                        memory=memory.snapshot(),
                        frozen=frozen,
                        optimized=current,
                    ),
                    immutable=True,
                )
                raise RuntimeError("BANKERS_V3_EQUIVALENCE_STOP")
            verify_compact(current, ordinary, memory, old.vehicle, old.identity)
            certs.append(current)
        count = sum(c["accepted"] for c in certs)
        if name == "V3_ONLY_ONE_CERTIFIED":
            require(count == 1, "one-action fixture construction")
        results.append(
            dict(
                identity=name,
                state=asdict(state),
                observation=ordinary.tolist(),
                memory=memory.snapshot(),
                admitted=count,
            )
        )
    result = dict(
        states=len(cases),
        grid_actions=21 * len(cases),
        mismatches=0,
        cases=results,
        cpu_s=time.process_time() - start,
    )
    atomic_json(HERE / "supplemental_equivalence.json", result, immutable=True)
    print("SUPPLEMENTAL PASS", result["states"], result["grid_actions"], flush=True)


if __name__ == "__main__":
    supplemental()
