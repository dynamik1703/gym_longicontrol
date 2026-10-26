"""Fresh V4 synthetic identities, read-only frozen V3 comparison inputs."""

import json
import time
from dataclasses import replace

from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG, CausalMemory
from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
from benchmarks.bankers_viability_shield_v3.controller_adapters import select
from benchmarks.bankers_viability_shield_v3.equivalence import test_states
from benchmarks.bankers_viability_shield_v3.fast_certificate import (
    FastBankersShield,
    safety_view,
)
from gym_longicontrol.domain.observation import observation

from .compact_certificate import CompactShield, safe_action_mask
from .independent_verification import verify_compact
from .integrity import HERE, source_identity, verify_sources
from .serialization import atomic_json, canonical_bytes, digest, require


def differential():
    identity = source_identity()
    verify_sources(identity)
    atomic_json(
        HERE / "started/V4_EQUIVALENCE_01.json",
        dict(identity=identity, rerun_forbidden=True),
        immutable=True,
    )
    start = time.process_time()
    reference = FastBankersShield()
    current = CompactShield(identity)
    physical = BankersShield()
    cases = []
    for fixture, track, state, history in test_states():
        memory = CausalMemory(state=state)
        if history:
            for x in range(0, 850, 10):
                memory.state = replace(state, position_m=float(x))
                memory.ingest(
                    observation(
                        memory.state,
                        track.sense(x, 150),
                        reference.vehicle.specs,
                        CONFIG,
                        1.0,
                    )
                )
            memory.state = state
        ordinary = observation(
            state,
            track.sense(state.position_m, 150),
            reference.vehicle.specs,
            CONFIG,
            1.0,
        )
        memory.ingest(ordinary)
        cases.append((fixture.replace("V3", "V4"), ordinary, memory))
    # Inherited adversarial state/memory identities are comparison fixtures only.
    old = json.loads(
        (
            HERE.parent / "bankers_viability_shield_v3/supplemental_equivalence.json"
        ).read_text()
    )
    cases.extend(
        (
            row["identity"].replace("V3", "V4"),
            row["observation"],
            CausalMemory.restore(row["memory"]),
        )
        for row in old["cases"]
    )
    summaries = []
    decisions = selections = 0
    for state_index, (name, ordinary, memory) in enumerate(cases):
        v3 = [reference.certify(ordinary, memory, a) for a in ACTION_GRID]
        v4 = safe_action_mask(current, ordinary, memory)
        try:
            for i, (before, after) in enumerate(zip(v3, v4)):
                require(
                    canonical_bytes(safety_view(before))
                    == canonical_bytes(safety_view(after)),
                    "V3/V4 safety meaning",
                )
                require(
                    before["backup_trajectory_sha256"]
                    == after["backup_trajectory_sha256"],
                    "V3/V4 transcript hash",
                )
                verify_compact(
                    after,
                    ordinary,
                    memory,
                    physical.vehicle,
                    physical.identity,
                    identity,
                    action_index=i,
                )
                decisions += 1
            before_mask = [c["accepted"] for c in v3]
            after_mask = [c["accepted"] for c in v4]
            require(before_mask == after_mask, "entire mask identity")
            # All original controller selections for deterministic continuous/grid proposals.  # noqa: E501
            if any(before_mask):
                for proposal in (*ACTION_GRID, -0.95, -0.05, 0.05, 0.95):
                    for method in ("BANKERS_GREEDY", "BANKERS_MASKED_CAUSAL60"):
                        require(
                            select(method, proposal, before_mask, None)
                            == select(method, proposal, after_mask, None),
                            "selected native action",
                        )
                        selections += 1
        except Exception as error:
            atomic_json(
                HERE / "first_equivalence_counterexample.json",
                dict(
                    identity=name,
                    state_index=state_index,
                    observation=list(ordinary),
                    memory=memory.snapshot(),
                    v3=v3,
                    v4=v4,
                    error=str(error),
                ),
                immutable=True,
            )
            atomic_json(
                HERE / "integrity_stop.json",
                dict(
                    classification="BANKERS_V4_EQUIVALENCE_STOP",
                    error=str(error),
                    rerun_forbidden=True,
                ),
                immutable=True,
            )
            raise
        summaries.append(
            dict(
                identity=name,
                input_sha256=v4[0]["input_sha256"],
                mask=after_mask,
                trajectory_hashes=[c["backup_trajectory_sha256"] for c in v4],
            )
        )
    result = dict(
        identity=identity,
        attempt="V4_EQUIVALENCE_01",
        states=len(cases),
        inherited_main_states=len(cases) - len(old["cases"]),
        grid_actions=decisions,
        independently_reconstructed_candidates=decisions,
        selections=selections,
        mismatches=0,
        cpu_s=time.process_time() - start,
        cases=summaries,
        cases_sha256=digest(summaries),
    )
    atomic_json(HERE / "equivalence_results.json", result, immutable=True)
    print("EQUIVALENCE PASS", result["states"], decisions, selections, flush=True)
    return result


if __name__ == "__main__":
    differential()
