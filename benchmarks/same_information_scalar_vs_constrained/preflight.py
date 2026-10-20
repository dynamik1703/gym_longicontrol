"""Read-only materialization of every preparation readiness gate."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_rl.costs import objective_reward, speed_cost
from benchmarks.constrained_rl_v2.costs import deadline_deficit_cost

from .checkpointing import capture_rng_states, load_verified, write_verified
from .config import load_configuration
from .execution import runtime_versions, source_hashes
from .initialization import verify_initialization
from .reference import verify_references
from .scalarization import frozen_scalarization
from .signals import canonical_transition_signals
from .split import build_split_ledger

HERE = Path(__file__).resolve().parent
PINNED_FSRL_SOURCE = {
    "policy/sac_lag.py": (
        "816829e005851c26701a4c28b5e05b444ac8a2cb07cfe2f3894274130fe59da1"
    ),
    "policy/lagrangian_base.py": (
        "3567ff5cd5dbb500927f5a64779d1bf76a81dfd7a9ceadaf9e2276b8c31f043c"
    ),
    "policy/base_policy.py": (
        "a3635c1156d42382429bf002bdab10da65cc0038e4f75c8b8587ae7a2298d7f0"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsrl_source_verified() -> bool:
    import fsrl

    root = Path(fsrl.__file__).resolve().parent
    return all(
        _sha256(root / relative) == expected
        for relative, expected in PINNED_FSRL_SOURCE.items()
    )


def _information_fixture_verified() -> bool:
    fixtures = (
        (0.001, 10.0, 12.0, 100.0, 30.0),
        (-0.0005, 10.0, 12.0, 120.0, 30.0),
        (0.0, 15.0, 12.0, 100.0, 30.0),
        (0.002, 14.0, 12.0, 130.0, 20.0),
    )
    for energy, velocity, limit, elapsed, remaining in fixtures:
        actual = canonical_transition_signals(
            signed_step_energy_kwh=energy,
            velocity_m_s=velocity,
            speed_limit_m_s=limit,
            elapsed_time_s=elapsed,
            optimistic_remaining_time_s=remaining,
        )
        expected = (
            objective_reward(energy, 0.25),
            speed_cost(max(0.0, velocity - limit), 0.1),
            deadline_deficit_cost(
                deficit_s=max(0.0, elapsed + remaining - 140.0),
                dt_s=0.1,
                normalization_s=140.0,
            ),
        )
        if not np.allclose(
            (actual.objective, actual.speed_cost, actual.deadline_cost),
            expected,
            rtol=0,
            atol=0,
        ):
            return False
    return True


def _recovery_verified(configuration) -> bool:
    payload: dict[str, Any] = {
        "checkpoint_schema_version": 1,
        "condition": "S11_sameinfo_scalar_seed11",
        "attempt_id": "preflight-fixture",
        "configuration_sha256": configuration.configuration_sha256,
        "source_hashes": source_hashes(),
        "runtime_versions": runtime_versions(),
        "policy": {},
        "policy_optimizers": {},
        "replay_buffer": [],
        "environment_state": {},
        "collector_state": {},
        "counters": {"real_transitions": 10_000, "rl_gradient_updates": 1_000},
        "diagnostic_checkpoints_completed": [],
        "rng_states": capture_rng_states(),
        "walltime_state": {"completed_segments_s": 1.0},
    }
    with tempfile.TemporaryDirectory(prefix="sameinfo-recovery-") as temporary:
        path = Path(temporary) / "checkpoint.pt"
        metadata = write_verified(path, payload)
        restored = load_verified(
            path,
            expected_sha256=metadata["sha256"],
            expected_configuration_sha256=configuration.configuration_sha256,
        )
    return restored["counters"] == payload["counters"]


def preparation_status() -> dict[str, Any]:
    configuration = load_configuration()
    scalar = frozen_scalarization()
    references = verify_references(configuration)
    ledger = build_split_ledger()
    initialization = verify_initialization(configuration)
    resource = json.loads((HERE / "resource_measurements.json").read_text())
    documents = (
        "README.md",
        "SOURCE_AUDIT.md",
        "DESIGN.md",
        "SCALARIZATION.md",
        "PROTOCOL.md",
        "SPLIT_LEDGER.md",
        "RESOURCE_REPORT.md",
        "canonical.json",
    )
    flags = {
        "scientific_design_frozen": all((HERE / name).is_file() for name in documents),
        "same_information_verified": _information_fixture_verified(),
        "scalarization_frozen": bool(
            np.isclose(scalar.w_reward + scalar.w_speed + scalar.w_deadline, 1.0)
        ),
        "reference_constrained_verified": references["reference_constrained_verified"],
        "canonical_streams_verified": ledger["checks"]["canonical_streams_have_865"],
        "diagnostic_set_frozen": len(
            ledger["evaluation_sets"]["diagnostic"]["accepted_tasks"]
        )
        == 18,
        "validation_set_frozen": len(
            ledger["evaluation_sets"]["validation"]["accepted_tasks"]
        )
        == 18,
        "route_clean_verified": ledger["route_clean_verified"],
        "scalar_runner_verified": initialization["passed"] and _fsrl_source_verified(),
        "computational_feasibility_verified": resource["passed"],
        "execution_infrastructure_ready": _recovery_verified(configuration)
        and (HERE / "launchd.plist").is_file(),
    }
    authorization = configuration.raw["authorization"]
    return {
        "schema_version": 1,
        "configuration_sha256": configuration.configuration_sha256,
        "runtime": runtime_versions(),
        "readiness": flags,
        "all_scientific_readiness_flags_true": all(flags.values()),
        "main_training_authorized": authorization["main_training_authorized"],
        "main_training_enabled": authorization["main_training_enabled"],
        "validation_authorized": authorization["validation_authorized"],
        "paper_test_authorized": authorization["paper_test_authorized"],
        "paper": ledger["paper"]["state"],
        "preparation_simulator_transitions": resource[
            "preparation_simulator_transitions"
        ],
        "new_policy_evaluations": resource["new_policy_evaluations"],
        "scientific_source_hashes": source_hashes(),
    }


if __name__ == "__main__":
    print(json.dumps(preparation_status(), indent=2, sort_keys=True))
