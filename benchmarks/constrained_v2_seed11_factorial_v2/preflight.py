"""Static route-clean gate for the repaired seed-11 factorial study.

This module never calls ``env.step`` and never constructs a Paper-final task.
It reconstructs the largest training-stream prefix that any physically possible
300,000-transition policy run can expose, using the canonical content fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_v2_repro_audit.harness import (
    current_track_fingerprints,
    historical_track_fingerprints,
)
from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import (
    continuous_tracks,
    duplicate_statistics,
    explicit_seed_stream,
    fingerprint_rows,
    track_from_seed,
)
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
CONTENT_AUDIT = REPOSITORY / "benchmarks/content_split_audit/results.json"
REPRO_AUDIT = REPOSITORY / "benchmarks/constrained_v2_repro_audit/results.json"
CANONICAL = HERE / "canonical.json"
TOTAL_TRANSITIONS = 300_000
TRAINING_SEED = 11
CURRENT_DOMAIN_TAG = 0x4D564531
DIAGNOSTIC_SEEDS = tuple(range(9_000_000, 9_000_018))
SOURCE_COMMIT = "ab6df07b2e28fadc87ab05c3f64f76b7b2bef6cb"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, Any]:
    import torch

    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gymnasium": _version("gymnasium"),
        "tianshou": _version("tianshou"),
        "fsrl": _version("fsrl"),
        "fsrl_revision": "e056fc9498d5d037869533da7cf976acf462f918",
    }


def fastest_physical_episode_steps() -> int:
    """Return the exact route-independent minimum under bounded actions.

    ``action=1`` selects the greatest available acceleration at every state.  The
    transition dynamics and action bounds are route independent, so this trajectory
    maximizes position at every step and supplies the exact lower episode-length
    bound without taking a Gymnasium simulator transition.
    """

    config = SimulationConfig()
    vehicle = VehicleModel()
    state = VehicleState()
    steps = 0
    while state.position_m < config.track_length_m:
        state, _power, _energy = advance(state, 1.0, vehicle, config)
        steps += 1
    return steps


def maximum_task_exposures() -> dict[str, int]:
    minimum_steps = fastest_physical_episode_steps()
    completed = TOTAL_TRANSITIONS // minimum_steps
    return {
        "minimum_physical_episode_steps": minimum_steps,
        "maximum_completed_episodes": completed,
        "maximum_exposed_tasks_including_final_active_episode": completed + 1,
    }


def _stream_rows(count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    semantics = v2_semantics()
    historical = continuous_tracks(TRAINING_SEED, list(range(0, 2 * count, 2)))
    historical_rows = fingerprint_rows(
        historical,
        semantics,
        numeric_seeds=[None] * count,
        draw_indices=list(range(0, 2 * count, 2)),
    )
    current_seeds = explicit_seed_stream(
        TRAINING_SEED,
        domain_tag=CURRENT_DOMAIN_TAG,
        count=count,
    )
    current_tracks = [track_from_seed(seed) for seed in current_seeds]
    current_rows = fingerprint_rows(
        current_tracks,
        semantics,
        numeric_seeds=current_seeds,
        draw_indices=list(range(count)),
    )
    return historical_rows, current_rows


def _stream_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episode_count": len(rows),
        "ordered_rows_sha256": canonical_sha256(list(rows)),
        "first_row": dict(rows[0]),
        "last_row": dict(rows[-1]),
        "duplicates": duplicate_statistics(rows),
    }


def _audit_prefix_checks() -> dict[str, Any]:
    audit = _load_json(REPRO_AUDIT)
    expected = audit["rng_and_tracks"]["by_seed"][str(TRAINING_SEED)]
    historical = historical_track_fingerprints(TRAINING_SEED, count=100)
    current = current_track_fingerprints(TRAINING_SEED, count=100)
    return {
        "audit_results_path": str(REPRO_AUDIT.relative_to(REPOSITORY)),
        "audit_results_sha256": file_sha256(REPRO_AUDIT),
        "historical_first_100_match": historical
        == expected["historical_first_100_used_track_fingerprints"],
        "current_first_100_match": current
        == expected["current_first_100_used_track_fingerprints"],
        "historical_first_100_sha256": canonical_sha256(historical),
        "current_first_100_sha256": canonical_sha256(current),
    }


def _diagnostic_rows() -> list[dict[str, Any]]:
    content_audit = _load_json(CONTENT_AUDIT)
    frozen = content_audit["proposed_diagnostic_set"]["candidates"]
    if [row["numeric_seed"] for row in frozen] != list(DIAGNOSTIC_SEEDS):
        raise RuntimeError("The content-audit diagnostic seed list changed")
    semantics = v2_semantics()
    reconstructed = []
    for frozen_row in frozen:
        seed = int(frozen_row["numeric_seed"])
        fingerprint = fingerprint_track(
            track_from_seed(seed), semantics, numeric_seed=seed
        )
        row = {
            "numeric_seed": seed,
            "route_sha256": fingerprint.route_sha256,
            "full_task_sha256": fingerprint.full_task_sha256,
        }
        expected = {
            key: frozen_row[key]
            for key in ("numeric_seed", "route_sha256", "full_task_sha256")
        }
        if row != expected:
            raise RuntimeError(f"Frozen diagnostic fingerprint changed for {seed}")
        reconstructed.append(row)
    return reconstructed


def _matches(
    diagnostic: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    route = []
    full_task = []
    numeric = []
    for row in rows:
        evidence = {
            "training_episode_index": int(row["episode_index"]),
            "training_reset_seed": row["reset_seed"],
            "training_rng_draw_index": row["rng_draw_index"],
        }
        if row["route_sha256"] == diagnostic["route_sha256"]:
            route.append(evidence)
        if row["full_task_sha256"] == diagnostic["full_task_sha256"]:
            full_task.append(evidence)
        if row["reset_seed"] == diagnostic["numeric_seed"]:
            numeric.append(evidence)
    return {"numeric_seed": numeric, "route": route, "full_task": full_task}


@lru_cache(maxsize=1)
def build_report() -> dict[str, Any]:
    coverage = maximum_task_exposures()
    count = coverage["maximum_exposed_tasks_including_final_active_episode"]
    historical, current = _stream_rows(count)
    diagnostic = _diagnostic_rows()
    clean_rows = []
    collisions = []
    for row in diagnostic:
        by_stream = {
            "T_HIST": _matches(row, historical),
            "T_CURRENT": _matches(row, current),
        }
        numeric_clean = all(not value["numeric_seed"] for value in by_stream.values())
        route_absent = all(not value["route"] for value in by_stream.values())
        full_absent = all(not value["full_task"] for value in by_stream.values())
        clean = numeric_clean and route_absent and full_absent
        clean_row = {
            **row,
            "numeric_seed_clean": numeric_clean,
            "route_sha256_absent_from_both_streams": route_absent,
            "full_task_sha256_absent_from_both_streams": full_absent,
            "clean": clean,
            "matches": by_stream,
        }
        clean_rows.append(clean_row)
        if not clean:
            collisions.append(clean_row)
    prefix_checks = _audit_prefix_checks()
    stream_verified = (
        prefix_checks["historical_first_100_match"]
        and prefix_checks["current_first_100_match"]
    )
    route_clean = not collisions
    return {
        "schema_version": 1,
        "study": (
            "Constrained V2 Seed-11 Track x Collector 2x2 -- "
            "Route-Clean Evaluation"
        ),
        "source_commit": SOURCE_COMMIT,
        "status": (
            "READY_FOR_REMAINING_PREFLIGHTS"
            if route_clean and stream_verified
            else "EXECUTION_GATE_FAILED"
        ),
        "stop_reason": (
            None
            if route_clean
            else "frozen diagnostic content occurs in an exact training stream"
        ),
        "main_training_authorized": False,
        "coverage": {
            **coverage,
            "total_transition_budget": TOTAL_TRANSITIONS,
            "derivation": (
                "action=1 maximizes acceleration and position at every "
                "route-independent "
                "physical step; after the maximum number of completed episodes, one "
                "additional task may be active at the lifetime truncation"
            ),
        },
        "streams": {
            "T_HIST": {
                "definition": "continuous RNG seed 11; used draws 0,2,4,...",
                **_stream_summary(historical),
            },
            "T_CURRENT": {
                "definition": (
                    "default_rng(SeedSequence([11, 0x4D564531])); explicit "
                    "reset seeds; legacy reserved numeric IDs rejected"
                ),
                **_stream_summary(current),
            },
        },
        "stream_anchor_checks": prefix_checks,
        "diagnostic_source": {
            "path": str(CONTENT_AUDIT.relative_to(REPOSITORY)),
            "sha256": file_sha256(CONTENT_AUDIT),
            "candidate_count": len(diagnostic),
            "ordered_fingerprints_sha256": canonical_sha256(diagnostic),
        },
        "diagnostic_route_clean_checks": clean_rows,
        "collisions": collisions,
        "checks": {
            "exact_stream_prefixes_match_reproducibility_audit": stream_verified,
            "all_18_diagnostic_tasks_route_clean": route_clean,
            "numeric_seed_clean": all(row["numeric_seed_clean"] for row in clean_rows),
            "route_hashes_absent": all(
                row["route_sha256_absent_from_both_streams"] for row in clean_rows
            ),
            "full_task_hashes_absent": all(
                row["full_task_sha256_absent_from_both_streams"]
                for row in clean_rows
            ),
        },
        "execution": {
            "attempts_created": 0,
            "conditions_started": 0,
            "real_training_transitions": 0,
            "gradient_updates": 0,
            "diagnostic_evaluation_episodes": 0,
            "legacy_development_evaluation_episodes": 0,
            "legacy_validation_evaluation_episodes": 0,
            "paper_evaluation_episodes": 0,
            "runtime_preflight_executed": False,
            "learner_initialization_preflight_executed": False,
            "collector_implementation_executed": False,
        },
        "seals": {
            "legacy_development": "NOT_EVALUATED",
            "legacy_validation": "NOT_EVALUATED",
            "paper_4000_4017": "SEALED_UNINSPECTED",
        },
        "runtime": runtime_versions(),
        "source_files": {
            "canonical.json": file_sha256(CANONICAL),
            "preflight.py": file_sha256(Path(__file__)),
        },
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize() -> dict[str, Any]:
    report = build_report()
    _atomic_json(HERE / "results.json", report)
    manifest = {
        "schema_version": 1,
        "study": report["study"],
        "source_commit": SOURCE_COMMIT,
        "state": report["status"],
        "reason": report["stop_reason"],
        "conditions": {
            name: {
                "state": "NOT_STARTED",
                "attempts": [],
                "real_transitions": 0,
                "gradient_updates": 0,
            }
            for name in _load_json(CANONICAL)["condition_order"]
        },
        "diagnostic": {"state": "NOT_EVALUATED", "episode_count": 0},
        "legacy_development": {"state": "NOT_EVALUATED", "episode_count": 0},
        "legacy_validation": {"state": "NOT_EVALUATED", "episode_count": 0},
        "paper": {"state": "SEALED_UNINSPECTED", "episode_count": 0},
    }
    _atomic_json(HERE / "manifest.json", manifest)
    ledger = {
        "schema_version": 1,
        "state": "NOT_EVALUATED",
        "reason": report["stop_reason"],
        "expected_episodes_if_all_gates_passed": 432,
        "actual_episodes": 0,
        "episodes": [],
    }
    _atomic_json(HERE / "diagnostic_ledger.json", ledger)
    freeze_names = (
        "canonical.json",
        "README.md",
        "DESIGN.md",
        "PROTOCOL.md",
        "RESOURCE_REPORT.md",
        "RESULTS.md",
        "preflight.py",
        "results.json",
        "manifest.json",
        "diagnostic_ledger.json",
    )
    freeze = {
        "schema_version": 1,
        "source_commit": SOURCE_COMMIT,
        "terminal_state": report["status"],
        "main_training_started": False,
        "files": {name: file_sha256(HERE / name) for name in freeze_names},
    }
    _atomic_json(HERE / "freeze.json", freeze)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args(argv)
    report = materialize() if args.materialize else build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] != "READY_FOR_REMAINING_PREFLIGHTS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
