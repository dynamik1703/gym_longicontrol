"""Split construction, fixtures, source hashing and resource preflight."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import (
    ALL_NUMERICALLY_RESERVED,
    PAPER_SEEDS,
    track_from_seed,
)
from benchmarks.gcsl_canonical.split import (
    _training_rows,
    evaluation_sets,
    opened_split_rows,
)
from gym_longicontrol.domain.track import Track

from .evaluation import canonical_sha256, rollout

HERE = Path(__file__).resolve().parent
DEV_ID = "PHYSICS_REFERENCE_DEV_V1"
HOLDOUT_ID = "PHYSICS_REFERENCE_HOLDOUT_V1"
DEV_START = 9_800_000
HOLDOUT_START = 9_900_000
DEV_COUNT = 18
HOLDOUT_COUNT = 36


def _projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_seed": int(row.get("numeric_seed", row.get("reset_seed"))),
        "route_sha256": row["route_sha256"],
        "full_task_sha256": row["full_task_sha256"],
    }


def protected_rows() -> list[dict[str, Any]]:
    rows = [
        _projection(row)
        for split_rows in opened_split_rows().values()
        for row in split_rows
    ]
    rows.extend(_projection(row) for row in _training_rows())
    rows.extend(
        _projection(row)
        for split in evaluation_sets().values()
        for row in split["accepted_tasks"]
    )
    return rows


def _select(
    split_id: str,
    start: int,
    count: int,
    *,
    protected: list[dict[str, Any]],
    other: list[dict[str, Any]],
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    numeric = {
        *ALL_NUMERICALLY_RESERVED,
        *PAPER_SEEDS,
        *(row["numeric_seed"] for row in protected),
        *(row["numeric_seed"] for row in other),
    }
    route_hashes = {
        *(row["route_sha256"] for row in protected),
        *(row["route_sha256"] for row in other),
    }
    full_hashes = {
        *(row["full_task_sha256"] for row in protected),
        *(row["full_task_sha256"] for row in other),
    }
    seed = start
    while len(accepted) < count:
        if seed in PAPER_SEEDS:
            raise RuntimeError("Paper candidate reached before content construction")
        track = track_from_seed(seed)
        fingerprint = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        reasons = []
        if seed in numeric:
            reasons.append("NUMERIC_RESERVED")
        if fingerprint.route_sha256 in route_hashes:
            reasons.append("PROTECTED_OR_INTERNAL_ROUTE")
        if fingerprint.full_task_sha256 in full_hashes:
            reasons.append("PROTECTED_OR_INTERNAL_FULL_TASK")
        row = {
            "candidate_index": seed - start,
            "numeric_seed": seed,
            "route_sha256": fingerprint.route_sha256,
            "full_task_sha256": fingerprint.full_task_sha256,
            "canonical_task_serialization": canonical_json(
                full_task_payload(track, v2_semantics())
            ),
            "accepted": not reasons,
            "rejection_reasons": reasons,
        }
        scanned.append(row)
        if not reasons:
            accepted.append(row)
            numeric.add(seed)
            route_hashes.add(fingerprint.route_sha256)
            full_hashes.add(fingerprint.full_task_sha256)
        seed += 1
    frozen = [
        {
            key: row[key]
            for key in (
                "numeric_seed",
                "route_sha256",
                "full_task_sha256",
                "canonical_task_serialization",
            )
        }
        for row in accepted
    ]
    return {
        "id": split_id,
        "candidate_start": start,
        "selection_order": "ascending numeric seed",
        "selection_inputs": "numeric seed plus route/full-task hashes only",
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(frozen),
    }


def build_split_ledger() -> dict[str, Any]:
    protected = protected_rows()
    development = _select(
        DEV_ID, DEV_START, DEV_COUNT, protected=protected, other=[]
    )
    holdout = _select(
        HOLDOUT_ID,
        HOLDOUT_START,
        HOLDOUT_COUNT,
        protected=protected,
        other=development["accepted_tasks"],
    )
    dev = development["accepted_tasks"]
    hold = holdout["accepted_tasks"]
    protected_routes = {row["route_sha256"] for row in protected}
    protected_full = {row["full_task_sha256"] for row in protected}
    checks = {
        "development_count_18": len(dev) == DEV_COUNT,
        "holdout_count_36": len(hold) == HOLDOUT_COUNT,
        "development_numeric_unique": len({row["numeric_seed"] for row in dev})
        == DEV_COUNT,
        "development_route_unique": len({row["route_sha256"] for row in dev})
        == DEV_COUNT,
        "development_full_task_unique": len(
            {row["full_task_sha256"] for row in dev}
        )
        == DEV_COUNT,
        "holdout_numeric_unique": len({row["numeric_seed"] for row in hold})
        == HOLDOUT_COUNT,
        "holdout_route_unique": len({row["route_sha256"] for row in hold})
        == HOLDOUT_COUNT,
        "holdout_full_task_unique": len(
            {row["full_task_sha256"] for row in hold}
        )
        == HOLDOUT_COUNT,
        "development_holdout_numeric_disjoint": not (
            {row["numeric_seed"] for row in dev}
            & {row["numeric_seed"] for row in hold}
        ),
        "development_holdout_route_disjoint": not (
            {row["route_sha256"] for row in dev}
            & {row["route_sha256"] for row in hold}
        ),
        "development_holdout_full_task_disjoint": not (
            {row["full_task_sha256"] for row in dev}
            & {row["full_task_sha256"] for row in hold}
        ),
        "new_routes_exclude_prior_training_and_opened_evaluation": all(
            row["route_sha256"] not in protected_routes for row in [*dev, *hold]
        ),
        "new_full_tasks_exclude_prior_training_and_opened_evaluation": all(
            row["full_task_sha256"] not in protected_full
            for row in [*dev, *hold]
        ),
        "paper_numeric_seeds_absent": not (
            {row["numeric_seed"] for row in [*dev, *hold]} & set(PAPER_SEEDS)
        ),
    }
    return {
        "schema_version": 1,
        "construction_order": [DEV_ID, HOLDOUT_ID],
        "protected_prior_row_count": len(protected),
        "evaluation_sets": {"development": development, "holdout": holdout},
        "evaluation_sha256": {
            "development": development["ordered_sha256"],
            "holdout": holdout["ordered_sha256"],
        },
        "checks": checks,
        "route_clean_verified": all(checks.values()),
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(PAPER_SEEDS),
            "content_fingerprints_constructed": 0,
        },
    }


def verify_runtime_row(row: dict[str, Any]) -> bool:
    seed = int(row["numeric_seed"])
    if seed in PAPER_SEEDS:
        raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
    fingerprint = fingerprint_track(
        track_from_seed(seed), v2_semantics(), numeric_seed=seed
    )
    if fingerprint.route_sha256 != row["route_sha256"]:
        raise RuntimeError("runtime route differs from frozen split")
    if fingerprint.full_task_sha256 != row["full_task_sha256"]:
        raise RuntimeError("runtime full task differs from frozen split")
    return True


def physical_fixtures() -> dict[str, Track]:
    def kmh(values: list[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) / 3.6

    return {
        "constant_20_km_h": Track(np.array([0.0]), kmh([20])),
        "constant_100_km_h": Track(np.array([0.0]), kmh([100])),
        "upward_changes": Track(np.array([0.0, 300.0, 600.0]), kmh([20, 60, 100])),
        "maximum_legal_downward_change": Track(
            np.array([0.0, 500.0]), kmh([100, 60])
        ),
        "consecutive_downward_changes": Track(
            np.array([0.0, 500.0, 600.0]), kmh([100, 60, 20])
        ),
        "short_braking_window": Track(
            np.array([0.0, 100.0]), kmh([100, 60])
        ),
        "route_endpoint_at_speed_boundary": Track(
            np.array([0.0, 1000.0]), kmh([60, 20])
        ),
        "insufficient_braking": Track(np.array([0.0, 50.0]), kmh([100, 20])),
        "forced_standstill": Track(np.array([0.0]), kmh([20])),
        "apparently_tight_deadline": Track(
            np.array([0.0, 500.0]), kmh([20, 30])
        ),
    }


def source_hashes() -> dict[str, str]:
    names = (
        "feasibility.py",
        "oracle_controller.py",
        "causal_controller.py",
        "evaluation.py",
        "diagnostics.py",
        "runner.py",
    )
    result = {}
    for name in names:
        path = HERE / name
        if path.exists():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def preflight_benchmark(seeds: tuple[int, ...]) -> dict[str, Any]:
    started = time.perf_counter()
    episodes = []
    for seed in seeds:
        row = rollout(seed, "oracle")
        row.pop("actions")
        episodes.append(row)
    elapsed = time.perf_counter() - started
    per_task = elapsed / len(seeds)
    projected_tasks = DEV_COUNT + HOLDOUT_COUNT
    projected_seconds = per_task * projected_tasks * 12
    # Twelve accounts for ten refinement rollouts, initial witness and replay.
    projected_artifacts = projected_tasks * 2_000 * 32
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "benchmark_seeds": list(seeds),
        "episodes": episodes,
        "wall_time_s": elapsed,
        "mean_time_per_simulated_trajectory_s": per_task,
        "projected_total_physical_reference_execution_s": projected_seconds,
        "projected_total_physical_reference_execution_hours": projected_seconds
        / 3600.0,
        "projected_retained_artifacts_bytes": projected_artifacts,
        "projected_retained_artifacts_gb": projected_artifacts / 1e9,
        "maximum_rss_platform_units": maximum_rss,
        "execution_gate_passed": projected_seconds <= 12 * 3600,
        "artifact_gate_passed": projected_artifacts <= 12 * 1_000_000_000,
        "checkpoint_overhead": "none; deterministic controllers have no checkpoints",
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
