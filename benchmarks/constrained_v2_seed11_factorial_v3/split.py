"""Deterministic V3 route-clean diagnostic selection and split ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import (
    continuous_tracks,
    explicit_seed_stream,
    fingerprint_rows,
    track_from_seed,
)

from .config import EXPECTED_STREAM_FINGERPRINTS, load_configuration

TRAINING_SEED = 11
CURRENT_DOMAIN_TAG = 0x4D564531
DEVELOPMENT_SEEDS = tuple(range(2000, 2009))
VALIDATION_SEEDS = tuple(range(3000, 3009))
PAPER_SEEDS = tuple(range(4000, 4018))
OLD_DIAGNOSTIC_SEEDS = tuple(range(9_000_000, 9_000_018))
V2_FAILED_DIAGNOSTIC_SEED = 9_000_005


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reachable_horizon() -> dict[str, int]:
    configuration = load_configuration().raw
    budget = int(configuration["real_transition_budget"])
    minimum = int(configuration["minimum_physically_possible_episode_length"])
    completed = budget // minimum
    return {
        "real_transition_budget": budget,
        "minimum_physically_possible_episode_length": minimum,
        "max_completed_episodes": completed,
        "max_exposed_tasks": completed + 1,
    }


@lru_cache(maxsize=1)
def stream_rows() -> dict[str, list[dict[str, Any]]]:
    count = reachable_horizon()["max_exposed_tasks"]
    semantics = v2_semantics()
    historical_tracks = continuous_tracks(
        TRAINING_SEED, list(range(0, 2 * count, 2))
    )
    historical = fingerprint_rows(
        historical_tracks,
        semantics,
        numeric_seeds=[None] * count,
        draw_indices=list(range(0, 2 * count, 2)),
    )
    current_seeds = explicit_seed_stream(
        TRAINING_SEED, domain_tag=CURRENT_DOMAIN_TAG, count=count
    )
    current = fingerprint_rows(
        [track_from_seed(seed) for seed in current_seeds],
        semantics,
        numeric_seeds=current_seeds,
        draw_indices=list(range(count)),
    )
    return {"T_HIST": historical, "T_CURRENT": current}


def stream_fingerprints() -> dict[str, str]:
    return {name: canonical_sha256(rows) for name, rows in stream_rows().items()}


@lru_cache(maxsize=1)
def opened_split_rows() -> dict[str, list[dict[str, Any]]]:
    semantics = v2_semantics()
    result = {}
    for name, seeds in (
        ("Development", DEVELOPMENT_SEEDS),
        ("Validation", VALIDATION_SEEDS),
    ):
        result[name] = fingerprint_rows(
            [track_from_seed(seed) for seed in seeds],
            semantics,
            numeric_seeds=list(seeds),
            draw_indices=[None] * len(seeds),
        )
    return result


def _lookup(rows: Sequence[Mapping[str, Any]], field: str) -> set[Any]:
    return {row[field] for row in rows}


def _candidate(
    seed: int,
    *,
    accepted: Sequence[Mapping[str, Any]],
    streams: Mapping[str, Sequence[Mapping[str, Any]]],
    opened: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if seed in PAPER_SEEDS:
        raise RuntimeError("Paper candidates are fail-closed")
    semantics = v2_semantics()
    track = track_from_seed(seed)
    fingerprint = fingerprint_track(track, semantics, numeric_seed=seed)
    numeric_exclusions = {
        *DEVELOPMENT_SEEDS,
        *VALIDATION_SEEDS,
        *PAPER_SEEDS,
        *OLD_DIAGNOSTIC_SEEDS,
        *(
            int(row["reset_seed"])
            for row in streams["T_CURRENT"]
            if row["reset_seed"] is not None
        ),
    }
    reasons = []
    if seed in numeric_exclusions:
        reasons.append("NUMERIC_SEPARATION")
    for stream in ("T_HIST", "T_CURRENT"):
        if fingerprint.route_sha256 in _lookup(streams[stream], "route_sha256"):
            reasons.append(f"{stream}_ROUTE")
        if fingerprint.full_task_sha256 in _lookup(
            streams[stream], "full_task_sha256"
        ):
            reasons.append(f"{stream}_FULL_TASK")
    for split in ("Development", "Validation"):
        if fingerprint.route_sha256 in _lookup(opened[split], "route_sha256"):
            reasons.append(f"{split.upper()}_ROUTE")
        if fingerprint.full_task_sha256 in _lookup(
            opened[split], "full_task_sha256"
        ):
            reasons.append(f"{split.upper()}_FULL_TASK")
    if seed in _lookup(accepted, "numeric_seed"):
        reasons.append("INTERNAL_NUMERIC_DUPLICATE")
    if fingerprint.route_sha256 in _lookup(accepted, "route_sha256"):
        reasons.append("INTERNAL_ROUTE_DUPLICATE")
    if fingerprint.full_task_sha256 in _lookup(accepted, "full_task_sha256"):
        reasons.append("INTERNAL_FULL_TASK_DUPLICATE")
    return {
        "candidate_index": seed - 9_100_000,
        "numeric_seed": seed,
        "route_sha256": fingerprint.route_sha256,
        "full_task_sha256": fingerprint.full_task_sha256,
        "canonical_task_serialization": canonical_json(
            full_task_payload(track, semantics)
        ),
        "accepted": not reasons,
        "rejection_reasons": reasons,
    }


@lru_cache(maxsize=1)
def build_split_ledger() -> dict[str, Any]:
    configuration = load_configuration()
    streams = stream_rows()
    fingerprints = stream_fingerprints()
    if fingerprints != EXPECTED_STREAM_FINGERPRINTS:
        raise RuntimeError("Reachable training-stream fingerprints changed")
    opened = opened_split_rows()
    accepted: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    seed = int(configuration.raw["diagnostic_selection"]["candidate_start"])
    target = int(configuration.raw["diagnostic_selection"]["accepted_count"])
    while len(accepted) < target:
        row = _candidate(seed, accepted=accepted, streams=streams, opened=opened)
        scanned.append(row)
        if row["accepted"]:
            accepted.append(row)
        seed += 1
    ordered_payload = [
        {
            "numeric_seed": row["numeric_seed"],
            "route_sha256": row["route_sha256"],
            "full_task_sha256": row["full_task_sha256"],
            "canonical_task_serialization": row["canonical_task_serialization"],
        }
        for row in accepted
    ]
    return {
        "schema_version": 1,
        "candidate_scan_start": configuration.raw["diagnostic_selection"][
            "candidate_start"
        ],
        "selection_rule": configuration.raw["diagnostic_selection"]["policy"],
        "reachable_horizon": reachable_horizon(),
        "stream_fingerprints": fingerprints,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "accepted_tasks": accepted,
        "ordered_diagnostic_set_sha256": canonical_sha256(ordered_payload),
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "content_fingerprints_constructed": 0,
            "numeric_exclusion_only": list(PAPER_SEEDS),
        },
    }


def diagnostic_seeds() -> tuple[int, ...]:
    return tuple(
        int(row["numeric_seed"]) for row in build_split_ledger()["accepted_tasks"]
    )


def v2_failure_regression() -> dict[str, Any]:
    return _candidate(
        V2_FAILED_DIAGNOSTIC_SEED,
        accepted=[],
        streams=stream_rows(),
        opened=opened_split_rows(),
    )


def write_ledger(path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps(build_split_ledger(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
