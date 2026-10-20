"""Deterministic route-clean splits against all legitimately opened content."""

from __future__ import annotations

import hashlib
import json
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
    ALL_NUMERICALLY_RESERVED,
    PAPER_SEEDS,
    track_from_seed,
)

from .config import TRAINING_SEEDS, load_configuration

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent
COLLECTOR_SPLIT = PRIOR / "constrained_v2_collector_replication" / "split_ledger.json"
COLLECTOR_STREAMS = (
    PRIOR / "constrained_v2_collector_replication" / "stream_ledger.json"
)
V3_SPLIT = PRIOR / "constrained_v2_seed11_factorial_v3" / "split_ledger.json"
OLD_DIAGNOSTIC_SEEDS = tuple(range(9_000_000, 9_000_018))


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_seed": int(row["numeric_seed"]),
        "route_sha256": row["route_sha256"],
        "full_task_sha256": row["full_task_sha256"],
    }


@lru_cache(maxsize=1)
def canonical_streams() -> dict[str, dict[str, Any]]:
    payload = json.loads(COLLECTOR_STREAMS.read_text(encoding="utf-8"))
    expected = load_configuration().raw["canonical_stream_sha256"]
    for seed in TRAINING_SEEDS:
        stream = payload[str(seed)]
        if len(stream["accepted_tasks"]) != 865:
            raise RuntimeError("Canonical stream is incomplete")
        if stream["ordered_sha256"] != expected[str(seed)]:
            raise RuntimeError("Canonical stream SHA-256 changed")
    return payload


@lru_cache(maxsize=1)
def opened_split_rows() -> dict[str, list[dict[str, Any]]]:
    semantics = v2_semantics()
    result: dict[str, list[dict[str, Any]]] = {}
    for name, seeds in (
        ("v1_exploratory", tuple(range(1000, 1009))),
        ("legacy_development", tuple(range(2000, 2009))),
        ("legacy_validation", tuple(range(3000, 3009))),
        ("old_diagnostic", OLD_DIAGNOSTIC_SEEDS),
    ):
        result[name] = []
        for seed in seeds:
            fingerprint = fingerprint_track(
                track_from_seed(seed), semantics, numeric_seed=seed
            )
            result[name].append(
                {
                    "numeric_seed": seed,
                    "route_sha256": fingerprint.route_sha256,
                    "full_task_sha256": fingerprint.full_task_sha256,
                }
            )
    v3 = json.loads(V3_SPLIT.read_text(encoding="utf-8"))
    result["v3_diagnostic"] = [_projection(row) for row in v3["accepted_tasks"]]
    collector = json.loads(COLLECTOR_SPLIT.read_text(encoding="utf-8"))
    for name in ("diagnostic", "validation"):
        result[f"collector_{name}"] = [
            _projection(row)
            for row in collector["evaluation_sets"][name]["accepted_tasks"]
        ]
    return result


def _protected_rows() -> list[dict[str, Any]]:
    opened = [row for rows in opened_split_rows().values() for row in rows]
    training = [
        {
            "numeric_seed": int(row["reset_seed"]),
            "route_sha256": row["route_sha256"],
            "full_task_sha256": row["full_task_sha256"],
        }
        for stream in canonical_streams().values()
        for row in stream["accepted_tasks"]
    ]
    return opened + training


def _candidate(
    seed: int,
    *,
    start: int,
    accepted: list[dict[str, Any]],
    other: list[dict[str, Any]],
) -> dict[str, Any]:
    if seed in PAPER_SEEDS:
        raise RuntimeError("Paper candidates are fail-closed")
    protected = _protected_rows()
    prior = [*accepted, *other]
    numeric = {
        *ALL_NUMERICALLY_RESERVED,
        *PAPER_SEEDS,
        *(int(row["numeric_seed"]) for row in protected),
        *(int(row["numeric_seed"]) for row in prior),
    }
    routes = {row["route_sha256"] for row in [*protected, *prior]}
    full = {row["full_task_sha256"] for row in [*protected, *prior]}
    track = track_from_seed(seed)
    fingerprint = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
    reasons = []
    if seed in numeric:
        reasons.append("NUMERIC_RESERVED")
    if fingerprint.route_sha256 in routes:
        reasons.append("PROTECTED_OR_INTERNAL_ROUTE")
    if fingerprint.full_task_sha256 in full:
        reasons.append("PROTECTED_OR_INTERNAL_FULL_TASK")
    return {
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


def _select(name: str, other: list[dict[str, Any]]) -> dict[str, Any]:
    selection = load_configuration().raw["evaluation_selections"][name]
    start = int(selection["candidate_start"])
    target = int(selection["accepted_count"])
    accepted: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    seed = start
    while len(accepted) < target:
        row = _candidate(seed, start=start, accepted=accepted, other=other)
        scanned.append(row)
        if row["accepted"]:
            accepted.append(row)
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
        "id": selection["id"],
        "candidate_start": start,
        "selection_order": "ascending numeric seed",
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(frozen),
    }


@lru_cache(maxsize=1)
def evaluation_sets() -> dict[str, dict[str, Any]]:
    diagnostic = _select("diagnostic", [])
    validation = _select("validation", diagnostic["accepted_tasks"])
    result = {"diagnostic": diagnostic, "validation": validation}
    expected = load_configuration().raw["expected_evaluation_sha256"]
    for name, split in result.items():
        frozen = expected[name]
        if frozen is not None and split["ordered_sha256"] != frozen:
            raise RuntimeError(f"Frozen {name} split changed")
    return result


def build_split_ledger() -> dict[str, Any]:
    splits = evaluation_sets()
    protected = _protected_rows()
    protected_routes = {row["route_sha256"] for row in protected}
    protected_full = {row["full_task_sha256"] for row in protected}
    combined = [row for split in splits.values() for row in split["accepted_tasks"]]
    checks = {
        "both_sets_have_18": all(
            len(split["accepted_tasks"]) == 18 for split in splits.values()
        ),
        "internal_numeric_unique": len({row["numeric_seed"] for row in combined}) == 36,
        "internal_route_unique": len({row["route_sha256"] for row in combined}) == 36,
        "internal_full_task_unique": len({row["full_task_sha256"] for row in combined})
        == 36,
        "protected_route_separation": all(
            row["route_sha256"] not in protected_routes for row in combined
        ),
        "protected_full_task_separation": all(
            row["full_task_sha256"] not in protected_full for row in combined
        ),
        "canonical_streams_have_865": all(
            len(stream["accepted_tasks"]) == 865
            for stream in canonical_streams().values()
        ),
    }
    return {
        "schema_version": 1,
        "construction_order": [
            "RECOVER_EXISTING_CANONICAL_STREAMS",
            "SAMEINFO_DIAGNOSTIC_V1",
            "SAMEINFO_VALIDATION_V1",
        ],
        "opened_sets": {
            name: {
                "count": len(rows),
                "numeric_seeds": [r["numeric_seed"] for r in rows],
            }
            for name, rows in opened_split_rows().items()
        },
        "canonical_stream_sha256": {
            seed: stream["ordered_sha256"]
            for seed, stream in canonical_streams().items()
        },
        "evaluation_sets": splits,
        "checks": checks,
        "route_clean_verified": all(checks.values()),
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(PAPER_SEEDS),
            "content_fingerprints_constructed": 0,
        },
    }


def stream_rows(training_seed: int) -> list[dict[str, Any]]:
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("Unknown canonical training seed")
    return canonical_streams()[str(training_seed)]["accepted_tasks"]


def diagnostic_seeds() -> tuple[int, ...]:
    return tuple(
        int(row["numeric_seed"])
        for row in evaluation_sets()["diagnostic"]["accepted_tasks"]
    )


def validation_seeds() -> tuple[int, ...]:
    return tuple(
        int(row["numeric_seed"])
        for row in evaluation_sets()["validation"]["accepted_tasks"]
    )
