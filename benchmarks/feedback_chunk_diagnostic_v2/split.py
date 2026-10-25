"""Performance-blind route-clean panel selection for Diagnostic V2."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed

from .constants import (
    CANDIDATE_START,
    HERE,
    PANEL_ID,
    ROOT,
    SPLIT_LEDGER,
    TASK_COUNT,
)
from .utils import canonical_sha256, file_sha256


def _walk_rows(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        numeric = value.get("numeric_seed", value.get("reset_seed"))
        route = value.get("route_sha256")
        full = value.get("full_task_sha256")
        if numeric is not None and route is not None and full is not None:
            if value.get("accepted", True):
                yield {
                    "numeric_seed": int(numeric),
                    "route_sha256": str(route),
                    "full_task_sha256": str(full),
                }
        for child in value.values():
            yield from _walk_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_rows(child)


def protected_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: dict[tuple[int, str, str], dict[str, Any]] = {}
    sources: list[dict[str, str]] = []
    for path in sorted(ROOT.glob("benchmarks/**/*ledger*.json")):
        if HERE in path.parents:
            continue
        found = list(_walk_rows(json.loads(path.read_text(encoding="utf-8"))))
        if found:
            sources.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": file_sha256(path),
                }
            )
        for row in found:
            rows[
                (row["numeric_seed"], row["route_sha256"], row["full_task_sha256"])
            ] = row
    return list(rows.values()), sources


def verify_namespace_unused() -> dict[str, Any]:
    collisions: list[str] = []
    for path in sorted(ROOT.glob("benchmarks/**/*.json")):
        if HERE in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        if PANEL_ID in text:
            collisions.append(str(path.relative_to(ROOT)))
    if collisions:
        raise RuntimeError(f"Diagnostic V2 namespace is already used: {collisions}")
    protected, _ = protected_rows()
    numeric_collision = any(row["numeric_seed"] == CANDIDATE_START for row in protected)
    return {
        "panel_id": PANEL_ID,
        "candidate_start": CANDIDATE_START,
        "id_collision_paths": collisions,
        "candidate_start_already_reserved": numeric_collision,
        "unused": not collisions and not numeric_collision,
    }


def build_split_ledger() -> dict[str, Any]:
    namespace = verify_namespace_unused()
    if not namespace["unused"]:
        raise RuntimeError("proposed V2 namespace is occupied")
    protected, sources = protected_rows()
    numeric = {row["numeric_seed"] for row in protected} | set(PAPER_SEEDS)
    routes = {row["route_sha256"] for row in protected}
    full_tasks = {row["full_task_sha256"] for row in protected}
    accepted: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    seed = CANDIDATE_START
    while len(accepted) < TASK_COUNT:
        if seed in PAPER_SEEDS:
            raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
        track = track_from_seed(seed)
        fingerprint = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        reasons: list[str] = []
        if seed in numeric:
            reasons.append("NUMERIC_RESERVED")
        if fingerprint.route_sha256 in routes:
            reasons.append("POLICY_R_ROUTE_RESERVED")
        if fingerprint.full_task_sha256 in full_tasks:
            reasons.append("POLICY_R_FULL_TASK_RESERVED")
        row = {
            "candidate_index": seed - CANDIDATE_START,
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
            routes.add(fingerprint.route_sha256)
            full_tasks.add(fingerprint.full_task_sha256)
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
    protected_numeric = {row["numeric_seed"] for row in protected}
    protected_routes = {row["route_sha256"] for row in protected}
    protected_full = {row["full_task_sha256"] for row in protected}
    checks = {
        "accepted_count_36": len(accepted) == TASK_COUNT,
        "numeric_unique": len({row["numeric_seed"] for row in accepted}) == TASK_COUNT,
        "route_unique": len({row["route_sha256"] for row in accepted}) == TASK_COUNT,
        "full_task_unique": len({row["full_task_sha256"] for row in accepted})
        == TASK_COUNT,
        "historical_numeric_disjoint": not (
            {row["numeric_seed"] for row in accepted} & protected_numeric
        ),
        "historical_route_disjoint": not (
            {row["route_sha256"] for row in accepted} & protected_routes
        ),
        "historical_full_task_disjoint": not (
            {row["full_task_sha256"] for row in accepted} & protected_full
        ),
        "paper_numeric_absent": not (
            {row["numeric_seed"] for row in accepted} & set(PAPER_SEEDS)
        ),
    }
    ledger = {
        "schema_version": 1,
        "panel": {
            "id": PANEL_ID,
            "state": "FROZEN_UNOPENED",
            "candidate_start": CANDIDATE_START,
            "accepted_count": TASK_COUNT,
            "selection_order": "ascending numeric seed; first passing Policy R",
            "selection_inputs": "numeric identity and content fingerprints only",
            "accepted_tasks": accepted,
            "scanned_candidates": scanned,
            "rejected_candidates": [row for row in scanned if not row["accepted"]],
            "ordered_sha256": canonical_sha256(frozen),
            "performance_episode_count": 0,
        },
        "namespace_audit": namespace,
        "protected_historical_unique_rows": len(protected),
        "protected_sources": sources,
        "checks": checks,
        "route_clean_verified": all(checks.values()),
        "validation": {"state": "FROZEN_UNOPENED", "performance_episode_count": 0},
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(PAPER_SEEDS),
            "content_fingerprints_constructed": 0,
            "performance_episode_count": 0,
        },
    }
    if not ledger["route_clean_verified"]:
        raise RuntimeError("Diagnostic V2 split failed Policy R")
    return ledger


def verify_task_row(row: dict[str, Any]) -> None:
    seed = int(row["numeric_seed"])
    if seed in PAPER_SEEDS:
        raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
    fingerprint = fingerprint_track(
        track_from_seed(seed), v2_semantics(), numeric_seed=seed
    )
    if fingerprint.route_sha256 != row["route_sha256"]:
        raise RuntimeError("runtime route fingerprint changed")
    if fingerprint.full_task_sha256 != row["full_task_sha256"]:
        raise RuntimeError("runtime full-task fingerprint changed")


def materialize() -> dict[str, Any]:
    from .utils import atomic_json

    ledger = build_split_ledger()
    atomic_json(SPLIT_LEDGER, ledger)
    return ledger
