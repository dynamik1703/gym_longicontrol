"""Performance-blind Policy-R panels for Feedback Chunk RL V1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "canonical.json"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        for row in found:
            key = (
                row["numeric_seed"],
                row["route_sha256"],
                row["full_task_sha256"],
            )
            rows[key] = row
    return list(rows.values()), sources


def _select(
    *,
    split_id: str,
    candidate_start: int,
    accepted_count: int,
    protected: list[dict[str, Any]],
    earlier: list[dict[str, Any]],
    state: str,
) -> dict[str, Any]:
    numeric = {
        *(row["numeric_seed"] for row in protected),
        *(row["numeric_seed"] for row in earlier),
        *PAPER_SEEDS,
    }
    routes = {
        *(row["route_sha256"] for row in protected),
        *(row["route_sha256"] for row in earlier),
    }
    full_tasks = {
        *(row["full_task_sha256"] for row in protected),
        *(row["full_task_sha256"] for row in earlier),
    }
    accepted: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    seed = int(candidate_start)
    while len(accepted) < accepted_count:
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
            "candidate_index": seed - candidate_start,
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
    return {
        "id": split_id,
        "state": state,
        "candidate_start": candidate_start,
        "selection_order": "ascending numeric seed; first passing Policy R",
        "selection_inputs": "numeric identity and content fingerprints only",
        "policy": "R: numeric, route and full-task clean",
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(frozen),
        "performance_episode_count": 0,
    }


def build_split_ledger() -> dict[str, Any]:
    config = json.loads(CANONICAL.read_text(encoding="utf-8"))["splits"]
    protected, sources = protected_rows()
    diagnostic = _select(
        split_id=config["diagnostic"]["id"],
        candidate_start=config["diagnostic"]["candidate_start"],
        accepted_count=config["diagnostic"]["accepted_count"],
        protected=protected,
        earlier=[],
        state="FROZEN_UNOPENED",
    )
    validation = _select(
        split_id=config["validation"]["id"],
        candidate_start=config["validation"]["candidate_start"],
        accepted_count=config["validation"]["accepted_count"],
        protected=protected,
        earlier=diagnostic["accepted_tasks"],
        state="FROZEN_UNOPENED",
    )
    rows = diagnostic["accepted_tasks"] + validation["accepted_tasks"]
    protected_numeric = {row["numeric_seed"] for row in protected}
    protected_routes = {row["route_sha256"] for row in protected}
    protected_tasks = {row["full_task_sha256"] for row in protected}
    checks = {
        "diagnostic_count_36": len(diagnostic["accepted_tasks"]) == 36,
        "validation_count_36": len(validation["accepted_tasks"]) == 36,
        "all_numeric_unique": len({row["numeric_seed"] for row in rows}) == 72,
        "all_routes_unique": len({row["route_sha256"] for row in rows}) == 72,
        "all_full_tasks_unique": len({row["full_task_sha256"] for row in rows}) == 72,
        "historical_numeric_disjoint": not (
            {row["numeric_seed"] for row in rows} & protected_numeric
        ),
        "historical_route_disjoint": not (
            {row["route_sha256"] for row in rows} & protected_routes
        ),
        "historical_full_task_disjoint": not (
            {row["full_task_sha256"] for row in rows} & protected_tasks
        ),
        "paper_numeric_absent": not (
            {row["numeric_seed"] for row in rows} & set(PAPER_SEEDS)
        ),
    }
    ledger = {
        "schema_version": 1,
        "policy": "Route-clean Policy R",
        "provenance": {
            "selection": "performance blind",
            "protected_historical_unique_rows": len(protected),
            "protected_sources": sources,
        },
        "diagnostic_panel": diagnostic,
        "validation_panel": validation,
        "checks": checks,
        "route_clean_verified": all(checks.values()),
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(PAPER_SEEDS),
            "content_fingerprints_constructed": 0,
            "episode_count": 0,
        },
    }
    if not ledger["route_clean_verified"]:
        raise RuntimeError("Feedback Chunk RL splits failed Policy R")
    return ledger


def materialize(path: Path | None = None) -> dict[str, Any]:
    ledger = build_split_ledger()
    destination = path or HERE / "split_ledger.json"
    destination.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ledger


def verify_runtime_row(row: dict[str, Any]) -> None:
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


if __name__ == "__main__":
    materialize()
