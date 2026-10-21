"""Performance-blind route-clean panels for Counterfactual Option Value V1."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed
from benchmarks.hybrid_rl_design.split import training_streams

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFERRED_BLOCKS = (10_900_000, 11_000_000, 11_100_000)
BLOCK_SCAN_WIDTH = 100_000
PANEL_SPECS = (
    ("fit_panel", "CF_OPTION_FIT_V1", 16),
    ("calibration_panel", "CF_OPTION_CALIBRATION_V1", 8),
    ("test_panel", "CF_OPTION_TEST_V1", 12),
)


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
        if (
            numeric is not None
            and route is not None
            and full is not None
            and value.get("accepted", True)
        ):
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


def _protected_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
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
            rows[
                (row["numeric_seed"], row["route_sha256"], row["full_task_sha256"])
            ] = row
    training = [
        {
            "numeric_seed": int(row["reset_seed"]),
            "route_sha256": row["route_sha256"],
            "full_task_sha256": row["full_task_sha256"],
        }
        for stream in training_streams().values()
        for row in stream["accepted_tasks"]
    ]
    for row in training:
        rows[(row["numeric_seed"], row["route_sha256"], row["full_task_sha256"])] = row
    return list(rows.values()), sources


def _choose_blocks(
    protected: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]]]:
    used = {int(row["numeric_seed"]) for row in protected}
    chosen: list[int] = []
    collisions: list[dict[str, Any]] = []
    for preferred in PREFERRED_BLOCKS:
        candidate = preferred
        while colliding := sorted(
            seed for seed in used if candidate <= seed < candidate + BLOCK_SCAN_WIDTH
        ):
            collisions.append(
                {
                    "preferred": preferred,
                    "colliding_block": candidate,
                    "colliding_numeric_seeds": colliding,
                    "rule": "advance by 1,000,000",
                }
            )
            candidate += 1_000_000
        chosen.append(candidate)
        used.update(range(candidate, candidate + BLOCK_SCAN_WIDTH))
    return chosen, collisions


def nominal_speed_limit_lower_bound_s(track: Any) -> float:
    endpoints = np.r_[track.positions_m[1:], 1_000.0]
    return float(np.sum((endpoints - track.positions_m) / track.limits_m_s))


def _select(
    *,
    split_id: str,
    candidate_start: int,
    accepted_count: int,
    protected: list[dict[str, Any]],
    earlier: list[dict[str, Any]],
) -> dict[str, Any]:
    numeric = {
        *(int(row["numeric_seed"]) for row in protected),
        *(int(row["numeric_seed"]) for row in earlier),
        *PAPER_SEEDS,
    }
    routes = {
        *(row["route_sha256"] for row in protected),
        *(row["route_sha256"] for row in earlier),
    }
    tasks = {
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
        if fingerprint.full_task_sha256 in tasks:
            reasons.append("POLICY_R_FULL_TASK_RESERVED")
        lower = nominal_speed_limit_lower_bound_s(track)
        row = {
            "candidate_index": seed - candidate_start,
            "numeric_seed": seed,
            "route_sha256": fingerprint.route_sha256,
            "full_task_sha256": fingerprint.full_task_sha256,
            "canonical_task_serialization": canonical_json(
                full_task_payload(track, v2_semantics())
            ),
            "nominal_speed_limit_time_lower_bound_s": lower,
            "feasibility_classification_at_selection": (
                "PROVEN_INFEASIBLE_NOMINAL_LIMIT_LOWER_BOUND"
                if lower > 140.0
                else "UNRESOLVED"
            ),
            "accepted": not reasons,
            "rejection_reasons": reasons,
        }
        scanned.append(row)
        if not reasons:
            accepted.append(row)
            numeric.add(seed)
            routes.add(fingerprint.route_sha256)
            tasks.add(fingerprint.full_task_sha256)
        seed += 1
    projection = [
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
        "state": "FROZEN_UNOPENED",
        "candidate_start": candidate_start,
        "selection_order": "ascending numeric seed; first passing Policy R",
        "selection_inputs": "numeric seed and content fingerprints only",
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(projection),
        "performance_episode_count": 0,
        "sampled_snapshot_count": 0,
    }


def build_split_ledger() -> dict[str, Any]:
    protected, sources = _protected_rows()
    blocks, collisions = _choose_blocks(protected)
    panels: dict[str, dict[str, Any]] = {}
    earlier: list[dict[str, Any]] = []
    for (key, split_id, count), block in zip(PANEL_SPECS, blocks, strict=True):
        panel = _select(
            split_id=split_id,
            candidate_start=block,
            accepted_count=count,
            protected=protected,
            earlier=earlier,
        )
        panels[key] = panel
        earlier.extend(panel["accepted_tasks"])
    rows = [row for panel in panels.values() for row in panel["accepted_tasks"]]
    protected_routes = {row["route_sha256"] for row in protected}
    protected_tasks = {row["full_task_sha256"] for row in protected}
    checks = {
        "panel_counts_16_8_12": [
            len(panels[key]["accepted_tasks"]) for key, _, _ in PANEL_SPECS
        ]
        == [16, 8, 12],
        "all_numeric_unique": len({row["numeric_seed"] for row in rows}) == 36,
        "all_routes_unique": len({row["route_sha256"] for row in rows}) == 36,
        "all_full_tasks_unique": len({row["full_task_sha256"] for row in rows}) == 36,
        "protected_route_disjoint": not (
            {row["route_sha256"] for row in rows} & protected_routes
        ),
        "protected_full_task_disjoint": not (
            {row["full_task_sha256"] for row in rows} & protected_tasks
        ),
        "paper_numeric_absent": not (
            {row["numeric_seed"] for row in rows} & set(PAPER_SEEDS)
        ),
    }
    ledger = {
        "schema_version": 1,
        "study": "COUNTERFACTUAL_OPTION_VALUE_V1",
        "policy": "Route-clean Policy R; performance blind; route-group partitions",
        "provenance": {
            "protected_unique_rows": len(protected),
            "protected_sources": sources,
            "canonical_training_rows": 2_595,
        },
        "namespace": {
            "preferred_blocks": list(PREFERRED_BLOCKS),
            "selected_blocks": blocks,
            "collision_log": collisions,
        },
        **panels,
        "checks": checks,
        "route_clean_verified": all(checks.values()),
        "historical_validation_panels_opened": 0,
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(PAPER_SEEDS),
            "content_fingerprints_constructed": 0,
            "episode_count": 0,
        },
    }
    if not ledger["route_clean_verified"]:
        raise RuntimeError("counterfactual panels failed Policy R")
    return ledger


def materialize(path: Path | None = None) -> dict[str, Any]:
    destination = path or HERE / "split_ledger.json"
    ledger = build_split_ledger()
    destination.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ledger


if __name__ == "__main__":
    materialize()
