"""Performance-blind construction of fresh route-clean option panels."""

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
PREFERRED_BLOCKS = (10_700_000, 10_800_000)


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
    sources = []
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
    return list(rows.values()), sources


def _choose_blocks(protected: list[dict[str, Any]]) -> tuple[int, int]:
    used_seeds = {
        int(row["numeric_seed"])
        for row in protected
        if row.get("numeric_seed") is not None
    }

    def next_unused(preferred: int) -> int:
        candidate = preferred
        while any(candidate <= seed < candidate + 100_000 for seed in used_seeds):
            candidate += 1_000_000
        return candidate

    return tuple(next_unused(preferred) for preferred in PREFERRED_BLOCKS)


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
    seed = candidate_start
    while len(accepted) < accepted_count:
        if seed in PAPER_SEEDS:
            raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
        track = track_from_seed(seed)
        fingerprint = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        reasons = []
        if seed in numeric:
            reasons.append("NUMERIC_RESERVED")
        if fingerprint.route_sha256 in routes:
            reasons.append("POLICY_R_ROUTE_RESERVED")
        if fingerprint.full_task_sha256 in tasks:
            reasons.append("POLICY_R_FULL_TASK_RESERVED")
        lower = nominal_speed_limit_lower_bound_s(track)
        classification = "PROVEN_INFEASIBLE" if lower > 140.0 else "UNRESOLVED"
        row = {
            "candidate_index": seed - candidate_start,
            "numeric_seed": seed,
            "route_sha256": fingerprint.route_sha256,
            "full_task_sha256": fingerprint.full_task_sha256,
            "canonical_task_serialization": canonical_json(
                full_task_payload(track, v2_semantics())
            ),
            "nominal_speed_limit_time_lower_bound_s": lower,
            "feasibility_classification_at_selection": classification,
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
        "state": "FROZEN_UNOPENED" if "CAPABILITY" in split_id else "FROZEN",
        "candidate_start": candidate_start,
        "selection_order": "ascending numeric seed; first passing Policy R",
        "selection_inputs": "numeric seed and fingerprints only; no controller outcome",
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(projection),
        "performance_episode_count": 0,
    }


def build_split_ledger() -> dict[str, Any]:
    protected, sources = _protected_rows()
    training = [
        {
            "numeric_seed": int(row["reset_seed"]),
            "route_sha256": row["route_sha256"],
            "full_task_sha256": row["full_task_sha256"],
        }
        for stream in training_streams().values()
        for row in stream["accepted_tasks"]
    ]
    protected_by_key = {
        (int(row["numeric_seed"]), row["route_sha256"], row["full_task_sha256"]): row
        for row in (*protected, *training)
    }
    protected = list(protected_by_key.values())
    calibration_block, capability_block = _choose_blocks(protected)
    calibration = _select(
        split_id="OPTIONS_CALIBRATION_V1",
        candidate_start=calibration_block,
        accepted_count=12,
        protected=protected,
        earlier=[],
    )
    capability = _select(
        split_id="OPTIONS_CAPABILITY_V1",
        candidate_start=capability_block,
        accepted_count=18,
        protected=protected,
        earlier=calibration["accepted_tasks"],
    )
    rows = calibration["accepted_tasks"] + capability["accepted_tasks"]
    checks = {
        "calibration_count_12": len(calibration["accepted_tasks"]) == 12,
        "capability_count_18": len(capability["accepted_tasks"]) == 18,
        "training_rows_2595": len(training) == 2_595,
        "all_numeric_unique": len({row["numeric_seed"] for row in rows}) == 30,
        "all_routes_unique": len({row["route_sha256"] for row in rows}) == 30,
        "all_full_tasks_unique": len({row["full_task_sha256"] for row in rows}) == 30,
        "protected_route_disjoint": not (
            {row["route_sha256"] for row in protected}
            & {row["route_sha256"] for row in rows}
        ),
        "protected_full_task_disjoint": not (
            {row["full_task_sha256"] for row in protected}
            & {row["full_task_sha256"] for row in rows}
        ),
        "paper_numeric_absent": not (
            {row["numeric_seed"] for row in rows} & set(PAPER_SEEDS)
        ),
    }
    ledger = {
        "schema_version": 1,
        "policy": "Route-clean Policy R; performance blind",
        "provenance": {
            "protected_unique_rows": len(protected),
            "protected_sources": sources,
            "canonical_training_rows": len(training),
        },
        "namespace": {
            "preferred_blocks": list(PREFERRED_BLOCKS),
            "selected_blocks": [calibration_block, capability_block],
            "preferred_available": [calibration_block, capability_block]
            == list(PREFERRED_BLOCKS),
        },
        "calibration_panel": calibration,
        "capability_panel": capability,
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
        raise RuntimeError("temporal-option panels failed Policy R")
    return ledger


def render_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# Temporal Options V1 Split Ledger",
        "",
        "Both panels were frozen before performance evaluation using route-clean",
        "Policy R and ascending numeric-seed order.",
    ]
    for label, key in (
        ("Calibration", "calibration_panel"),
        ("Capability", "capability_panel"),
    ):
        panel = ledger[key]
        lines.extend(
            [
                "",
                f"## {label}: {panel['id']}",
                "",
                (
                    f"State: `{panel['state']}`. Ordered SHA-256: "
                    f"`{panel['ordered_sha256']}`."
                ),
                "",
                (
                    "| Seed | Route SHA-256 | Full-task SHA-256 | "
                    "Selection classification |"
                ),
                "| ---: | --- | --- | --- |",
            ]
        )
        for row in panel["accepted_tasks"]:
            lines.append(
                f"| {row['numeric_seed']} | `{row['route_sha256']}` | "
                f"`{row['full_task_sha256']}` | "
                f"{row['feasibility_classification_at_selection']} |"
            )
    lines.extend(
        [
            "",
            "All 2,595 reachable canonical training rows and all legitimately",
            "available historical panel fingerprints were protected. Paper seeds",
            "4000–4017 were excluded numerically and never instantiated.",
            "",
        ]
    )
    return "\n".join(lines)


def materialize() -> dict[str, Any]:
    ledger = build_split_ledger()
    (HERE / "split_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (HERE / "SPLIT_LEDGER.md").write_text(render_markdown(ledger), encoding="utf-8")
    return ledger


if __name__ == "__main__":
    materialize()
