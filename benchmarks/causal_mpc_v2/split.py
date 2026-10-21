"""Performance-blind construction of the Causal MPC V2 panels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLIT_SPECS = {
    "calibration": ("MPC_V2_CALIBRATION", 10_500_000, 12),
    "capability": ("MPC_V2_CAPABILITY", 10_600_000, 18),
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ledger_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(ROOT.glob("benchmarks/**/*ledger*.json"))
        if HERE not in path.parents
    )


def _walk_rows(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        numeric = value.get("numeric_seed", value.get("reset_seed"))
        route = value.get("route_sha256")
        full = value.get("full_task_sha256")
        if numeric is not None and route is not None and full is not None:
            # Scanned-but-rejected candidates did not enter a historical panel.
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
    for path in _ledger_files():
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = list(_walk_rows(payload))
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


def nominal_speed_limit_lower_bound_s(track: Any) -> float:
    endpoints = np.r_[track.positions_m[1:], 1_000.0]
    lengths = endpoints - track.positions_m
    return float(np.sum(lengths / track.limits_m_s))


def _select(
    *,
    split_id: str,
    candidate_start: int,
    accepted_count: int,
    protected: list[dict[str, Any]],
    earlier: list[dict[str, Any]],
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
        lower_bound = nominal_speed_limit_lower_bound_s(track)
        classification = (
            "PROVEN_INFEASIBLE_NOMINAL_LIMIT_LOWER_BOUND"
            if lower_bound > 140.0
            else "UNRESOLVED"
        )
        row = {
            "candidate_index": seed - candidate_start,
            "numeric_seed": seed,
            "route_sha256": fingerprint.route_sha256,
            "full_task_sha256": fingerprint.full_task_sha256,
            "canonical_task_serialization": canonical_json(
                full_task_payload(track, v2_semantics())
            ),
            "nominal_speed_limit_time_lower_bound_s": lower_bound,
            "feasibility_classification_at_selection": classification,
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
        "state": "FROZEN_UNOPENED" if "CAPABILITY" in split_id else "FROZEN",
        "candidate_start": candidate_start,
        "selection_order": "ascending numeric seed; first passing Policy R",
        "selection_inputs": "numeric seed and fingerprints only; no performance",
        "policy": "R: numeric, route and full-task clean",
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(frozen),
        "performance_episode_count": 0,
    }


def build_split_ledger() -> dict[str, Any]:
    protected, sources = protected_rows()
    calibration = _select(
        split_id=SPLIT_SPECS["calibration"][0],
        candidate_start=SPLIT_SPECS["calibration"][1],
        accepted_count=SPLIT_SPECS["calibration"][2],
        protected=protected,
        earlier=[],
    )
    capability = _select(
        split_id=SPLIT_SPECS["capability"][0],
        candidate_start=SPLIT_SPECS["capability"][1],
        accepted_count=SPLIT_SPECS["capability"][2],
        protected=protected,
        earlier=calibration["accepted_tasks"],
    )
    rows = calibration["accepted_tasks"] + capability["accepted_tasks"]
    protected_numeric = {row["numeric_seed"] for row in protected}
    protected_routes = {row["route_sha256"] for row in protected}
    protected_tasks = {row["full_task_sha256"] for row in protected}
    checks = {
        "calibration_count_12": len(calibration["accepted_tasks"]) == 12,
        "capability_count_18": len(capability["accepted_tasks"]) == 18,
        "all_numeric_unique": len({row["numeric_seed"] for row in rows}) == 30,
        "all_routes_unique": len({row["route_sha256"] for row in rows}) == 30,
        "all_full_tasks_unique": len({row["full_task_sha256"] for row in rows})
        == 30,
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
        "calibration_panel": calibration,
        "capability_panel": capability,
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
        raise RuntimeError("Causal MPC V2 splits failed Policy R")
    return ledger


def materialize(path: Path | None = None) -> dict[str, Any]:
    destination = path or HERE / "split_ledger.json"
    ledger = build_split_ledger()
    destination.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ledger


def render_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# Causal MPC V2 Split Ledger",
        "",
        "Both panels were selected before controller evaluation, in ascending",
        "numeric-seed order, using only numeric and canonical content fingerprints.",
    ]
    for label, key in (("Calibration", "calibration_panel"), ("Capability", "capability_panel")):
        split = ledger[key]
        lines.extend(
            [
                "",
                f"## {label}: {split['id']}",
                "",
                f"State: `{split['state']}`. Ordered SHA-256: `{split['ordered_sha256']}`.",
                "",
                "| Seed | Route SHA-256 | Full-task SHA-256 | Selection metadata |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for row in split["accepted_tasks"]:
            lines.append(
                f"| {row['numeric_seed']} | `{row['route_sha256']}` | "
                f"`{row['full_task_sha256']}` | "
                f"{row['feasibility_classification_at_selection']} |"
            )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"Protected unique historical rows: {ledger['provenance']['protected_historical_unique_rows']}.",
            "All checks pass. Paper seeds 4000–4017 remain `SEALED_UNINSPECTED`;",
            "they were excluded numerically and never instantiated or fingerprinted.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    frozen = materialize()
    (HERE / "SPLIT_LEDGER.md").write_text(
        render_markdown(frozen), encoding="utf-8"
    )

