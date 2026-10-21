"""Route-clean Development, Diagnostic and unopened Validation panels."""

from __future__ import annotations

import hashlib
import json
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
from benchmarks.hybrid_rl_validation_v2.split import protected_rows as v2_protected_rows

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HYBRID_V2_LEDGER = ROOT / "benchmarks/hybrid_rl_validation_v2/split_ledger.json"
TRAINING_STREAM_LEDGER = (
    ROOT / "benchmarks/constrained_v2_collector_replication/stream_ledger.json"
)

SPLIT_SPECS = {
    "development": ("SLACKTRADER_MPC_DEVELOPMENT_V1", 10_250_000, 8),
    "diagnostic": ("SLACKTRADER_DIAGNOSTIC_V1", 10_300_000, 18),
    "validation": ("SLACKTRADER_VALIDATION_V1", 10_400_000, 36),
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_seed": int(row.get("numeric_seed", row.get("reset_seed"))),
        "route_sha256": row["route_sha256"],
        "full_task_sha256": row["full_task_sha256"],
    }


def _historical_protected_rows() -> list[dict[str, Any]]:
    prior, _ = v2_protected_rows()
    v2 = json.loads(HYBRID_V2_LEDGER.read_text(encoding="utf-8"))
    rows = [_projection(row) for row in prior]
    rows.extend(_projection(row) for row in v2["accepted_tasks"])
    unique = {
        (row["numeric_seed"], row["route_sha256"], row["full_task_sha256"]): row
        for row in rows
    }
    return list(unique.values())


def nominal_speed_limit_lower_bound_s(track) -> float:
    endpoints = np.r_[track.positions_m[1:], 1_000.0]
    lengths = endpoints - track.positions_m
    return float(np.sum(lengths / track.limits_m_s))


def _select(
    split_id: str,
    candidate_start: int,
    accepted_count: int,
    *,
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
    while len(accepted) < int(accepted_count):
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
            "feasibility_classification": classification,
            "proven_feasible_witness": False,
            "proven_infeasible": classification.startswith("PROVEN_INFEASIBLE"),
            "unresolved": classification == "UNRESOLVED",
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
        "state": "FROZEN_UNOPENED" if "VALIDATION" in split_id else "FROZEN",
        "candidate_start": candidate_start,
        "selection_order": "ascending numeric seed; first passing Policy R",
        "selection_inputs": (
            "numeric seed and fingerprints only; no controller performance"
        ),
        "policy": "R: numeric, route and full-task clean",
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(frozen),
        "performance_episode_count": 0,
    }


def build_split_ledger() -> dict[str, Any]:
    protected = _historical_protected_rows()
    selected: dict[str, dict[str, Any]] = {}
    earlier: list[dict[str, Any]] = []
    for name in ("development", "diagnostic", "validation"):
        split_id, start, count = SPLIT_SPECS[name]
        selected[name] = _select(
            split_id,
            start,
            count,
            protected=protected,
            earlier=earlier,
        )
        earlier.extend(selected[name]["accepted_tasks"])
    training = [
        _projection(row)
        for stream in training_streams().values()
        for row in stream["accepted_tasks"]
    ]
    all_rows = [row for split in selected.values() for row in split["accepted_tasks"]]
    checks = {
        "development_count_8": len(selected["development"]["accepted_tasks"]) == 8,
        "diagnostic_count_18": len(selected["diagnostic"]["accepted_tasks"]) == 18,
        "validation_count_36": len(selected["validation"]["accepted_tasks"]) == 36,
        "all_numeric_unique": len({row["numeric_seed"] for row in all_rows}) == 62,
        "all_routes_unique": len({row["route_sha256"] for row in all_rows}) == 62,
        "all_full_tasks_unique": len({row["full_task_sha256"] for row in all_rows})
        == 62,
        "training_rows_2595": len(training) == 2_595,
        "training_route_disjoint": not (
            {row["route_sha256"] for row in training}
            & {row["route_sha256"] for row in all_rows}
        ),
        "training_full_task_disjoint": not (
            {row["full_task_sha256"] for row in training}
            & {row["full_task_sha256"] for row in all_rows}
        ),
        "historical_route_disjoint": not (
            {row["route_sha256"] for row in protected}
            & {row["route_sha256"] for row in all_rows}
        ),
        "historical_full_task_disjoint": not (
            {row["full_task_sha256"] for row in protected}
            & {row["full_task_sha256"] for row in all_rows}
        ),
        "paper_numeric_absent": not (
            {row["numeric_seed"] for row in all_rows} & set(PAPER_SEEDS)
        ),
    }
    streams = training_streams()
    ledger = {
        "schema_version": 1,
        "policy": "Route-clean Policy R",
        "protected_historical_row_count": len(protected),
        "training": {
            "source": str(TRAINING_STREAM_LEDGER.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(
                TRAINING_STREAM_LEDGER.read_bytes()
            ).hexdigest(),
            "reachable_rows_per_seed": 865,
            "total_reachable_rows": 2_595,
            "streams": {
                seed: {
                    "ordered_sha256": stream["ordered_sha256"],
                    "accepted_task_count": len(stream["accepted_tasks"]),
                }
                for seed, stream in streams.items()
            },
        },
        "development_panel": selected["development"],
        "evaluation_sets": {
            "diagnostic": selected["diagnostic"],
            "validation": selected["validation"],
        },
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
        raise RuntimeError("SlackTrader splits failed Policy R")
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
        "# SlackTrader V1 Split Ledger",
        "",
        "All panels were selected in ascending seed order under route-clean Policy R",
        "without controller-performance measurements. Validation remains unopened.",
    ]
    panels = {
        "Development": ledger["development_panel"],
        "Diagnostic": ledger["evaluation_sets"]["diagnostic"],
        "Validation": ledger["evaluation_sets"]["validation"],
    }
    for label, split in panels.items():
        lines.extend(
            [
                "",
                f"## {label}: {split['id']}",
                "",
                f"Ordered SHA-256: `{split['ordered_sha256']}`.",
                "",
                "| Seed | Route SHA-256 | Full-task SHA-256 | Feasibility metadata |",
                "| ---: | --- | --- | --- |",
            ]
        )
        for row in split["accepted_tasks"]:
            lines.append(
                f"| {row['numeric_seed']} | `{row['route_sha256']}` | "
                f"`{row['full_task_sha256']}` | {row['feasibility_classification']} |"
            )
    lines.extend(
        [
            "",
            "## Paper",
            "",
            "Seeds 4000–4017 remain `SEALED_UNINSPECTED`; they were excluded",
            "numerically and were never instantiated or fingerprinted.",
            "",
        ]
    )
    return "\n".join(lines)
