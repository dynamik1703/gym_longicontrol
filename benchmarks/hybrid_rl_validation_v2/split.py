"""Deterministic construction of the unopened route-clean V2 panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed
from benchmarks.hybrid_rl_design.split import (
    protected_rows as pre_hybrid_protected_rows,
)
from benchmarks.hybrid_rl_design.split import training_streams

from .constants import (
    ROOT,
    V1_ID,
    V1_ORDERED_SHA256,
    V2_CANDIDATE_START,
    V2_ID,
    V2_TASK_COUNT,
)
from .ledger import atomic_json, canonical_sha256

V1_SPLIT_LEDGER = ROOT / "benchmarks/hybrid_rl_design/split_ledger.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_seed": int(row.get("numeric_seed", row.get("reset_seed"))),
        "route_sha256": row["route_sha256"],
        "full_task_sha256": row["full_task_sha256"],
    }


def protected_rows() -> tuple[list[dict[str, Any]], dict[str, int]]:
    inherited = [_projection(row) for row in pre_hybrid_protected_rows()]
    v1 = _load(V1_SPLIT_LEDGER)
    validation = v1["evaluation_sets"]["validation"]
    diagnostic = v1["evaluation_sets"]["diagnostic"]
    if validation["id"] != V1_ID:
        raise RuntimeError("historical V1 identity changed")
    if validation["ordered_sha256"] != V1_ORDERED_SHA256:
        raise RuntimeError("historical V1 ordered hash changed")
    v1_diagnostic = [_projection(row) for row in diagnostic["accepted_tasks"]]
    v1_validation = [_projection(row) for row in validation["accepted_tasks"]]
    combined = [*inherited, *v1_diagnostic, *v1_validation]
    unique = {
        (row["numeric_seed"], row["route_sha256"], row["full_task_sha256"]): row
        for row in combined
    }
    counts = {
        "inherited_historical_physical_and_training_rows": len(inherited),
        "hybrid_diagnostic_v1_rows": len(v1_diagnostic),
        "hybrid_validation_v1_rows": len(v1_validation),
        "unique_protected_rows": len(unique),
    }
    return list(unique.values()), counts


def build_split_ledger() -> dict[str, Any]:
    protected, protected_counts = protected_rows()
    numeric = {*(row["numeric_seed"] for row in protected), *PAPER_SEEDS}
    routes = {row["route_sha256"] for row in protected}
    full_tasks = {row["full_task_sha256"] for row in protected}
    protected_routes = set(routes)
    protected_full_tasks = set(full_tasks)
    accepted: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    seed = V2_CANDIDATE_START
    while len(accepted) < V2_TASK_COUNT:
        if seed in PAPER_SEEDS:
            raise PermissionError("Paper task construction is forbidden")
        track = track_from_seed(seed)
        fingerprint = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        reasons = []
        if seed in numeric:
            reasons.append("NUMERIC_RESERVED")
        if fingerprint.route_sha256 in routes:
            reasons.append("POLICY_R_ROUTE_RESERVED")
        if fingerprint.full_task_sha256 in full_tasks:
            reasons.append("POLICY_R_FULL_TASK_RESERVED")
        row = {
            "candidate_index": seed - V2_CANDIDATE_START,
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
    training = [
        _projection(row)
        for stream in training_streams().values()
        for row in stream["accepted_tasks"]
    ]
    checks = {
        "accepted_count_36": len(accepted) == V2_TASK_COUNT,
        "all_candidates_accounted_for": len(scanned)
        == len(accepted) + sum(not row["accepted"] for row in scanned),
        "internal_numeric_unique": len({row["numeric_seed"] for row in accepted})
        == V2_TASK_COUNT,
        "internal_route_unique": len({row["route_sha256"] for row in accepted})
        == V2_TASK_COUNT,
        "internal_full_task_unique": len(
            {row["full_task_sha256"] for row in accepted}
        )
        == V2_TASK_COUNT,
        "training_rows_2595": len(training) == 2_595,
        "protected_route_disjoint": all(
            row["route_sha256"] not in protected_routes for row in accepted
        ),
        "protected_full_task_disjoint": all(
            row["full_task_sha256"] not in protected_full_tasks for row in accepted
        ),
        "paper_numeric_absent": not (
            {row["numeric_seed"] for row in accepted} & set(PAPER_SEEDS)
        ),
        "original_v1_fully_protected": protected_counts[
            "hybrid_validation_v1_rows"
        ]
        == 36,
    }
    ledger = {
        "schema_version": 1,
        "id": V2_ID,
        "state": "FROZEN_UNOPENED",
        "candidate_start": V2_CANDIDATE_START,
        "selection_order": "ascending numeric seed; first 36 passing Policy R",
        "selection_inputs": "numeric seed and task fingerprints only; no performance",
        "fingerprint_implementation": (
            "benchmarks.content_split_audit.fingerprints"
        ),
        "task_generator": "benchmarks.content_split_audit.streams.track_from_seed",
        "historical_v1": {
            "id": V1_ID,
            "state": "INCIDENT_ABORTED",
            "ordered_sha256": V1_ORDERED_SHA256,
            "all_36_tasks_protected": True,
        },
        "protected_sources": protected_counts,
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(frozen),
        "checks": checks,
        "route_clean_verified": all(checks.values()),
        "performance_episode_count": 0,
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(PAPER_SEEDS),
            "content_fingerprints_constructed": 0,
            "episode_count": 0,
        },
        "provenance": {
            "v1_split_ledger_sha256": hashlib.sha256(
                V1_SPLIT_LEDGER.read_bytes()
            ).hexdigest(),
        },
    }
    if not ledger["route_clean_verified"]:
        raise RuntimeError("V2 split failed route-clean Policy R")
    return ledger


def materialize(path: Path) -> dict[str, Any]:
    ledger = build_split_ledger()
    atomic_json(path, ledger)
    return ledger


def verify_regeneration(path: Path) -> dict[str, Any]:
    stored = _load(path)
    regenerated = build_split_ledger()
    if stored != regenerated:
        raise RuntimeError("V2 split does not regenerate deterministically")
    return {
        "deterministic_regeneration": True,
        "id": stored["id"],
        "accepted_count": len(stored["accepted_tasks"]),
        "rejected_count": len(stored["rejected_candidates"]),
        "ordered_sha256": stored["ordered_sha256"],
        "route_clean_verified": stored["route_clean_verified"],
    }


def render_split_ledger(ledger: dict[str, Any]) -> str:
    lines = [
        "# Hybrid RL Validation V2 Split Ledger",
        "",
        "The panel was selected deterministically after the V1 infrastructure",
        "incident, using ascending numeric candidates and route-clean Policy R.",
        "No controller performance was evaluated during construction.",
        "",
        f"Ordered SHA-256: `{ledger['ordered_sha256']}`.",
        "",
        "| Ordinal | Seed | Route SHA-256 | Full-task SHA-256 |",
        "| ---: | ---: | --- | --- |",
    ]
    for ordinal, row in enumerate(ledger["accepted_tasks"]):
        lines.append(
            f"| {ordinal} | {row['numeric_seed']} | `{row['route_sha256']}` | "
            f"`{row['full_task_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Candidate accounting",
            "",
            f"Accepted: `{len(ledger['accepted_tasks'])}`; rejected: "
            f"`{len(ledger['rejected_candidates'])}`; all scanned candidates and "
            "reasons are retained in `split_ledger.json`.",
            "",
            "Paper seeds 4000–4017 were excluded numerically only; no Paper task",
            "was constructed or fingerprinted.",
            "",
        ]
    )
    return "\n".join(lines)
