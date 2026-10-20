"""Route-clean Hybrid RL streams and evaluation splits (Policy R)."""

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
from benchmarks.physical_feasibility_reference.diagnostics import (
    protected_rows as prior_protected_rows,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "canonical.json"
PHYSICAL_SPLIT = (
    ROOT / "benchmarks/physical_feasibility_reference/split_ledger.json"
)
TRAINING_STREAMS = (
    ROOT / "benchmarks/constrained_v2_collector_replication/stream_ledger.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def training_streams() -> dict[str, Any]:
    streams = _load(TRAINING_STREAMS)
    expected = {
        "11": "08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c",
        "29": "ca8912845c1293e86fd326a7415901c9beb0c70ae6a32db5e78263fceb355312",
        "47": "2b1323d664f8ca470975ce4238f33a1883bf6d88eb43d035681b5c1520b15255",
    }
    for seed, ordered in expected.items():
        stream = streams[seed]
        if len(stream["accepted_tasks"]) != 865:
            raise RuntimeError(f"CANONICAL_{seed} reachable horizon changed")
        if stream["ordered_sha256"] != ordered:
            raise RuntimeError(f"CANONICAL_{seed} ordered hash changed")
    return streams


def protected_rows() -> list[dict[str, Any]]:
    rows = [_projection(row) for row in prior_protected_rows()]
    physical = _load(PHYSICAL_SPLIT)
    rows.extend(
        _projection(row)
        for split in physical["evaluation_sets"].values()
        for row in split["accepted_tasks"]
    )
    rows.extend(
        _projection(row)
        for stream in training_streams().values()
        for row in stream["accepted_tasks"]
    )
    unique = {
        (row["numeric_seed"], row["route_sha256"], row["full_task_sha256"]): row
        for row in rows
    }
    return list(unique.values())


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
        *(row["numeric_seed"] for row in protected),
        *(row["numeric_seed"] for row in other),
        *PAPER_SEEDS,
    }
    routes = {
        *(row["route_sha256"] for row in protected),
        *(row["route_sha256"] for row in other),
    }
    full_tasks = {
        *(row["full_task_sha256"] for row in protected),
        *(row["full_task_sha256"] for row in other),
    }
    seed = int(start)
    while len(accepted) < count:
        if seed in PAPER_SEEDS:
            raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")
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
        "candidate_start": start,
        "selection_order": "ascending numeric seed",
        "policy": "R: numeric, route and full-task clean",
        "accepted_tasks": accepted,
        "scanned_candidates": scanned,
        "rejected_candidates": [row for row in scanned if not row["accepted"]],
        "ordered_sha256": canonical_sha256(frozen),
    }


def build_split_ledger() -> dict[str, Any]:
    config = _load(CONFIG)
    protected = protected_rows()
    diagnostic_config = config["splits"]["diagnostic"]
    validation_config = config["splits"]["validation"]
    diagnostic = _select(
        diagnostic_config["id"],
        diagnostic_config["candidate_start"],
        diagnostic_config["accepted_count"],
        protected=protected,
        other=[],
    )
    validation = _select(
        validation_config["id"],
        validation_config["candidate_start"],
        validation_config["accepted_count"],
        protected=protected,
        other=diagnostic["accepted_tasks"],
    )
    evaluation = [
        *diagnostic["accepted_tasks"],
        *validation["accepted_tasks"],
    ]
    training = [
        _projection(row)
        for stream in training_streams().values()
        for row in stream["accepted_tasks"]
    ]
    training_routes = {row["route_sha256"] for row in training}
    training_full = {row["full_task_sha256"] for row in training}
    protected_routes = {row["route_sha256"] for row in protected}
    protected_full = {row["full_task_sha256"] for row in protected}
    checks = {
        "diagnostic_count_18": len(diagnostic["accepted_tasks"]) == 18,
        "validation_count_36": len(validation["accepted_tasks"]) == 36,
        "evaluation_numeric_unique": len(
            {row["numeric_seed"] for row in evaluation}
        )
        == 54,
        "evaluation_route_unique": len({row["route_sha256"] for row in evaluation})
        == 54,
        "evaluation_full_task_unique": len(
            {row["full_task_sha256"] for row in evaluation}
        )
        == 54,
        "training_stream_rows_2595": len(training) == 2595,
        "training_evaluation_route_disjoint": not (
            training_routes & {row["route_sha256"] for row in evaluation}
        ),
        "training_evaluation_full_task_disjoint": not (
            training_full & {row["full_task_sha256"] for row in evaluation}
        ),
        "prior_evaluation_route_disjoint": all(
            row["route_sha256"] not in protected_routes for row in evaluation
        ),
        "prior_evaluation_full_task_disjoint": all(
            row["full_task_sha256"] not in protected_full for row in evaluation
        ),
        "paper_numeric_absent": not (
            {row["numeric_seed"] for row in evaluation} & set(PAPER_SEEDS)
        ),
    }
    streams = training_streams()
    return {
        "schema_version": 1,
        "policy": "Route-clean Policy R",
        "training": {
            "source": str(TRAINING_STREAMS.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(TRAINING_STREAMS.read_bytes()).hexdigest(),
            "reachable_rows_per_seed": 865,
            "total_reachable_rows": 2595,
            "streams": {
                seed: {
                    "id": f"CANONICAL_{seed}",
                    "ordered_sha256": stream["ordered_sha256"],
                    "accepted_task_count": len(stream["accepted_tasks"]),
                }
                for seed, stream in streams.items()
            },
        },
        "protected_prior_row_count": len(protected),
        "evaluation_sets": {
            "diagnostic": diagnostic,
            "validation": validation,
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
        raise RuntimeError("runtime route differs from frozen Hybrid split")
    if fingerprint.full_task_sha256 != row["full_task_sha256"]:
        raise RuntimeError("runtime full task differs from frozen Hybrid split")
    return True


def render_split_ledger(ledger: dict[str, Any]) -> str:
    lines = [
        "# Hybrid RL Split Ledger",
        "",
        "Policy: **Route-clean Policy R**. The final Validation tasks are selected",
        "before training and have not been used for Hybrid performance evaluation.",
        "",
        "Training reuses the three immutable 865-row canonical streams. Exact",
        "route and full-task collision checks cover all 2,595 reachable rows.",
    ]
    for name, split in ledger["evaluation_sets"].items():
        lines.extend(
            [
                "",
                f"## {split['id']}",
                "",
                f"Ordered SHA-256: `{split['ordered_sha256']}`.",
                "",
                "| Seed | Route SHA-256 | Full-task SHA-256 |",
                "| ---: | --- | --- |",
            ]
        )
        for row in split["accepted_tasks"]:
            lines.append(
                f"| {row['numeric_seed']} | `{row['route_sha256']}` | "
                f"`{row['full_task_sha256']}` |"
            )
    lines.extend(
        [
            "",
            "## Paper",
            "",
            "`SEALED_UNINSPECTED`; only numeric exclusion of 4000–4017 was used.",
            "",
        ]
    )
    return "\n".join(lines)
