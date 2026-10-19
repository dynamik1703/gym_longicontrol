"""Evaluation-first split selection and neutral canonical training streams."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import (
    ALL_NUMERICALLY_RESERVED,
    DEVELOPMENT_SEEDS,
    PAPER_SEEDS,
    fingerprint_rows,
    track_from_seed,
)
from benchmarks.content_split_audit.streams import (
    VALIDATION_SEEDS as LEGACY_VALIDATION_SEEDS,
)

from .config import CANONICAL_NAMESPACE, TRAINING_SEEDS, load_configuration

HERE = Path(__file__).resolve().parent
V3_LEDGER = HERE.parent / "constrained_v2_seed11_factorial_v3" / "split_ledger.json"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def reachable_horizon() -> dict[str, int]:
    raw = load_configuration().raw
    completed = int(raw["real_transition_budget"]) // int(
        raw["minimum_physically_possible_episode_length"]
    )
    return {
        "real_transition_budget": int(raw["real_transition_budget"]),
        "minimum_physically_possible_episode_length": int(
            raw["minimum_physically_possible_episode_length"]
        ),
        "max_completed_episodes": completed,
        "max_exposed_tasks": completed + 1,
    }


def _hash_sets(rows: Iterable[dict[str, Any]]) -> tuple[set[str], set[str]]:
    rows = list(rows)
    return (
        {row["route_sha256"] for row in rows},
        {row["full_task_sha256"] for row in rows},
    )


@lru_cache(maxsize=1)
def opened_split_rows() -> dict[str, list[dict[str, Any]]]:
    semantics = v2_semantics()
    result: dict[str, list[dict[str, Any]]] = {}
    for name, seeds in (
        ("legacy_development", DEVELOPMENT_SEEDS),
        ("legacy_validation", LEGACY_VALIDATION_SEEDS),
    ):
        generated = fingerprint_rows(
            [track_from_seed(seed) for seed in seeds],
            semantics,
            numeric_seeds=list(seeds),
            draw_indices=[None] * len(seeds),
        )
        result[name] = [
            {
                "numeric_seed": int(row["reset_seed"]),
                "route_sha256": row["route_sha256"],
                "full_task_sha256": row["full_task_sha256"],
            }
            for row in generated
        ]
    v3 = json.loads(V3_LEDGER.read_text(encoding="utf-8"))
    result["v3_diagnostic"] = [
        {
            "numeric_seed": int(row["numeric_seed"]),
            "route_sha256": row["route_sha256"],
            "full_task_sha256": row["full_task_sha256"],
        }
        for row in v3["accepted_tasks"]
    ]
    return result


def _evaluation_candidate(
    seed: int,
    *,
    candidate_start: int,
    accepted_this_set: list[dict[str, Any]],
    accepted_other_set: list[dict[str, Any]],
) -> dict[str, Any]:
    if seed in PAPER_SEEDS:
        raise RuntimeError("Paper candidates are fail-closed")
    opened = opened_split_rows()
    opened_rows = [row for rows in opened.values() for row in rows]
    prior = [*accepted_other_set, *accepted_this_set]
    numeric_reserved = {
        *ALL_NUMERICALLY_RESERVED,
        *(int(row["numeric_seed"]) for row in opened_rows),
        *(int(row["numeric_seed"]) for row in prior),
    }
    track = track_from_seed(seed)
    fingerprint = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
    opened_routes, opened_full = _hash_sets(opened_rows)
    prior_routes, prior_full = _hash_sets(prior)
    reasons: list[str] = []
    if seed in numeric_reserved:
        reasons.append("NUMERIC_RESERVED")
    if fingerprint.route_sha256 in opened_routes:
        reasons.append("OPENED_ROUTE")
    if fingerprint.full_task_sha256 in opened_full:
        reasons.append("OPENED_FULL_TASK")
    if fingerprint.route_sha256 in prior_routes:
        reasons.append("CROSS_SET_OR_INTERNAL_ROUTE")
    if fingerprint.full_task_sha256 in prior_full:
        reasons.append("CROSS_SET_OR_INTERNAL_FULL_TASK")
    return {
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


def _select_evaluation_set(name: str, other: list[dict[str, Any]]) -> dict[str, Any]:
    selection = load_configuration().raw["evaluation_selections"][name]
    start = int(selection["candidate_start"])
    target = int(selection["accepted_count"])
    accepted: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    seed = start
    while len(accepted) < target:
        row = _evaluation_candidate(
            seed,
            candidate_start=start,
            accepted_this_set=accepted,
            accepted_other_set=other,
        )
        scanned.append(row)
        if row["accepted"]:
            accepted.append(row)
        seed += 1
    frozen_rows = [
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
        "ordered_sha256": canonical_sha256(frozen_rows),
    }


@lru_cache(maxsize=1)
def evaluation_sets() -> dict[str, dict[str, Any]]:
    diagnostic = _select_evaluation_set("diagnostic", [])
    validation = _select_evaluation_set("validation", diagnostic["accepted_tasks"])
    return {"diagnostic": diagnostic, "validation": validation}


def _reserved_for_training() -> tuple[set[int], set[str], set[str]]:
    opened = opened_split_rows()
    evaluations = evaluation_sets()
    rows = [row for values in opened.values() for row in values]
    rows += evaluations["diagnostic"]["accepted_tasks"]
    rows += evaluations["validation"]["accepted_tasks"]
    numeric = {
        *ALL_NUMERICALLY_RESERVED,
        *PAPER_SEEDS,
        *(int(row["numeric_seed"]) for row in rows),
    }
    routes, full = _hash_sets(rows)
    return numeric, routes, full


@cache
def canonical_stream(training_seed: int) -> dict[str, Any]:
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("Canonical stream requested for an unfrozen training seed")
    target = reachable_horizon()["max_exposed_tasks"]
    numeric_reserved, route_reserved, full_reserved = _reserved_for_training()
    rng = np.random.default_rng(
        np.random.SeedSequence([training_seed, CANONICAL_NAMESPACE])
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    raw_index = 0
    while len(accepted) < target:
        reset_seed = int(rng.integers(0, 2**31 - 1))
        reasons: list[str] = []
        if reset_seed in numeric_reserved:
            reasons.append("NUMERIC_RESERVED")
            rejected.append(
                {
                    "raw_candidate_index": raw_index,
                    "reset_seed": reset_seed,
                    "rejection_reasons": reasons,
                }
            )
            raw_index += 1
            continue
        track = track_from_seed(reset_seed)
        fingerprint = fingerprint_track(track, v2_semantics(), numeric_seed=reset_seed)
        if fingerprint.route_sha256 in route_reserved:
            reasons.append("EVALUATION_OR_OPENED_ROUTE")
        if fingerprint.full_task_sha256 in full_reserved:
            reasons.append("EVALUATION_OR_OPENED_FULL_TASK")
        row = {
            "nominal_policy_seed": training_seed,
            "accepted_task_index": len(accepted),
            "raw_candidate_index": raw_index,
            "reset_seed": reset_seed,
            "route_sha256": fingerprint.route_sha256,
            "full_task_sha256": fingerprint.full_task_sha256,
        }
        if reasons:
            rejected.append({**row, "rejection_reasons": reasons})
        else:
            accepted.append(row)
        raw_index += 1
    route_counts = Counter(row["route_sha256"] for row in accepted)
    full_counts = Counter(row["full_task_sha256"] for row in accepted)
    return {
        "nominal_policy_seed": training_seed,
        "seed_sequence": [training_seed, CANONICAL_NAMESPACE],
        "accepted_tasks": accepted,
        "rejected_candidates": rejected,
        "ordered_sha256": canonical_sha256(accepted),
        "duplicates": {
            "route_repetitions": len(accepted) - len(route_counts),
            "full_task_repetitions": len(accepted) - len(full_counts),
            "maximum_route_multiplicity": max(route_counts.values(), default=0),
            "maximum_full_task_multiplicity": max(full_counts.values(), default=0),
        },
    }


def final_route_clean_gate() -> dict[str, Any]:
    evaluations = evaluation_sets()
    streams = {str(seed): canonical_stream(seed) for seed in TRAINING_SEEDS}
    eval_rows = {name: value["accepted_tasks"] for name, value in evaluations.items()}
    collisions: list[dict[str, Any]] = []
    for split_name, rows in eval_rows.items():
        for seed, stream in streams.items():
            routes, full = _hash_sets(stream["accepted_tasks"])
            for row in rows:
                if row["route_sha256"] in routes or row["full_task_sha256"] in full:
                    collisions.append(
                        {
                            "evaluation_set": split_name,
                            "training_seed": int(seed),
                            "numeric_seed": row["numeric_seed"],
                        }
                    )
    return {"passed": not collisions, "collisions": collisions}


@lru_cache(maxsize=1)
def build_split_ledger() -> dict[str, Any]:
    streams = {str(seed): canonical_stream(seed) for seed in TRAINING_SEEDS}
    evaluation_fingerprints = {
        name: value["ordered_sha256"] for name, value in evaluation_sets().items()
    }
    stream_fingerprints = {
        seed: stream["ordered_sha256"] for seed, stream in streams.items()
    }
    raw = load_configuration().raw
    if evaluation_fingerprints != raw["expected_evaluation_fingerprints"]:
        raise RuntimeError("Frozen evaluation-set fingerprints changed")
    if stream_fingerprints != raw["expected_stream_fingerprints"]:
        raise RuntimeError("Frozen canonical stream fingerprints changed")
    gate = final_route_clean_gate()
    if not gate["passed"]:
        raise RuntimeError(
            "Final static route-clean gate failed; refreeze sets before training"
        )
    return {
        "schema_version": 1,
        "construction_order": [
            "COLLECTOR_DIAGNOSTIC_V1",
            "COLLECTOR_VALIDATION_V1",
            "CANONICAL_11",
            "CANONICAL_29",
            "CANONICAL_47",
        ],
        "reachable_horizon": reachable_horizon(),
        "evaluation_sets": evaluation_sets(),
        "canonical_streams": streams,
        "evaluation_fingerprints": evaluation_fingerprints,
        "stream_fingerprints": stream_fingerprints,
        "final_static_route_clean_gate": gate,
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "content_fingerprints_constructed": 0,
            "numeric_exclusion_only": list(PAPER_SEEDS),
        },
    }


def stream_rows(training_seed: int) -> list[dict[str, Any]]:
    return canonical_stream(training_seed)["accepted_tasks"]


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


def write_ledger(path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(
        json.dumps(build_split_ledger(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
