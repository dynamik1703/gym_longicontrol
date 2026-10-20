"""Canonical streams, explicit split scopes, and current-study separation."""

from __future__ import annotations

import copy
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
HISTORICAL_OVERLAP_LEDGER = HERE / "historical_overlap_ledger.json"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def reachable_horizon() -> dict[str, int]:
    raw = load_configuration().raw
    budget = int(raw["real_transition_budget"])
    minimum = int(raw["minimum_physically_possible_episode_length"])
    completed = budget // minimum
    return {
        "real_transition_budget": budget,
        "minimum_physically_possible_episode_length": minimum,
        "max_completed_episodes": completed,
        "max_exposed_tasks": completed + 1,
    }


def _projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "numeric_seed": int(row["numeric_seed"]),
        "route_sha256": row["route_sha256"],
        "full_task_sha256": row["full_task_sha256"],
    }


@lru_cache(maxsize=1)
def canonical_streams() -> dict[str, dict[str, Any]]:
    """Recover the canonical C streams used verbatim as G streams."""

    payload = json.loads(COLLECTOR_STREAMS.read_text(encoding="utf-8"))
    expected = load_configuration().raw["canonical_stream_sha256"]
    maximum = reachable_horizon()["max_exposed_tasks"]
    for seed in TRAINING_SEEDS:
        stream = payload[str(seed)]
        if len(stream["accepted_tasks"]) != maximum:
            raise RuntimeError("Canonical stream does not cover maximum exposure")
        if stream["ordered_sha256"] != expected[str(seed)]:
            raise RuntimeError("Canonical stream SHA-256 changed")
        if stream["seed_sequence"] != [seed, 0x43414E4F]:
            raise RuntimeError("Canonical SeedSequence changed")
        for index, row in enumerate(stream["accepted_tasks"]):
            if row["accepted_task_index"] != index:
                raise RuntimeError("Canonical stream ordering changed")
            if row["nominal_policy_seed"] != seed:
                raise RuntimeError("Canonical C/G seed pairing changed")
    return payload


@lru_cache(maxsize=1)
def opened_split_rows() -> dict[str, list[dict[str, Any]]]:
    """Return only prior split content already legitimately available."""

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
    from benchmarks.same_information_scalar_vs_constrained.split import (
        build_split_ledger as build_sameinfo_ledger,
    )

    sameinfo = build_sameinfo_ledger()
    for name in ("diagnostic", "validation"):
        result[f"collector_{name}"] = [
            _projection(row)
            for row in collector["evaluation_sets"][name]["accepted_tasks"]
        ]
        result[f"sameinfo_{name}"] = [
            _projection(row)
            for row in sameinfo["evaluation_sets"][name]["accepted_tasks"]
        ]
    return result


def _training_rows() -> list[dict[str, Any]]:
    return [
        {
            "training_policy": f"G{seed}",
            "accepted_task_index": int(row["accepted_task_index"]),
            "numeric_seed": int(row["reset_seed"]),
            "route_sha256": row["route_sha256"],
            "full_task_sha256": row["full_task_sha256"],
        }
        for seed, stream in canonical_streams().items()
        for row in stream["accepted_tasks"]
    ]


def historical_overlap_ledger() -> dict[str, Any]:
    ledger = json.loads(HISTORICAL_OVERLAP_LEDGER.read_text(encoding="utf-8"))
    if ledger.get("collision_count") != 6 or len(ledger.get("collisions", [])) != 6:
        raise RuntimeError("Historical-overlap ledger changed")
    return ledger


def detected_historical_overlaps() -> list[dict[str, Any]]:
    """Exhaustively detect pairwise route-and-full-task historical matches."""

    detected: list[dict[str, Any]] = []
    for training in _training_rows():
        for historical_set, rows in opened_split_rows().items():
            for historical in rows:
                if (
                    training["route_sha256"] == historical["route_sha256"]
                    and training["full_task_sha256"] == historical["full_task_sha256"]
                ):
                    detected.append(
                        {
                            "training_policy": training["training_policy"],
                            "accepted_task_index": training["accepted_task_index"],
                            "reset_seed": training["numeric_seed"],
                            "historical_set": historical_set,
                            "historical_numeric_seed": historical["numeric_seed"],
                            "route_sha256": training["route_sha256"],
                            "full_task_sha256": training["full_task_sha256"],
                        }
                    )
    return sorted(
        detected,
        key=lambda row: (
            int(row["training_policy"][1:]),
            row["accepted_task_index"],
            row["historical_set"],
            row["historical_numeric_seed"],
        ),
    )


def verify_historical_overlap_ledger() -> bool:
    recorded = []
    for row in historical_overlap_ledger()["collisions"]:
        normalized = dict(row)
        if normalized["historical_set"] == "diagnostic":
            normalized["historical_set"] = "old_diagnostic"
        elif normalized["historical_set"] == "V1 exploratory":
            normalized["historical_set"] = "v1_exploratory"
        recorded.append(normalized)
    recorded.sort(
        key=lambda row: (
            int(row["training_policy"][1:]),
            row["accepted_task_index"],
            row["historical_set"],
            row["historical_numeric_seed"],
        )
    )
    if detected_historical_overlaps() != recorded:
        raise RuntimeError("Detected historical overlaps differ from permanent ledger")
    return True


def _candidate(
    seed: int,
    *,
    start: int,
    accepted: list[dict[str, Any]],
    other: list[dict[str, Any]],
) -> dict[str, Any]:
    if seed in PAPER_SEEDS:
        raise RuntimeError("Paper candidates are fail-closed")
    protected = [
        *(row for rows in opened_split_rows().values() for row in rows),
        *_training_rows(),
    ]
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
    reasons: list[str] = []
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


def current_gcsl_evaluation_reserved() -> dict[str, list[dict[str, Any]]]:
    return {
        split["id"]: split["accepted_tasks"] for split in evaluation_sets().values()
    }


def assert_training_row_clean(row: dict[str, Any]) -> None:
    """Enforce only Paper numeric sealing and current-study reservations.

    Historical overlap is deliberately not a runtime failure: these immutable
    streams predate this study and the matched sets are not GCSL outcomes.
    """

    if int(row["reset_seed"]) in PAPER_SEEDS:
        raise RuntimeError("PAPER:NUMERIC_SEED")
    for split_id, protected in current_gcsl_evaluation_reserved().items():
        numeric = {int(value["numeric_seed"]) for value in protected}
        routes = {value["route_sha256"] for value in protected}
        full = {value["full_task_sha256"] for value in protected}
        if int(row["reset_seed"]) in numeric:
            raise RuntimeError(f"{split_id}:NUMERIC_SEED")
        if row["route_sha256"] in routes:
            raise RuntimeError(f"{split_id}:ROUTE")
        if row["full_task_sha256"] in full:
            raise RuntimeError(f"{split_id}:FULL_TASK")


def verify_runtime_track(row: dict[str, Any], track: Any) -> None:
    """Verify frozen content after reset; Paper is never instantiated here."""

    assert_training_row_clean(row)
    fingerprint = fingerprint_track(
        track, v2_semantics(), numeric_seed=int(row["reset_seed"])
    )
    if fingerprint.route_sha256 != row["route_sha256"]:
        raise RuntimeError("Runtime canonical route differs from frozen stream")
    if fingerprint.full_task_sha256 != row["full_task_sha256"]:
        raise RuntimeError("Runtime canonical full task differs from frozen stream")


def build_split_ledger() -> dict[str, Any]:
    splits = evaluation_sets()
    combined = [row for split in splits.values() for row in split["accepted_tasks"]]
    training = _training_rows()
    protected = [row for rows in opened_split_rows().values() for row in rows]
    historical_and_training = [*protected, *training]
    protected_routes = {row["route_sha256"] for row in historical_and_training}
    protected_full = {row["full_task_sha256"] for row in historical_and_training}
    diagnostic = splits["diagnostic"]["accepted_tasks"]
    validation = splits["validation"]["accepted_tasks"]
    checks = {
        "reachable_horizon_recomputed_to_865": reachable_horizon()["max_exposed_tasks"]
        == 865,
        "both_sets_have_18": all(
            len(split["accepted_tasks"]) == 18 for split in splits.values()
        ),
        "diagnostic_numeric_unique": len({row["numeric_seed"] for row in diagnostic})
        == 18,
        "diagnostic_route_unique": len({row["route_sha256"] for row in diagnostic})
        == 18,
        "diagnostic_full_task_unique": len(
            {row["full_task_sha256"] for row in diagnostic}
        )
        == 18,
        "validation_numeric_unique": len({row["numeric_seed"] for row in validation})
        == 18,
        "validation_route_unique": len({row["route_sha256"] for row in validation})
        == 18,
        "validation_full_task_unique": len(
            {row["full_task_sha256"] for row in validation}
        )
        == 18,
        "diagnostic_validation_numeric_disjoint": not (
            {row["numeric_seed"] for row in diagnostic}
            & {row["numeric_seed"] for row in validation}
        ),
        "diagnostic_validation_route_disjoint": not (
            {row["route_sha256"] for row in diagnostic}
            & {row["route_sha256"] for row in validation}
        ),
        "diagnostic_validation_full_task_disjoint": not (
            {row["full_task_sha256"] for row in diagnostic}
            & {row["full_task_sha256"] for row in validation}
        ),
        "new_evaluation_historical_and_training_route_separation": all(
            row["route_sha256"] not in protected_routes for row in combined
        ),
        "new_evaluation_historical_and_training_full_task_separation": all(
            row["full_task_sha256"] not in protected_full for row in combined
        ),
        "canonical_streams_have_865": all(
            len(stream["accepted_tasks"]) == 865
            for stream in canonical_streams().values()
        ),
        "current_study_training_evaluation_gate": all(
            _row_is_clean(row)
            for stream in canonical_streams().values()
            for row in stream["accepted_tasks"]
        ),
        "six_historical_overlaps_match_permanent_ledger": (
            verify_historical_overlap_ledger()
        ),
    }
    return {
        "schema_version": 1,
        "policy": "Current-study route-clean Policy R",
        "scope": {
            "current_gcsl_evaluation_reserved": [
                "GCSL_DIAGNOSTIC_V1",
                "GCSL_VALIDATION_V1",
            ],
            "historically_opened_content": sorted(opened_split_rows()),
            "fixed_training_streams": (
                "fixed canonical training streams with disclosed "
                "historical-content overlap"
            ),
        },
        "construction_order": [
            "RECOVER_EXISTING_CANONICAL_STREAMS",
            "GCSL_DIAGNOSTIC_V1",
            "GCSL_VALIDATION_V1",
        ],
        "reachable_horizon": reachable_horizon(),
        "opened_sets": {
            name: {
                "count": len(rows),
                "numeric_seeds": [row["numeric_seed"] for row in rows],
            }
            for name, rows in opened_split_rows().items()
        },
        "canonical_streams": canonical_streams(),
        "canonical_stream_sha256": {
            seed: stream["ordered_sha256"]
            for seed, stream in canonical_streams().items()
        },
        "evaluation_sets": splits,
        "evaluation_sha256": {
            name: split["ordered_sha256"] for name, split in splits.items()
        },
        "checks": checks,
        "historical_overlap_count": len(detected_historical_overlaps()),
        "current_study_route_clean_verified": all(checks.values()),
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(PAPER_SEEDS),
            "content_fingerprints_constructed": 0,
        },
    }


def _row_is_clean(row: dict[str, Any]) -> bool:
    try:
        assert_training_row_clean(row)
    except RuntimeError:
        return False
    return True


def stream_rows(training_seed: int) -> list[dict[str, Any]]:
    if training_seed not in TRAINING_SEEDS:
        raise ValueError("Unknown canonical training seed")
    return copy.deepcopy(canonical_streams()[str(training_seed)]["accepted_tasks"])


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


class CanonicalTaskStream:
    """Serializable cursor over one immutable 865-task explicit stream."""

    def __init__(self, training_seed: int, *, cursor: int = 0):
        self.training_seed = int(training_seed)
        self.rows = stream_rows(training_seed)
        self.cursor = int(cursor)
        if not 0 <= self.cursor <= len(self.rows):
            raise ValueError("Canonical task-stream cursor is invalid")

    def next_row(self) -> dict[str, Any]:
        if self.cursor >= len(self.rows):
            raise RuntimeError("Frozen 865-task canonical stream exhausted")
        row = copy.deepcopy(self.rows[self.cursor])
        assert_training_row_clean(row)
        self.cursor += 1
        return row

    def state_dict(self) -> dict[str, int]:
        return {"training_seed": self.training_seed, "cursor": self.cursor}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> CanonicalTaskStream:
        return cls(int(state["training_seed"]), cursor=int(state["cursor"]))
