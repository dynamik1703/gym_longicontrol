"""Deterministic, append-oriented search-history persistence."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_sha256

DEFAULT_HISTORY_PATH = Path(__file__).with_name("search_history.json")


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def new_history(protocol) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha256(protocol),
        "status": "NOT_STARTED",
        "candidate_slots_by_generation": {
            str(index): count
            for index, count in enumerate(
                protocol.search_budget.candidate_counts_by_generation
            )
        },
        "candidates": [],
        "parent_selections": [],
        "best_generation_zero_candidate_id": None,
        "winner": None,
        "engineering_effort": {
            "generated_reward_candidate_count": 0,
            "generation_count_started": 0,
            "screening_rl_transitions": 0,
            "final_training_transitions": 0,
            "human_reward_edits": 0,
            "human_selected_coefficients_during_search": 0,
            "explicit_task_metrics_in_reflection": 13,
        },
    }


def load_history(path: str | Path, protocol) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Unsupported search-history schema")
    if raw.get("protocol_id") != protocol.protocol_id:
        raise ValueError("Search history belongs to another protocol")
    if raw.get("protocol_sha256") != protocol_sha256(protocol):
        raise ValueError("Search history protocol hash mismatch")
    if raw.get("status") not in {"NOT_STARTED", "OPEN", "CLOSED"}:
        raise ValueError("Invalid search-history status")
    expected_slots = {
        str(index): count
        for index, count in enumerate(
            protocol.search_budget.candidate_counts_by_generation
        )
    }
    if raw.get("candidate_slots_by_generation") != expected_slots:
        raise ValueError("Search-history candidate slots changed")
    ids = [candidate["candidate_id"] for candidate in raw["candidates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Search history contains duplicate candidate IDs")
    for generation, maximum in enumerate(
        protocol.search_budget.candidate_counts_by_generation
    ):
        candidates = [
            item for item in raw["candidates"] if item["generation"] == generation
        ]
        count = len(candidates)
        if count > maximum:
            raise ValueError(f"Generation {generation} exceeds its search budget")
        expected_ids = {f"g{generation}-c{index:02d}" for index in range(1, count + 1)}
        if {item["candidate_id"] for item in candidates} != expected_ids:
            raise ValueError(
                f"Generation {generation} candidate IDs are not contiguous"
            )
    allowed_statuses = {
        "repair_required",
        "technical_failure",
        "validated",
        "screened",
        "reflected",
        "ranked",
    }
    if any(item.get("status") not in allowed_statuses for item in raw["candidates"]):
        raise ValueError("Search history contains an invalid candidate status")
    if any(
        item.get("repair_attempt_count") not in {0, 1} for item in raw["candidates"]
    ):
        raise ValueError("Search history exceeds the single-repair budget")
    return raw


def initialize_history(path: str | Path, protocol, *, overwrite: bool = False) -> Path:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Search history already exists: {destination}")
    return _write_json(destination, new_history(protocol))


def start_search(history: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(history)
    if result["status"] != "NOT_STARTED":
        raise ValueError("Only a NOT_STARTED search can be opened")
    result["status"] = "OPEN"
    return result


def require_open(history: dict[str, Any]) -> None:
    if history["status"] != "OPEN":
        raise RuntimeError("Search history must be OPEN")


def candidate_by_id(history: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [
        item for item in history["candidates"] if item["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate candidate ID: {candidate_id}")
    return matches[0]


def next_candidate_id(history: dict[str, Any], protocol, generation: int) -> str:
    require_open(history)
    if generation < 0 or generation > protocol.search_budget.maximum_generation:
        raise ValueError("Generation lies outside the frozen search")
    existing = [
        item for item in history["candidates"] if item["generation"] == generation
    ]
    maximum = protocol.search_budget.candidate_counts_by_generation[generation]
    if len(existing) >= maximum:
        raise RuntimeError(f"Generation {generation} already consumed all slots")
    if generation > 0:
        previous = [
            item
            for item in history["candidates"]
            if item["generation"] == generation - 1
        ]
        if (
            len(previous)
            != protocol.search_budget.candidate_counts_by_generation[generation - 1]
        ):
            raise RuntimeError("Previous generation has not consumed every slot")
        if not any(
            item["for_generation"] == generation
            for item in history["parent_selections"]
        ):
            raise RuntimeError("Parents have not been selected deterministically")
    return f"g{generation}-c{len(existing) + 1:02d}"


def save_history(path: str | Path, history: dict[str, Any], protocol) -> Path:
    temporary = Path(path).with_name(f".{Path(path).name}.validate")
    _write_json(temporary, history)
    try:
        load_history(temporary, protocol)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _write_json(path, history)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    initialize.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    initialize.add_argument("--overwrite", action="store_true")
    start = subparsers.add_parser("start")
    start.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    start.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.command == "initialize":
        print(initialize_history(args.history, protocol, overwrite=args.overwrite))
    else:
        history = load_history(args.history, protocol)
        print(save_history(args.history, start_search(history), protocol))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
