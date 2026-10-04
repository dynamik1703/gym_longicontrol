"""Frozen lexicographic ranking, parent selection and final reward freeze."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from .candidate_validation import source_sha256
from .history import (
    DEFAULT_HISTORY_PATH,
    candidate_by_id,
    load_history,
    save_history,
)
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_sha256


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ranking_key(candidate_id: str, fields: dict[str, Any]) -> tuple[Any, ...]:
    severity = fields["speed_violation_severity_among_relevant"]
    energy = fields["mean_feasible_energy_kwh"]
    return (
        -float(fields["development_rsr"]),
        -float(fields["completion_rate"]),
        -float(fields["median_incomplete_route_progress"]),
        -float(fields["deadline_compliance_among_completed"]),
        -float(fields["speed_compliance_among_relevant"]),
        float(severity) if severity is not None else float("inf"),
        float(energy) if energy is not None else float("inf"),
        candidate_id,
    )


def rank_candidates(history: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = []
    for candidate in history["candidates"]:
        if candidate["status"] not in {"reflected", "ranked"}:
            continue
        reflection = _read(candidate["reflection"]["json_path"])
        fields = reflection["ranking_fields"]
        ranked.append(
            {
                "candidate_id": candidate["candidate_id"],
                "generation": candidate["generation"],
                "fields": fields,
                "_key": ranking_key(candidate["candidate_id"], fields),
            }
        )
    ranked.sort(key=lambda item: item["_key"])
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
        del item["_key"]
    return ranked


def _generation_resolved(history, protocol, generation: int) -> bool:
    candidates = [
        item for item in history["candidates"] if item["generation"] == generation
    ]
    if (
        len(candidates)
        != protocol.search_budget.candidate_counts_by_generation[generation]
    ):
        return False
    return all(
        item["status"] in {"reflected", "ranked", "technical_failure"}
        for item in candidates
    )


def select_parents(history: dict[str, Any], protocol, for_generation: int):
    if history["status"] != "OPEN":
        raise RuntimeError("Parent selection requires an OPEN search")
    if for_generation not in {1, 2}:
        raise ValueError("Parents are selected only for Generation 1 or 2")
    if not _generation_resolved(history, protocol, for_generation - 1):
        raise RuntimeError("Previous generation is not fully resolved")
    if any(
        item["for_generation"] == for_generation
        for item in history["parent_selections"]
    ):
        raise RuntimeError("Parents were already selected for this generation")
    ranked = rank_candidates(history)
    if len(ranked) < protocol.search_budget.parent_count:
        raise RuntimeError("Fewer than two valid screened candidates remain")
    parent_ids = [
        item["candidate_id"] for item in ranked[: protocol.search_budget.parent_count]
    ]
    updated = deepcopy(history)
    for item in ranked:
        candidate_by_id(updated, item["candidate_id"])["ranking"] = {
            "rank_at_selection": item["rank"],
            "fields": item["fields"],
        }
        candidate_by_id(updated, item["candidate_id"])["status"] = "ranked"
    updated["parent_selections"].append(
        {
            "for_generation": for_generation,
            "candidate_ids": parent_ids,
            "eligible_candidate_count": len(ranked),
        }
    )
    if for_generation == 1:
        generation_zero = [item for item in ranked if item["generation"] == 0]
        updated["best_generation_zero_candidate_id"] = generation_zero[0][
            "candidate_id"
        ]
    return updated, tuple(parent_ids)


def freeze_winner(
    history: dict[str, Any],
    protocol,
    *,
    destination: str | Path,
) -> tuple[dict[str, Any], Path]:
    if history["status"] != "OPEN":
        raise RuntimeError("Winner freeze requires an OPEN search")
    selected_generations = {
        item["for_generation"] for item in history["parent_selections"]
    }
    if selected_generations != {1, 2}:
        raise RuntimeError("Both deterministic parent selections are required")
    if history["best_generation_zero_candidate_id"] is None:
        raise RuntimeError("The zero-shot Generation-0 control was not recorded")
    for generation in range(protocol.search_budget.maximum_generation + 1):
        if not _generation_resolved(history, protocol, generation):
            raise RuntimeError("Every generation must be resolved before freeze")
    ranked = rank_candidates(history)
    if not ranked:
        raise RuntimeError("No successfully screened candidate can be frozen")
    winner_id = ranked[0]["candidate_id"]
    winner = candidate_by_id(history, winner_id)
    source = Path(winner["source_path"])
    digest = source_sha256(source.read_text(encoding="utf-8"))
    if digest != winner["source_sha256"]:
        raise ValueError("Winning source changed after screening")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Frozen reward already exists: {destination}")
    shutil.copyfile(source, destination)
    metadata_path = destination.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol_sha256(protocol),
                "candidate_id": winner_id,
                "generation": winner["generation"],
                "source_sha256": digest,
                "selection_rank": 1,
                "search_status": "CLOSED",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    updated = deepcopy(history)
    updated["status"] = "CLOSED"
    updated["winner"] = {
        "candidate_id": winner_id,
        "generation": winner["generation"],
        "source_sha256": digest,
        "final_reward_path": str(destination),
        "metadata_path": str(metadata_path),
    }
    for item in ranked:
        candidate_by_id(updated, item["candidate_id"])["ranking"] = {
            "final_rank": item["rank"],
            "fields": item["fields"],
        }
    return updated, destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parents = subparsers.add_parser("select-parents")
    parents.add_argument("--for-generation", type=int, required=True)
    freeze = subparsers.add_parser("freeze-winner")
    freeze.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("FINAL_REWARD.py")
    )
    for command in (parents, freeze):
        command.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
        command.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    history = load_history(args.history, protocol)
    if args.command == "select-parents":
        updated, selected = select_parents(history, protocol, args.for_generation)
        save_history(args.history, updated, protocol)
        print("\n".join(selected))
    else:
        updated, path = freeze_winner(history, protocol, destination=args.output)
        save_history(args.history, updated, protocol)
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
