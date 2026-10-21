"""Authorization and scientific-boundary checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.content_split_audit.streams import PAPER_SEEDS

HERE = Path(__file__).resolve().parent
CANONICAL = HERE / "canonical.json"
SPLIT_LEDGER = HERE / "split_ledger.json"


def require_not_paper(seed: int) -> None:
    if int(seed) in PAPER_SEEDS:
        raise PermissionError("PAPER_TEST_SEALED_UNINSPECTED")


def require_validation_authorized() -> None:
    config = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if not config["authorization"]["validation_authorized"]:
        raise PermissionError("SLACKTRADER_VALIDATION_FROZEN_UNOPENED")


def verify_flags() -> dict[str, bool]:
    config = json.loads(CANONICAL.read_text(encoding="utf-8"))
    flags = dict(config["authorization"])
    required_false = (
        "main_training_authorized",
        "main_training_enabled",
        "validation_authorized",
        "paper_authorized",
    )
    if any(flags[name] for name in required_false):
        raise RuntimeError("preparation authorization boundary changed")
    return flags


def verify_split_ledger(path: Path = SPLIT_LEDGER) -> dict[str, Any]:
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if not ledger["route_clean_verified"]:
        raise RuntimeError("SlackTrader split is not route-clean")
    if ledger["paper"]["state"] != "SEALED_UNINSPECTED":
        raise RuntimeError("Paper state changed")
    if ledger["paper"]["content_fingerprints_constructed"] != 0:
        raise RuntimeError("Paper content was inspected")
    if tuple(ledger["paper"]["numeric_exclusion_only"]) != tuple(PAPER_SEEDS):
        raise RuntimeError("Paper numeric exclusion changed")
    if ledger["evaluation_sets"]["validation"]["performance_episode_count"] != 0:
        raise RuntimeError("Validation is no longer unopened")
    return ledger
