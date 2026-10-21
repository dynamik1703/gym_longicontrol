"""Aggregation-only integrity audit; never executes a controller episode."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .execution import atomic_json
from .integrity import CANONICAL, HERE, PAIR_LEDGER


def apply_integrity_stop() -> dict:
    pair_ledger = json.loads(PAIR_LEDGER.read_text(encoding="utf-8"))
    if pair_ledger["state"] != "COMPLETE" or any(
        pair["state"] != "COMPLETE" for pair in pair_ledger["pairs"].values()
    ):
        raise RuntimeError("post-Capability audit requires 72 completed pairs")
    path = HERE / "capability_results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    original = results["scientific_classification"]
    incident = {
        "id": "OPTIONS_V1_DEADLINE_BOUND_RETENTION_20261021",
        "detected_utc": datetime.now(timezone.utc).isoformat(),
        "affected_metric": "deadline-bound falsifications",
        "finding": (
            "Per-step completion_bound_s was computed online but omitted from "
            "immutable Capability records, so empty falsification lists cannot "
            "be independently recomputed after execution."
        ),
        "raw_pair_records_modified": False,
        "pair_reruns": 0,
        "physics_or_controller_changes": 0,
        "aggregation_action": "fail closed",
        "original_automatic_classification": original,
        "corrected_classification": "SEMANTIC_OR_INTEGRITY_STOP",
    }
    results["scientific_classification"] = "SEMANTIC_OR_INTEGRITY_STOP"
    results["integrity_incident"] = incident
    results["measurement_availability"] = {
        "deadline_bound_falsifications": "UNAVAILABLE_POST_HOC",
        "all_other_registered_episode_and_option_metrics": "AVAILABLE",
    }
    for arm in results["arms"].values():
        arm["summary"]["deadline_bound_falsification_count"] = None
    atomic_json(path, results)
    atomic_json(HERE / "results.json", results)
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    canonical["execution"]["integrity_stop"] = incident["id"]
    atomic_json(CANONICAL, canonical)
    atomic_json(HERE / "integrity_incident.json", incident)
    return results


if __name__ == "__main__":
    apply_integrity_stop()
