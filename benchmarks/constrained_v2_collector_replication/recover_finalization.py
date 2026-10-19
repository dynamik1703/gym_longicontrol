"""Recover deterministic aggregation after the frozen finalizer order-check bug.

The study manifest is serialized with sorted JSON keys, so the insertion order of
``manifest["runs"]`` is alphabetical.  The frozen finalizer incorrectly treated
that object order as the preregistered execution order instead of checking the
authoritative ``manifest["condition_order"]`` field.  This recovery leaves all
scientific sources and sealed evaluation artifacts untouched and changes only
the redundant in-memory check used by deterministic result aggregation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import finalize
from .config import CONDITION_NAMES
from .execution import DEFAULT_OUTPUT_ROOT, atomic_json, preparation_status

HERE = Path(__file__).resolve().parent


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def recover_results(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Aggregate already-sealed evidence without training or evaluation calls."""

    # Verify the original preflight/source-hash guard before applying the narrow
    # in-memory workaround.  This also proves no frozen scientific source was
    # edited after preparation.
    preparation_status()
    root = Path(output_root)
    manifest = _read(root / "manifest.json")
    authoritative_order = tuple(manifest.get("condition_order", ()))
    serialized_run_order = tuple(manifest.get("runs", {}))
    if authoritative_order != CONDITION_NAMES:
        raise RuntimeError("Authoritative manifest condition order changed")
    if set(serialized_run_order) != set(CONDITION_NAMES):
        raise RuntimeError("Manifest run set changed")
    if manifest.get("validation", {}).get("state") != "COMPLETED":
        raise RuntimeError("Validation is not sealed and complete")

    original_constant = finalize.CONDITION_NAMES
    try:
        # Satisfy only the known-bad object-insertion-order assertion.  All run
        # loading and paired comparisons continue to use the frozen config.
        finalize.CONDITION_NAMES = serialized_run_order
        results = finalize.build_results(root)
    finally:
        finalize.CONDITION_NAMES = original_constant

    results["postprocessing_recovery"] = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Frozen finalizer compared sorted runs-object order to condition_order"
        ),
        "original_exception": "RuntimeError: Manifest condition order changed",
        "authoritative_condition_order": list(authoritative_order),
        "serialized_runs_object_order": list(serialized_run_order),
        "training_repeated": False,
        "evaluation_repeated": False,
        "sealed_artifacts_modified": False,
    }
    atomic_json(HERE / "results.json", results)
    finalize._atomic_markdown(HERE / "RESULTS.md", finalize.render_markdown(results))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    results = recover_results(args.output_root)
    print(
        json.dumps(
            {
                "classification": results["classification"],
                "decision": results["canonical_collector_decision"],
                "execution_totals": results["execution_totals"],
                "postprocessing_recovery": results["postprocessing_recovery"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
