"""Hash verification for the preregistered Physics-only study."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.model_based_rl.checkpointing import file_sha256

FREEZE_PATH = Path(__file__).with_name("freeze.json")


def verify_study_freeze(repository: str | Path = ".") -> dict[str, str]:
    root = Path(repository)
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    expected = {**freeze["parent_files"], **freeze["study_files"]}
    actual = {name: file_sha256(root / name) for name in expected}
    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"Frozen Physics-only sources changed: {mismatches}")
    return actual
