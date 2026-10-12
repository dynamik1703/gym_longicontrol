"""Bounded Physics parity confirmation without repeating the preparation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.model_based_rl.checkpointing import file_sha256
from benchmarks.model_based_rl.config import load_configuration as load_parent
from benchmarks.model_based_rl.execution import atomic_json, scientific_hashes, utc_now
from benchmarks.model_based_rl.resource_check import collect_parity_data

DEFAULT_OUTPUT = Path(__file__).with_name("parity_verification.json")
PARENT_ARTIFACT = Path("benchmarks/model_based_rl/physics_parity.json")
BOUNDED_TRANSITIONS = 18


def verify(repository: str | Path = ".") -> dict:
    root = Path(repository)
    parent_path = root / PARENT_ARTIFACT
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if not parent.get("verified"):
        raise RuntimeError("Frozen parent Physics parity artifact is not verified")
    if int(parent["simulator_transitions"]) != 495:
        raise RuntimeError("Frozen parent Physics parity interaction count changed")
    configuration = load_parent()
    bounded, *_unused = collect_parity_data(
        configuration, BOUNDED_TRANSITIONS, targeted_states=True
    )
    if not bounded["verified"]:
        raise RuntimeError("Bounded Physics parity confirmation failed")
    return {
        "schema_version": 1,
        "verified_at_utc": utc_now(),
        "method": "reuse frozen 495-transition artifact plus bounded targeted smoke",
        "parent_artifact_path": str(PARENT_ARTIFACT),
        "parent_artifact_sha256": file_sha256(parent_path),
        "parent_artifact": parent,
        "bounded_confirmation": bounded,
        "new_simulator_transitions": BOUNDED_TRANSITIONS,
        "validation_transitions": 0,
        "paper_test_transitions": 0,
        "scientific_source_sha256": scientific_hashes(root),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = verify(args.repository)
    atomic_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
