"""Evidence-preserving closure for the resource-infeasible MBRL study."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .checkpointing import file_sha256
from .execution import DEFAULT_OUTPUT_ROOT, atomic_json, scientific_hashes, utc_now

CLASSIFICATION = "TERMINATED_RESOURCE_INFEASIBLE"
RUN_ID = "learned-seed-11"
ATTEMPT_ID = "862895c0-1b25-4e5f-bc40-5d1e5519921a"
REMAINING_RUN_IDS = (
    "learned-seed-29",
    "learned-seed-47",
    "physics-seed-11",
    "physics-seed-29",
    "physics-seed-47",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def close_resource_infeasible_study(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    evidence_path: str | Path = Path(__file__).with_name("termination.json"),
) -> tuple[Path, Path, Path]:
    """Validate the stopped execution and permanently close the mixed study."""

    root = Path(output_root)
    evidence = _load(Path(evidence_path))
    if evidence["classification"] != CLASSIFICATION:
        raise RuntimeError("Termination evidence has an unexpected classification")

    manifest_path = root / "manifest.json"
    launcher_path = root / "launcher-status.json"
    heartbeat_path = root / RUN_ID / "heartbeat.json"
    manifest = _load(manifest_path)
    launcher = _load(launcher_path)
    heartbeat = _load(heartbeat_path)

    run = manifest["runs"][RUN_ID]
    attempt = run["attempts"][-1]
    if attempt["attempt_id"] != ATTEMPT_ID or attempt["state"] != "INTERRUPTED":
        raise RuntimeError("Attempt 3 did not exit through the controlled interrupt")
    if run["state"] != "INTERRUPTED":
        raise RuntimeError("Run is not in the expected post-interrupt state")
    if (root / RUN_ID / ".run.lock").exists():
        raise RuntimeError("Active run lock remains after worker exit")
    if launcher["active"] is not None:
        raise RuntimeError("Matrix launcher still reports an active worker")
    if launcher["state"] != "STOPPED_ON_WORKER_FAILURE":
        raise RuntimeError("Matrix launcher did not stop after the worker exit")
    if launcher["failure"]["terminating_signal"] != 2:
        raise RuntimeError("Worker did not exit through the recorded SIGINT")
    if heartbeat["attempt_id"] != ATTEMPT_ID:
        raise RuntimeError("Final heartbeat belongs to a different attempt")

    observed = evidence["observed_safe_boundary"]
    heartbeat_names = {
        "real_transitions": "real_transition_count",
        "rl_gradient_updates": "rl_update_count",
        "model_gradient_updates": "model_update_count",
        "model_refreshes": "model_refresh_count",
        "synthetic_generated": "synthetic_generated_count",
        "synthetic_sampled": "synthetic_sampled_count",
    }
    for name, heartbeat_name in heartbeat_names.items():
        if int(heartbeat[heartbeat_name]) != int(observed[name]):
            raise RuntimeError(f"Final heartbeat differs for {name}")

    durable = evidence["durable_state"]
    checkpoint = root / durable["checkpoint_relative_path"]
    if file_sha256(checkpoint) != durable["checkpoint_sha256"]:
        raise RuntimeError("Last durable checkpoint hash differs")
    latest = _load(root / RUN_ID / "recovery" / "latest.json")
    if latest["sha256"] != durable["checkpoint_sha256"]:
        raise RuntimeError("Latest recovery pointer differs from termination evidence")
    if int(latest["transition_count"]) != int(durable["real_transitions"]):
        raise RuntimeError("Latest recovery count differs from termination evidence")

    if manifest["validation"] != {"state": "SEALED", "opened_at": None}:
        raise RuntimeError("Validation was opened before termination")
    if manifest["paper_test"] != {"state": "SEALED", "opened_at": None}:
        raise RuntimeError("Paper-test tracks were opened before termination")
    if any(
        manifest["runs"][run_id]["state"] != "NOT_STARTED"
        for run_id in REMAINING_RUN_IDS
    ):
        raise RuntimeError("A remaining policy started before study closure")
    if any(process_alive(int(pid)) for pid in evidence["execution"]["stopped_pids"]):
        raise RuntimeError("An MBRL launcher or worker is still alive")

    scientific_hashes()
    applied_at = utc_now()
    termination_reference = "benchmarks/model_based_rl/termination.json"
    attempt.update(
        {
            "state": CLASSIFICATION,
            "classification": CLASSIFICATION,
            "termination_reason": "COMPUTATIONAL_RESOURCE_INFEASIBILITY",
            "recovery_class": "TERMINATED_RESOURCE_INFEASIBLE_NO_RESUME",
            "termination_decision_at_utc": evidence["decision_at_utc"],
            "safe_boundary_at_utc": evidence["safe_boundary_at_utc"],
            "durable_scientific_work": durable["counters"],
            "actual_consumed_scientific_work": observed,
            "additional_consumed_after_checkpoint": evidence[
                "additional_consumed_after_checkpoint"
            ],
            "known_resource_consumption": evidence["resource_consumption"],
            "termination_artifact": termination_reference,
        }
    )
    run.update(
        {
            "state": CLASSIFICATION,
            "termination_classification": CLASSIFICATION,
            "termination_reason": "COMPUTATIONAL_RESOURCE_INFEASIBILITY",
            "durable_transition_count": int(durable["real_transitions"]),
            "actual_consumed_transition_count": int(observed["real_transitions"]),
            "termination_artifact": termination_reference,
            "resume_prohibited": True,
        }
    )
    manifest.update(
        {
            "state": CLASSIFICATION,
            "terminated_at_utc": applied_at,
            "termination": {
                "classification": CLASSIFICATION,
                "reason": "COMPUTATIONAL_RESOURCE_INFEASIBILITY",
                "decision_at_utc": evidence["decision_at_utc"],
                "safe_boundary_at_utc": evidence["safe_boundary_at_utc"],
                "artifact": termination_reference,
                "validation_state": "SEALED",
                "paper_test_state": "SEALED",
            },
        }
    )
    launcher.update(
        {
            "state": CLASSIFICATION,
            "terminated_at_utc": applied_at,
            "termination": {
                "classification": CLASSIFICATION,
                "reason": "COMPUTATIONAL_RESOURCE_INFEASIBILITY",
                "decision_at_utc": evidence["decision_at_utc"],
                "safe_boundary_at_utc": evidence["safe_boundary_at_utc"],
                "remaining_policies_started": False,
            },
        }
    )
    runtime_evidence = {
        **evidence,
        "applied_at_utc": applied_at,
        "manifest_path": str(manifest_path.resolve()),
        "launcher_status_path": str(launcher_path.resolve()),
    }
    runtime_path = root / "termination.json"
    atomic_json(runtime_path, runtime_evidence)
    atomic_json(launcher_path, launcher)
    atomic_json(manifest_path, manifest)
    return manifest_path, launcher_path, runtime_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("close",))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(__file__).with_name("termination.json"),
    )
    args = parser.parse_args(argv)
    paths = close_resource_infeasible_study(
        output_root=args.output_root, evidence_path=args.evidence
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
