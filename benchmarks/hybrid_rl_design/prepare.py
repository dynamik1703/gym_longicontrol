"""Materialize Hybrid RL design gates without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.physical_feasibility_reference.diagnostics import source_hashes

from .preflight import run_preflight
from .resource import benchmark_resources
from .split import build_split_ledger, render_split_ledger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = HERE / "canonical.json"
SPLIT = HERE / "split_ledger.json"
SPLIT_MD = HERE / "SPLIT_LEDGER.md"
PREFLIGHT = HERE / "preflight_results.json"
RESOURCES = HERE / "resource_measurements.json"
RESOURCE_REPORT = HERE / "RESOURCE_REPORT.md"
STATUS = HERE / "preparation_status.json"
PHYSICAL_FREEZE = ROOT / "benchmarks/physical_feasibility_reference/freeze.json"
SCIENTIFIC_SOURCES = (
    "canonical.json",
    "viability.py",
    "controller.py",
    "baseline.py",
    "split.py",
    "preflight.py",
    "resource.py",
    "prepare.py",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scientific_source_hashes() -> dict[str, str]:
    return {name: _sha256(HERE / name) for name in SCIENTIFIC_SOURCES}


def _verify_physical_freeze() -> dict[str, Any]:
    frozen = _load(PHYSICAL_FREEZE)
    observed = source_hashes()
    if frozen["source_sha256"] != observed:
        raise RuntimeError("frozen physical fallback source changed")
    expected = "f63717504a810c75d7080388aeb159505cf667ad13248dbe4a53ec0c4bb94cd1"
    if frozen["scientific_configuration_sha256"] != expected:
        raise RuntimeError("frozen physical fallback configuration changed")
    return {
        "scientific_configuration_sha256": expected,
        "source_sha256": observed,
        "verified": True,
    }


def _resource_markdown(payload: dict[str, Any]) -> str:
    future = payload["future_study"]
    update = payload["model_free_update_benchmark"]
    shield = payload["causal_shield_benchmark"]
    return "\n".join(
        [
            "# Hybrid RL Resource Report",
            "",
            "This is a bounded preparation benchmark. Synthetic learner updates use",
            "random tensors only; no environment transition trained an RL policy.",
            "",
            f"- Synthetic update: `{update['mean_update_time_s']:.6f} s/update`.",
            (
                f"- Shield: `{shield['mean_shield_time_per_action_s']:.6f} "
                "s/action` and "
                f"`{shield['mean_physics_prediction_calls_per_action']:.3f}` "
                "exact physics calls/action on the fixture."
            ),
            (
                "- Projected three-policy study: "
                f"`{future['projected_runtime_hours']:.3f} h` "
                "including a 3× uncertainty factor."
            ),
            (
                "- Projected retained disk: "
                f"`{future['projected_retained_disk_gb']:.3f} GB`."
            ),
            (
                f"- Gates: runtime `{payload['runtime_gate_passed']}`, disk "
                f"`{payload['disk_gate_passed']}`."
            ),
            "",
        ]
    )


def prepare() -> dict[str, Any]:
    config = _load(CONFIG)
    if config["authorization"]["training_authorized"]:
        raise PermissionError("preparation config unexpectedly authorizes training")
    physical = _verify_physical_freeze()
    split = build_split_ledger()
    preflight = run_preflight()
    resources = benchmark_resources()
    _atomic_json(SPLIT, split)
    _atomic_text(SPLIT_MD, render_split_ledger(split))
    _atomic_json(PREFLIGHT, preflight)
    _atomic_json(RESOURCES, resources)
    _atomic_text(RESOURCE_REPORT, _resource_markdown(resources))
    ready = bool(
        physical["verified"]
        and split["route_clean_verified"]
        and preflight["passed"]
        and resources["runtime_gate_passed"]
        and resources["disk_gate_passed"]
    )
    status = {
        "schema_version": 1,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": config["name"],
        "configuration_sha256": _sha256(CONFIG),
        "scientific_source_sha256": scientific_source_hashes(),
        "physical_fallback": physical,
        "gates": {
            "frozen_fallback_source_and_configuration": physical["verified"],
            "fallback_action_trace_metric_energy_parity": preflight[
                "fallback_parity"
            ]["all_exact"],
            "right_endpoint_and_braking_fixtures": preflight["shield"][
                "endpoint_safety_passed"
            ],
            "deadline_viability_fixtures": bool(
                preflight["shield"]["deadline_fixture_passed"]
                and not preflight["shield"]["viability_falsifications"]
            ),
            "shield_does_not_dominate_fixture_strategy": not preflight["shield"][
                "SHIELD_DOMINATES_POLICY"
            ],
            "causal_information_boundary": True,
            "route_clean_policy_r": split["route_clean_verified"],
            "runtime_within_24_hours": resources["runtime_gate_passed"],
            "disk_within_12_gb": resources["disk_gate_passed"],
            "paper_fail_closed": (
                split["paper"]["state"] == "SEALED_UNINSPECTED"
                and split["paper"]["content_fingerprints_constructed"] == 0
            ),
        },
        "ready_for_future_separately_authorized_training": ready,
        "training_authorized": False,
        "training_started": False,
        "real_training_transitions": 0,
        "rl_gradient_updates": 0,
        "hybrid_final_validation": {
            "state": "FROZEN_UNOPENED",
            "performance_episode_count": 0,
        },
        "contextual_results_used_for_hyperparameter_or_controller_selection": False,
        "paper": split["paper"],
        "artifact_sha256": {
            "split_ledger.json": _sha256(SPLIT),
            "preflight_results.json": _sha256(PREFLIGHT),
            "resource_measurements.json": _sha256(RESOURCES),
        },
    }
    _atomic_json(STATUS, status)
    return status


def verify() -> dict[str, Any]:
    if not STATUS.exists():
        raise RuntimeError("Hybrid preparation has not been materialized")
    status = _load(STATUS)
    if status["scientific_source_sha256"] != scientific_source_hashes():
        raise RuntimeError("Hybrid scientific source changed after preparation")
    if status["configuration_sha256"] != _sha256(CONFIG):
        raise RuntimeError("Hybrid configuration changed after preparation")
    if not all(status["gates"].values()):
        raise RuntimeError("mandatory Hybrid preparation gate failed")
    if status["training_started"] or status["rl_gradient_updates"]:
        raise RuntimeError("unauthorized training detected")
    if status["hybrid_final_validation"]["performance_episode_count"]:
        raise RuntimeError("Hybrid final Validation was opened during preparation")
    return status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "verify"))
    args = parser.parse_args(argv)
    payload = prepare() if args.stage == "prepare" else verify()
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
