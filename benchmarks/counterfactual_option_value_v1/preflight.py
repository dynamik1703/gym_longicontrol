"""Unreserved-fixture parity and resource gates before scientific labels."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym

import gym_longicontrol  # noqa: F401
from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice

from .alternative_selector import select_alternatives
from .branch_runner import (
    ADAPTIVE_HIGH_RESERVE_S,
    ADAPTIVE_LOW_RESERVE_S,
    read_trace,
    run_branch,
)
from .execution import atomic_json
from .instrumented_controller import InstrumentedContinueController
from .integrity import CANONICAL, HERE, verify_frozen_method, verify_native_records
from .preparation import STATUS
from .snapshot import capture_snapshot, restore_snapshot, snapshot_bytes

FIXTURE_SEED = 12_345_678
FIXTURE_TARGET_M = 430.0
MAX_BRANCHES = 288 * 3


def _fixture_snapshot() -> tuple[dict[str, Any], str, tuple[str, ...]]:
    environment = gym.make("StochasticTrack-v1", max_episode_steps=1_800)
    controller = InstrumentedContinueController()
    selector = AdaptivePrice(
        high_reserve_s=ADAPTIVE_HIGH_RESERVE_S,
        low_reserve_s=ADAPTIVE_LOW_RESERVE_S,
    )
    observation, info = environment.reset(seed=FIXTURE_SEED)
    controller.reset()
    selector.reset()
    held_price = None
    step = 0
    while True:
        boundary = step % 10 == 0
        payload = None
        if boundary and float(info["position_m"]) >= FIXTURE_TARGET_M:
            payload = capture_snapshot(
                environment=environment,
                controller=controller,
                observation=observation,
                info=info,
                held_price=held_price,
                native_step=step,
                selector_last_action=selector.last_action,
            )
        if boundary:
            base = controller.prepare_strategy_step(observation)
            held_price = float(selector.action(base))
        action = controller.action(observation, held_price)
        if payload is not None:
            if controller.last_selection is None or controller.last_decision is None:
                raise RuntimeError("fixture strategic selection failed")
            reference = controller.last_decision.high_level_action
            alternatives = select_alternatives(
                reference_action_id=reference,
                candidates=controller.last_selection.candidates,
            ).alternative_action_ids
            environment.close()
            return payload, reference, alternatives
        observation, _, terminated, truncated, info = environment.step([action])
        controller.observe_executed_action(
            action, episode_ended=bool(terminated or truncated)
        )
        step += 1
        if terminated or truncated:
            raise RuntimeError("fixture did not reach the preregistered target")


def _branch_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "native_step_count",
        "remaining_signed_energy_kwh",
        "remaining_travel_time_s",
        "feasible",
        "native_action_sha256",
        "first_block_action_sha256",
        "first_block_option_memory_sha256",
        "option_memory_sha256",
    )
    return all(left[key] == right[key] for key in keys)


def run_preflight() -> dict[str, Any]:
    verify_frozen_method()
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if canonical["execution"]["preflight_complete"]:
        raise PermissionError("preflight is already complete")
    payload, reference_action, alternatives = _fixture_snapshot()
    if not alternatives:
        raise RuntimeError("fixture has no valid alternative")
    deterministic_snapshot = snapshot_bytes(payload) == snapshot_bytes(payload)
    restored_a = restore_snapshot(payload)
    restored_b = restore_snapshot(payload)
    restoration_equal = (
        restored_a[2].tolist() == restored_b[2].tolist()
        and restored_a[3] == restored_b[3]
        and restored_a[1].snapshot() == restored_b[1].snapshot()
        and restored_a[0].unwrapped.state == restored_b[0].unwrapped.state
    )
    restored_a[0].close()
    restored_b[0].close()
    with tempfile.TemporaryDirectory(prefix="cfov-preflight-") as temporary:
        root = Path(temporary)
        ref_a = run_branch(
            payload,
            branch_id="PREFLIGHT_REFERENCE_A",
            action_id=None,
            trace_path=root / "ref-a.json.gz",
        )
        ref_b = run_branch(
            payload,
            branch_id="PREFLIGHT_REFERENCE_B",
            action_id=None,
            trace_path=root / "ref-b.json.gz",
        )
        injected_reference = run_branch(
            payload,
            branch_id="PREFLIGHT_BASELINE_INJECTION",
            action_id=reference_action,
            trace_path=root / "ref-injected.json.gz",
        )
        alternative = run_branch(
            payload,
            branch_id="PREFLIGHT_ALTERNATIVE",
            action_id=alternatives[0],
            trace_path=root / "alternative.json.gz",
        )
        traces = {
            name: read_trace(root / path, result["trace"]["sha256"])
            for name, path, result in (
                ("ref_a", "ref-a.json.gz", ref_a),
                ("ref_b", "ref-b.json.gz", ref_b),
                ("injected_reference", "ref-injected.json.gz", injected_reference),
                ("alternative", "alternative.json.gz", alternative),
            )
        }
        audits = {name: verify_native_records(trace) for name, trace in traces.items()}
        retained_bytes = sum(
            result["trace"]["size_bytes"]
            for result in (ref_a, ref_b, injected_reference, alternative)
        )
    reference_repeat = _branch_equal(ref_a, ref_b)
    baseline_injection = _branch_equal(ref_a, injected_reference)
    first_ref = traces["ref_a"][0]
    first_alt = traces["alternative"][0]
    injection_isolated = (
        first_ref["causal_state_before"] == first_alt["causal_state_before"]
        and first_ref["causal_observation"] == first_alt["causal_observation"]
        and first_ref["fallback_action"] == first_alt["fallback_action"]
        and first_ref["natural_high_level_action"]
        == first_alt["natural_high_level_action"]
        and first_alt["high_level_action"] == alternatives[0]
    )
    process_times = [
        result["active_process_time_s"]
        for result in (ref_a, ref_b, injected_reference, alternative)
    ]
    projected_hours = max(process_times) * MAX_BRANCHES / 3600.0
    projected_bytes = int(retained_bytes / 4.0 * MAX_BRANCHES * 1.25)
    memory_restore = restore_snapshot(payload)
    option_memory_survives = (
        payload["controller"]["active_option"]
        == memory_restore[1].snapshot()["active_option"]
    )
    memory_restore[0].close()
    checks = {
        "snapshot_serialization_deterministic": deterministic_snapshot,
        "snapshot_restore_exact": restoration_equal,
        "two_reference_continuations_identical": reference_repeat,
        "baseline_action_injection_exact": baseline_injection,
        "alternative_injection_isolated": injection_isolated,
        "option_memory_survives_restore": option_memory_survives,
        "all_native_audits_pass": len(audits) == 4,
        "active_compute_projection_pass": projected_hours <= 12.0,
        "retained_artifact_projection_pass": projected_bytes <= 4_000_000_000,
    }
    if not all(checks.values()):
        raise RuntimeError(f"preflight failed closed: {checks}")
    result = {
        "schema_version": 1,
        "fixture_seed": FIXTURE_SEED,
        "fixture_target_m": FIXTURE_TARGET_M,
        "reference_action_id": reference_action,
        "alternative_action_id": alternatives[0],
        "checks": checks,
        "branch_active_process_time_s": process_times,
        "branch_native_steps": [
            result["native_step_count"]
            for result in (ref_a, ref_b, injected_reference, alternative)
        ],
        "branch_physics_prediction_calls": [
            result["physics_prediction_calls"]
            for result in (ref_a, ref_b, injected_reference, alternative)
        ],
        "projection": {
            "maximum_scientific_branches": MAX_BRANCHES,
            "conservative_basis": "maximum observed fixture branch process time",
            "active_compute_hours": projected_hours,
            "retained_artifact_bytes": projected_bytes,
            "active_compute_limit_hours": 12.0,
            "retained_artifact_limit_bytes": 4_000_000_000,
        },
        "fixture_snapshot_sha256": hashlib.sha256(snapshot_bytes(payload)).hexdigest(),
        "scientific_panel_episode_count": 0,
        "paper_episode_count": 0,
    }
    parity_lines = [
        "# Snapshot Parity",
        "",
        f"Unreserved fixture seed: `{FIXTURE_SEED}` at `{FIXTURE_TARGET_M:.0f} m`.",
        "",
        *[f"- {key}: `{value}`" for key, value in checks.items()],
        "",
        "The alternative branch began from identical physical, observation, RNG,",
        "controller and option memory. Only its first high-level decision was",
        "overridden; later decisions again used frozen causal CONTINUE_FEEDBACK.",
        "",
    ]
    (HERE / "SNAPSHOT_PARITY.md").write_text("\n".join(parity_lines), encoding="utf-8")
    resource_lines = [
        "# Resource Report",
        "",
        "Preflight projection before scientific labels:",
        "",
        f"- Maximum branches: {MAX_BRANCHES}",
        f"- Projected active compute: {projected_hours:.3f} h / 12 h",
        f"- Projected retained artifacts: {projected_bytes} B / 4000000000 B",
        "- Simulator steps, physics predictions and later predictor updates are",
        "  accounted separately in the final result.",
        "",
    ]
    (HERE / "RESOURCE_REPORT.md").write_text(
        "\n".join(resource_lines), encoding="utf-8"
    )
    canonical["execution"]["preflight_complete"] = True
    canonical["preflight"] = result
    atomic_json(CANONICAL, canonical)
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    status["state"] = "PREFLIGHT_PASSED_SCIENTIFIC_LABELS_UNOPENED"
    status["preflight_checks"] = checks
    status["resource_projection"] = result["projection"]
    atomic_json(STATUS, status)
    return result


if __name__ == "__main__":
    print(json.dumps(run_preflight(), sort_keys=True))
