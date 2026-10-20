"""Materialize all preparation-only gates without opening reserved evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.hybrid_rl_design.preflight import run_preflight
from benchmarks.physical_feasibility_reference.diagnostics import (
    source_hashes as physical_source_hashes,
)

from .evaluation import verify_frozen_splits_without_opening
from .execution import configuration_sha256, source_hashes
from .macro_transition import (
    GAMMA_DECISION,
    GAMMA_NATIVE,
    MacroAccumulator,
    cost_bellman_target,
    reward_bellman_target,
)
from .resource import benchmark_actual_pipeline
from .runner import UpdateBudgetAccumulator, cumulative_update_target
from .safety import run_safety_audit
from .verification import causal_information_boundary, exact_resume_probe

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "canonical.json"
STATUS = HERE / "preparation_status.json"
SAFETY_JSON = HERE / "safety_audit.json"
RESOURCE_JSON = HERE / "resource_measurements.json"
RECOVERY_JSON = HERE / "recovery_probe.json"
PREFLIGHT_JSON = HERE / "preflight_results.json"
PHYSICAL_FREEZE = ROOT / "benchmarks/physical_feasibility_reference/freeze.json"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _physical_freeze() -> dict[str, Any]:
    frozen = json.loads(PHYSICAL_FREEZE.read_text(encoding="utf-8"))
    observed = physical_source_hashes()
    expected_configuration = (
        "f63717504a810c75d7080388aeb159505cf667ad13248dbe4a53ec0c4bb94cd1"
    )
    verified = bool(
        frozen["source_sha256"] == observed
        and frozen["scientific_configuration_sha256"] == expected_configuration
    )
    return {
        "verified": verified,
        "source_sha256": observed,
        "scientific_configuration_sha256": expected_configuration,
    }


def _temporal_semantics_checks() -> dict[str, Any]:
    full = MacroAccumulator(np.zeros(16), 0.25)
    for _ in range(10):
        full.add_native_step(
            reward=1.0,
            speed_cost=2.0,
            terminal_cost=0.0,
            intervention_cost=0.5,
        )
    full_transition = full.finish(np.ones(16))
    shortened = MacroAccumulator(np.zeros(16), -0.5)
    for index in range(4):
        shortened.add_native_step(
            reward=float(index + 1),
            speed_cost=0.1 * index,
            terminal_cost=float(index == 3),
            intervention_cost=0.2,
            terminated=index == 3,
        )
    shortened_transition = shortened.finish(np.zeros(16))
    reward = np.array([[3.0]])
    discount = np.array([[GAMMA_NATIVE**4]])
    next_q = np.array([[5.0]])
    next_log_probability = np.array([[-0.7]])
    alpha = 0.2
    computed_reward_target = reward_bellman_target(
        reward, discount, next_q, next_log_probability, alpha
    )
    hand_reward_target = reward + discount * (next_q - alpha * next_log_probability)
    computed_cost_target = cost_bellman_target(
        np.array([[0.4]]), discount, np.array([[2.0]])
    )
    hand_cost_target = np.array([[0.4]]) + discount * np.array([[2.0]])
    scheduler = UpdateBudgetAccumulator()
    update_checkpoints = (
        9_999,
        10_000,
        10_009,
        10_010,
        50_000,
        100_000,
        150_000,
        200_000,
        250_000,
        299_999,
        300_000,
    )
    update_counts = {
        str(native_transitions): scheduler.cumulative_target(native_transitions)
        for native_transitions in update_checkpoints
    }
    checks = {
        "gamma_native_tenth_power_is_gamma_decision": math.isclose(
            GAMMA_NATIVE**10, GAMMA_DECISION, abs_tol=1e-15
        ),
        "full_block_continuation_discount": math.isclose(
            full_transition.continuation_discount,
            GAMMA_DECISION,
            abs_tol=1e-15,
        ),
        "short_terminal_block_has_no_bootstrap": (
            shortened_transition.native_steps == 4
            and shortened_transition.continuation_discount == 0.0
        ),
        "reward_target_matches_hand_calculation": np.array_equal(
            computed_reward_target, hand_reward_target
        ),
        "cost_target_matches_hand_calculation": np.array_equal(
            computed_cost_target, hand_cost_target
        ),
        "no_updates_before_warmup": scheduler.cumulative_target(9_999) == 0,
        "no_updates_at_warmup": scheduler.cumulative_target(10_000) == 0,
        "no_update_before_first_post_warmup_target": (
            scheduler.cumulative_target(10_009) == 0
        ),
        "first_update_at_native_transition_10010": (
            scheduler.cumulative_target(10_010) == 1
        ),
        "canonical_function_matches_accumulator": all(
            cumulative_update_target(native_transitions)
            == scheduler.cumulative_target(native_transitions)
            for native_transitions in (0, *update_checkpoints)
        ),
        "exact_final_update_budget": scheduler.cumulative_target(300_000) == 30_000,
    }
    return {
        "gamma_decision": GAMMA_DECISION,
        "gamma_native": GAMMA_NATIVE,
        "full_block_discount": full_transition.continuation_discount,
        "shortened_terminal_native_steps": shortened_transition.native_steps,
        "shortened_terminal_discount": shortened_transition.continuation_discount,
        "cumulative_update_targets": update_counts,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _resource_report(payload: dict[str, Any]) -> str:
    actual = payload["actual_pipeline"]
    future = payload["future_complete_three_policy_study"]
    return "\n".join(
        [
            "# Hybrid RL V1 Actual-Pipeline Resource Report",
            "",
            "This bounded preparation benchmark used only the unreserved",
            "`consecutive_downward_changes` physical fixture. It did not run a",
            "frozen Diagnostic, Validation, Paper, or main-training condition.",
            "",
            (
                f"- Real fixture collection: `{actual['native_transitions']}` native "
                f"transitions and `{actual['macro_transitions']}` macro transitions."
            ),
            (
                f"- Actual constrained-SAC updates: `{actual['gradient_updates']}` "
                f"at `{actual['mean_actual_sac_update_time_s']:.6f} s/update`."
            ),
            (
                "- Full-capacity replay checkpoint: "
                f"`{actual['checkpoint_size_bytes']}` bytes; write "
                f"`{actual['checkpoint_write_s']:.6f} s`; restore "
                f"`{actual['checkpoint_restore_s']:.6f} s`."
            ),
            (
                "- Actor-only unreserved fixture evaluation: "
                f"`{actual['actor_only_fixture_evaluation']['wall_time_s']:.6f} s`."
            ),
            (
                "- Projected complete study runtime (2× factor): "
                f"`{future['projected_total_runtime_hours']:.3f} h`."
            ),
            (
                "- Projected retained disk: "
                f"`{future['projected_retained_disk_gb']:.3f} GB`."
            ),
            (
                f"- Gates: runtime `{payload['runtime_gate_passed']}`; "
                f"disk `{payload['disk_gate_passed']}`."
            ),
            "",
        ]
    )


def _safety_report(payload: dict[str, Any], frozen: dict[str, Any]) -> str:
    lines = [
        "# Hybrid RL V1 Safety and Viability Audit",
        "",
        "The frozen causal controller and Hybrid shield were not modified. This",
        "audit is deterministic simulator evidence, not a formal global safety proof.",
        "",
        f"- Extended adversarial cases: `{len(payload['rows'])}`.",
        (
            "- Exact right-endpoint speed compliance: "
            f"`{payload['exact_right_endpoint_speed_passed']}`."
        ),
        f"- Eventual completion: `{payload['eventual_completion_passed']}`.",
        f"- Deadline compliance: `{payload['deadline_passed']}`.",
        (
            "- Deadline-bound falsifications: "
            f"`{len(payload['viability_falsifications'])}`."
        ),
        "- Stress coverage: repeated maximum legal reductions, slow sections,",
        "  consecutive transitions, near-deadline/20 km/h proposals, gradual",
        "  reserve consumption, fallback after accepted slow actions, low-speed",
        "  acceleration, and an endpoint-interrupted macro block.",
        "",
        "Frozen six-fixture shield reproduction:",
        "",
        f"- override fraction: `{frozen['override_fraction']:.12f}`;",
        f"- native-action influence fraction: `{frozen['influence_fraction']:.12f}`;",
        f"- influenced fixtures: `{frozen['influenced_fixture_count']}`;",
        f"- viability falsifications: `{len(frozen['viability_falsifications'])}`.",
        "",
        "Any future falsification is a terminal gate failure; margins, deadline,",
        "or zero speed tolerance may not be relaxed to rescue the study.",
        "",
    ]
    return "\n".join(lines)


def prepare() -> dict[str, Any]:
    canonical = json.loads(CANONICAL.read_text(encoding="utf-8"))
    authorization = canonical["authorization"]
    if (
        authorization["main_training_authorized"]
        or authorization["main_training_enabled"]
    ):
        raise PermissionError("preparation may not authorize or enable main training")
    if authorization["validation_authorized"] or authorization["paper_authorized"]:
        raise PermissionError("preparation may not open Validation or Paper")
    physical = _physical_freeze()
    temporal = _temporal_semantics_checks()
    frozen_preflight = run_preflight()
    safety = run_safety_audit()
    boundary = causal_information_boundary()
    splits = verify_frozen_splits_without_opening()
    recovery = exact_resume_probe()
    resources = benchmark_actual_pipeline()
    _atomic_json(PREFLIGHT_JSON, frozen_preflight)
    _atomic_json(SAFETY_JSON, safety)
    _atomic_json(RECOVERY_JSON, recovery)
    _atomic_json(RESOURCE_JSON, resources)
    _atomic_text(HERE / "RESOURCE_REPORT.md", _resource_report(resources))
    _atomic_text(
        HERE / "SAFETY_AUDIT.md",
        _safety_report(safety, frozen_preflight["shield"]),
    )
    gates = {
        "frozen_physical_source_and_configuration": physical["verified"],
        "temporal_smdp_analytical_targets": temporal["passed"],
        "fallback_action_trace_metric_energy_parity": frozen_preflight[
            "fallback_parity"
        ]["all_exact"],
        "frozen_shield_fixtures": frozen_preflight["shield"]["endpoint_safety_passed"]
        and frozen_preflight["shield"]["deadline_fixture_passed"]
        and not frozen_preflight["shield"]["viability_falsifications"],
        "extended_deadline_viability_stress": safety["passed"],
        "shield_not_dominating_fixture_strategy": not frozen_preflight["shield"][
            "SHIELD_DOMINATES_POLICY"
        ],
        "causal_information_boundary": boundary["passed"],
        "diagnostic_split_hash": splits["diagnostic"]["ordered_sha256"]
        == canonical["splits"]["diagnostic_ordered_sha256"],
        "validation_split_frozen_unopened": (
            splits["validation"]["ordered_sha256"]
            == canonical["splits"]["validation_ordered_sha256"]
            and splits["validation"]["performance_episodes"] == 0
        ),
        "paper_fail_closed": (
            splits["paper"]["state"] == "SEALED_UNINSPECTED"
            and splits["paper"]["content_fingerprints_constructed"] == 0
        ),
        "exact_mid_block_recovery": recovery["passed"],
        "actual_pipeline_runtime_within_24_hours": resources["runtime_gate_passed"],
        "retained_disk_within_12_gb": resources["disk_gate_passed"],
        "main_training_disabled": not authorization["main_training_enabled"],
        "main_training_unauthorized": not authorization["main_training_authorized"],
        "validation_unauthorized": not authorization["validation_authorized"],
        "paper_unauthorized": not authorization["paper_authorized"],
    }
    ready = bool(authorization["ready_for_main_training"] and all(gates.values()))
    frozen_steps = sum(
        row["steps"] for row in frozen_preflight["fallback_parity"]["fixtures"]
    )
    safety_steps = sum(row["native_proposal_count"] for row in safety["rows"])
    evaluation_steps = int(
        resources["actual_pipeline"]["actor_only_fixture_evaluation"]["native_steps"]
    )
    preparation_native_steps = (
        2 * frozen_steps
        + frozen_preflight["shield"]["proposal_count"]
        + safety_steps
        + resources["preparation_interactions"]["real_native_transitions"]
        + evaluation_steps
        + recovery["preparation_interactions"]["real_native_transitions"]
    )
    status = {
        "schema_version": 2,
        "protocol_revision": canonical["protocol_revision"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": canonical["name"],
        "source_commit": canonical["lineage"]["source_commit"],
        "configuration_sha256": configuration_sha256(),
        "scientific_source_sha256": source_hashes(),
        "physical_fallback": physical,
        "temporal_semantics": temporal,
        "gates": gates,
        "ready_for_main_training": ready,
        "main_training_authorized": False,
        "main_training_enabled": False,
        "validation_authorized": False,
        "paper_authorized": False,
        "main_training_started": False,
        "main_training_real_native_transitions": 0,
        "main_training_gradient_updates": 0,
        "diagnostic": {
            **splits["diagnostic"],
            "performance_episodes": 0,
        },
        "validation": {
            **splits["validation"],
            "state": "FROZEN_UNOPENED",
            "performance_episodes": 0,
        },
        "paper": {**splits["paper"], "episode_count": 0},
        "shield_reproduction": {
            key: frozen_preflight["shield"][key]
            for key in (
                "proposal_count",
                "override_count",
                "override_fraction",
                "influenced_step_count",
                "influence_fraction",
                "influenced_fixture_count",
                "SHIELD_DOMINATES_POLICY",
            )
        },
        "exact_recovery": recovery,
        "resources": resources["future_complete_three_policy_study"],
        "preparation_interactions_consumed": {
            "real_environment_native_steps": preparation_native_steps,
            "sac_gradient_updates": (
                resources["preparation_interactions"]["gradient_updates"]
                + recovery["preparation_interactions"]["gradient_updates"]
            ),
            "reserved_diagnostic_episodes": 0,
            "reserved_validation_episodes": 0,
            "paper_episodes": 0,
            "used_only_deterministic_unreserved_fixtures": True,
        },
        "artifacts": {
            path.name: _sha256(path)
            for path in (
                PREFLIGHT_JSON,
                SAFETY_JSON,
                RECOVERY_JSON,
                RESOURCE_JSON,
            )
        },
    }
    if not ready:
        status["terminal_failure"] = "one or more mandatory preparation gates failed"
    _atomic_json(STATUS, status)
    return status


def verify() -> dict[str, Any]:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if status["configuration_sha256"] != configuration_sha256():
        raise RuntimeError("Hybrid execution configuration changed")
    if status["scientific_source_sha256"] != source_hashes():
        raise RuntimeError("Hybrid execution source changed")
    if not status["ready_for_main_training"] or not all(status["gates"].values()):
        raise RuntimeError("Hybrid execution preparation is not ready")
    if any(
        (
            status["main_training_authorized"],
            status["main_training_enabled"],
            status["validation_authorized"],
            status["paper_authorized"],
            status["main_training_started"],
            status["main_training_real_native_transitions"],
            status["main_training_gradient_updates"],
            status["validation"]["performance_episodes"],
            status["paper"]["episode_count"],
        )
    ):
        raise RuntimeError("preparation authorization boundary was crossed")
    return status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "verify"))
    arguments = parser.parse_args(argv)
    payload = prepare() if arguments.stage == "prepare" else verify()
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
