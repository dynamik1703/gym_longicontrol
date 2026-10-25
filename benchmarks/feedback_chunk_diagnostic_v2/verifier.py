"""Independent reconstruction of V2 outcomes from native evidence only."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .constants import ABSOLUTE_FLOAT_TOLERANCE
from .utils import canonical_sha256, file_sha256

TASK = TaskSpecification(140.0, 0.0)
STATUS_FIELDS = (
    "proposed_residual",
    "executed_residual",
    "endpoint_speed_check",
    "known_transition_braking_check",
    "unknown_preview_safety_check",
    "deadline_completion_bound_s",
    "viability_flag",
    "shield_intervention",
    "intervention_or_cancellation_reason",
)
STATE_FIELDS = (
    "position_m",
    "velocity_m_s",
    "acceleration_m_s2",
    "elapsed_time_s",
    "total_energy_kwh",
)
OUTCOME_FIELDS = (
    "completed",
    "travel_time_s",
    "energy_kwh",
    "speed_violation_count",
    "max_speed_violation_m_s",
    "integrated_speed_violation_m",
    "feasible",
    "feasible_energy_kwh",
    "deadline_compliant",
    "speed_compliant",
)


def _close(left: float, right: float) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=ABSOLUTE_FLOAT_TOLERANCE,
    )


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is not finite")
    return result


def _validate_status(field: str, wrapped: Any) -> None:
    if not isinstance(wrapped, dict):
        raise RuntimeError(f"{field} lacks explicit applicability metadata")
    status = wrapped.get("status")
    if status == "APPLICABLE":
        if "value" not in wrapped or wrapped["value"] is None:
            raise RuntimeError(f"{field} is applicable but missing")
    elif status == "NOT_APPLICABLE":
        if wrapped.get("value") is not None or not wrapped.get("reason"):
            raise RuntimeError(f"{field} has invalid NOT_APPLICABLE encoding")
    else:
        raise RuntimeError(f"{field} has invalid applicability status")


def verify_native_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute all physical episode accounting without trusting the outcome."""

    if not rows:
        raise RuntimeError("native evidence is empty")
    previous: dict[str, Any] | None = None
    violation_active = False
    violation_count = 0
    maximum_excess = 0.0
    integrated_excess = 0.0
    energy_sum = 0.0
    controller_id = rows[0].get("controller_id")
    for expected_step, row in enumerate(rows):
        if row.get("schema_version") != 1:
            raise RuntimeError("native evidence schema changed")
        if row.get("controller_id") != controller_id:
            raise RuntimeError("controller changed within an episode")
        if row.get("native_step") != expected_step:
            raise RuntimeError("native steps are not contiguous")
        for field in STATUS_FIELDS:
            _validate_status(field, row.get(field))
        action = _finite_number(row.get("final_actuator_action"), "action")
        if not -1.0 <= action <= 1.0:
            raise RuntimeError("final actuator action is outside [-1, 1]")
        proposed = _finite_number(
            row.get("proposed_controller_action"), "proposed action"
        )
        if not -1.0 <= proposed <= 1.0:
            raise RuntimeError("proposed action is outside [-1, 1]")
        pre = row.get("causal_pre_state")
        post = row.get("actual_successor_state")
        if not isinstance(pre, dict) or not isinstance(post, dict):
            raise RuntimeError("causal state evidence is incomplete")
        if previous is not None:
            for field in STATE_FIELDS:
                if not _close(previous[field], pre[field]):
                    raise RuntimeError(f"native state continuity failed at {field}")
        visible = row.get("visible_speed_limits")
        if (
            not isinstance(visible, dict)
            or len(visible.get("future_m_s", [])) != 2
            or len(visible.get("transition_distances_m", [])) != 2
        ):
            raise RuntimeError("visible speed-limit evidence is incomplete")
        velocity = _finite_number(post.get("velocity_m_s"), "successor velocity")
        limit = _finite_number(post.get("speed_limit_m_s"), "successor speed limit")
        excess = max(0.0, velocity - limit)
        if excess > 0.0 and not violation_active:
            violation_count += 1
        violation_active = excess > 0.0
        maximum_excess = max(maximum_excess, excess)
        integrated_excess += excess * 0.1
        energy_sum += _finite_number(
            row.get("signed_step_energy_kwh"), "signed step energy"
        )
        previous = post
        terminal = bool(row.get("terminated") or row.get("truncated"))
        if terminal != (expected_step == len(rows) - 1):
            raise RuntimeError("termination marker does not delimit the evidence")
    assert previous is not None
    completed = bool(rows[-1]["terminated"])
    if rows[-1]["truncated"] and completed:
        raise RuntimeError("episode cannot terminate and truncate simultaneously")
    if not _close(energy_sum, previous["total_energy_kwh"]):
        raise RuntimeError("signed native energy does not match successor state")
    metrics = EpisodeMetrics(
        completed=completed,
        travel_time_s=float(previous["elapsed_time_s"]),
        energy_kwh=float(previous["total_energy_kwh"]),
        speed_violation_count=violation_count,
        max_speed_violation_m_s=maximum_excess,
        integrated_speed_violation_m=integrated_excess,
    )
    feasible = is_feasible(metrics, TASK)
    result = {
        "completed": metrics.completed,
        "travel_time_s": metrics.travel_time_s,
        "energy_kwh": metrics.energy_kwh,
        "speed_violation_count": metrics.speed_violation_count,
        "max_speed_violation_m_s": metrics.max_speed_violation_m_s,
        "integrated_speed_violation_m": metrics.integrated_speed_violation_m,
        "feasible": feasible,
        "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
        "deadline_compliant": metrics.travel_time_s <= TASK.max_time_s,
        "speed_compliant": (
            metrics.max_speed_violation_m_s <= TASK.max_speed_violation_m_s
        ),
        "native_step_count": len(rows),
        "final_position_m": float(previous["position_m"]),
    }
    result["reconstruction_sha256"] = canonical_sha256(result)
    return result


def verify_outcome(
    outcome: dict[str, Any], reconstruction: dict[str, Any]
) -> dict[str, Any]:
    missing = sorted(set(OUTCOME_FIELDS) - set(outcome))
    if missing:
        raise RuntimeError(f"outcome fields are missing: {missing}")
    for field in OUTCOME_FIELDS:
        observed = outcome[field]
        expected = reconstruction[field]
        if isinstance(expected, float):
            if not _close(observed, expected):
                raise RuntimeError(f"outcome mismatch: {field}")
        elif observed != expected:
            raise RuntimeError(f"outcome mismatch: {field}")
    return {
        "verified": True,
        "outcome_sha256": canonical_sha256(outcome),
        "reconstruction_sha256": reconstruction["reconstruction_sha256"],
    }


def read_evidence(path: Path) -> list[dict[str, Any]]:
    payload = gzip.decompress(path.read_bytes()).decode("utf-8")
    return [json.loads(line) for line in payload.splitlines() if line]


def verify_files(result_path: Path, evidence_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["evidence_sha256"] != file_sha256(evidence_path):
        raise RuntimeError("compressed evidence hash mismatch")
    rows = read_evidence(evidence_path)
    if result["native_evidence_sha256"] != canonical_sha256(rows):
        raise RuntimeError("native evidence content hash mismatch")
    reconstruction = verify_native_evidence(rows)
    verification = verify_outcome(result["outcome"], reconstruction)
    expected = canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    if result.get("result_sha256") != expected:
        raise RuntimeError("result hash mismatch")
    return {
        **verification,
        "result_sha256": expected,
        "evidence_file_sha256": file_sha256(evidence_path),
        "native_step_count": len(rows),
    }
