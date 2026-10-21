"""Exact full-continuation execution from immutable causal snapshots."""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.physical_feasibility_reference.causal_controller import (
    decode_observation,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .snapshot import restore_snapshot

DECISION_INTERVAL_NATIVE_STEPS = 10
ADAPTIVE_HIGH_RESERVE_S = 24.0
ADAPTIVE_LOW_RESERVE_S = 2.0
TASK = TaskSpecification(140.0, 0.0)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def write_trace(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    raw = canonical_json_bytes(records)
    encoded = gzip.compress(raw, compresslevel=6, mtime=0)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return {
        "path": str(destination),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(encoded),
        "native_record_count": len(records),
    }


def read_trace(path: Path, expected_sha256: str | None = None) -> list[dict[str, Any]]:
    encoded = Path(path).read_bytes()
    if (
        expected_sha256 is not None
        and hashlib.sha256(encoded).hexdigest() != expected_sha256
    ):
        raise RuntimeError("trace hash mismatch")
    value = json.loads(gzip.decompress(encoded))
    if not isinstance(value, list):
        raise ValueError("native trace must be a list")
    return value


def _state(controller: Any) -> dict[str, float]:
    return asdict(controller.state)


def run_branch(
    snapshot_payload: dict[str, Any],
    *,
    branch_id: str,
    action_id: str | None,
    trace_path: Path | None,
) -> dict[str, Any]:
    """Run one reference or injected decision and then the frozen continuation."""

    (
        environment,
        controller,
        observation,
        info,
        held_price,
        step,
        selector_last_action,
    ) = restore_snapshot(snapshot_payload)
    if step % DECISION_INTERVAL_NATIVE_STEPS:
        raise RuntimeError("snapshot is not a strategic boundary")
    selector = AdaptivePrice(
        high_reserve_s=ADAPTIVE_HIGH_RESERVE_S,
        low_reserve_s=ADAPTIVE_LOW_RESERVE_S,
    )
    selector.last_action = selector_last_action
    if action_id is not None:
        controller.force_next_high_level_action(action_id)
    snapshot_energy = float(info["total_energy_kwh"])
    snapshot_time = float(info["elapsed_time_s"])
    records: list[dict[str, Any]] = []
    actions: list[float] = []
    first_block: list[float] = []
    physics_calls = 0
    first_block_option_memory: dict[str, Any] | None = None
    start_process = time.process_time()
    start_wall = time.perf_counter()
    ended = False
    while not ended:
        boundary = step % DECISION_INTERVAL_NATIVE_STEPS == 0
        state_before = _state(controller)
        transitions = [
            {"position_m": transition.position_m, "limit_m_s": transition.limit_m_s}
            for transition in controller._transitions()
        ]
        current_limit = float(decode_observation(observation)["current_limit_m_s"])
        observation_before = np.asarray(observation, dtype=np.float64).tolist()
        if boundary:
            base = controller.prepare_strategy_step(observation)
            held_price = float(selector.action(base))
        if held_price is None:
            raise RuntimeError("missing held price")
        action = controller.action(observation, held_price)
        decision = controller.last_decision
        if decision is None:
            raise RuntimeError("instrumented controller omitted a decision")
        before_energy = float(info["total_energy_kwh"])
        next_observation, reward, terminated, truncated, next_info = environment.step(
            [action]
        )
        ended = bool(terminated or truncated)
        controller.observe_executed_action(action, episode_ended=ended)
        row = {
            "native_step": int(step),
            "causal_observation": observation_before,
            "causal_state_before": state_before,
            "known_transitions": transitions,
            "current_limit_m_s": current_limit,
            **decision.as_dict(),
            "actual_next_state": _state(controller),
            "actual_next_observation": np.asarray(
                next_observation, dtype=np.float64
            ).tolist(),
            "native_reward": float(reward),
            "signed_step_energy_kwh": float(next_info["total_energy_kwh"])
            - before_energy,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        records.append(row)
        actions.append(float(action))
        if len(first_block) < DECISION_INTERVAL_NATIVE_STEPS:
            first_block.append(float(action))
        if (
            len(first_block) == DECISION_INTERVAL_NATIVE_STEPS
            and first_block_option_memory is None
        ):
            first_block_option_memory = {
                "active_option": (
                    None
                    if controller.active_option is None
                    else controller.active_option.as_dict()
                ),
                "option_history": [
                    option.as_dict() for option in controller.option_history
                ],
                "previous_termination": controller.previous_termination,
                "next_instance_id": controller._next_instance_id,
            }
        physics_calls += int(decision.physics_prediction_calls)
        observation, info = next_observation, next_info
        step += 1
    process_time = time.process_time() - start_process
    wall_time = time.perf_counter() - start_wall
    metrics = EpisodeMetrics(**info["episode_metrics"])
    feasible = is_feasible(metrics, TASK)
    trace = None if trace_path is None else write_trace(trace_path, records)
    option_memory = {
        "active_option": (
            None
            if controller.active_option is None
            else controller.active_option.as_dict()
        ),
        "option_history": [option.as_dict() for option in controller.option_history],
        "previous_termination": controller.previous_termination,
        "next_instance_id": controller._next_instance_id,
    }
    result = {
        "schema_version": 1,
        "branch_id": branch_id,
        "injected_action_id": action_id,
        "actual_first_high_level_action": records[0]["high_level_action"],
        "natural_first_high_level_action": records[0]["natural_high_level_action"],
        "action_injected": bool(records[0]["action_injected"]),
        "native_step_count": len(records),
        "native_step_start": int(snapshot_payload["native_step"]),
        "native_step_end_exclusive": step,
        "remaining_signed_energy_kwh": float(metrics.energy_kwh - snapshot_energy),
        "remaining_travel_time_s": float(metrics.travel_time_s - snapshot_time),
        "final_episode_metrics": asdict(metrics),
        "feasible": feasible,
        "completed": metrics.completed,
        "deadline_compliant": metrics.travel_time_s <= TASK.max_time_s,
        "speed_compliant": (
            metrics.max_speed_violation_m_s <= TASK.max_speed_violation_m_s
        ),
        "first_block_actions": first_block,
        "first_block_action_sha256": hashlib.sha256(
            canonical_json_bytes(first_block)
        ).hexdigest(),
        "first_block_option_memory_sha256": hashlib.sha256(
            canonical_json_bytes(first_block_option_memory)
        ).hexdigest(),
        "native_action_sha256": hashlib.sha256(
            canonical_json_bytes(actions)
        ).hexdigest(),
        "option_memory_sha256": hashlib.sha256(
            canonical_json_bytes(option_memory)
        ).hexdigest(),
        "intervention_count": sum(not row["accepted"] for row in records),
        "intervention_reasons": dict(
            sorted(
                Counter(
                    row["intervention_reason"] for row in records if not row["accepted"]
                ).items()
            )
        ),
        "physics_prediction_calls": physics_calls,
        "active_process_time_s": process_time,
        "wall_time_s": wall_time,
        "trace": trace,
    }
    environment.close()
    return result
