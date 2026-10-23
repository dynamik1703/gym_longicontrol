"""Mandatory synthetic correctness, recovery, stability, and resource gates."""

from __future__ import annotations

import gzip
import json
import math
import resource
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.causal_mpc_v2.controller import PriceMPCController
from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.domain.track import Track
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from .checkpointing import file_sha256
from .chunk_decoder import RESIDUAL_SCALE
from .execution import atomic_json, utc_now
from .integrity import (
    AUTHORIZATION,
    CANONICAL,
    HERE,
    PREPARATION_STATUS,
    SPLIT_LEDGER,
    scientific_source_hashes,
    sha256_file,
    verify_repository,
)
from .macro_transition import GAMMA_NATIVE, MacroAccumulator
from .residual_controller import FeedbackResidualController
from .sac_agent import ConstrainedResidualSAC
from .training import TrainingRunner

PREFLIGHT_RESULTS = HERE / "preflight_results.json"
RESOURCE_MEASUREMENTS = HERE / "resource_measurements.json"
RESOURCE_REPORT = HERE / "RESOURCE_REPORT.md"
CANONICAL_TASK = TaskSpecification(140.0, 0.0)
PARITY_FIXTURES = (
    "constant_100_km_h",
    "upward_changes",
    "maximum_legal_downward_change",
    "consecutive_downward_changes",
    "short_braking_window",
    "route_endpoint_at_speed_boundary",
)


def _environment(track: Track) -> gym.Env:
    return gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=lambda _rng: track),
        max_episode_steps=1_800,
    )


def _trace(info: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(info["position_m"]),
        float(info["velocity_m_s"]),
        float(info["acceleration_m_s2"]),
        float(info["total_energy_kwh"]),
        float(info["speed_limit_m_s"]),
        float(info["elapsed_time_s"]),
    )


def _adaptive_rollout(track: Track) -> dict[str, Any]:
    environment = _environment(track)
    controller = PriceMPCController()
    selector = AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
    rows = []
    try:
        observation, _ = environment.reset(seed=20261023)
        controller.reset()
        selector.reset()
        held = 1.0
        step = 0
        ended = False
        while not ended:
            if step % 10 == 0:
                held = selector.action(controller.prepare_strategy_step(observation))
            action = controller.action(observation, held)
            next_observation, _, terminated, truncated, info = environment.step(
                [action]
            )
            controller.observe_executed_action(action)
            rows.append(
                {
                    "action": action,
                    "trace": _trace(info),
                    "mpc_decision": controller.last_decision.as_dict(),
                }
            )
            observation = next_observation
            step += 1
            ended = bool(terminated or truncated)
        return {
            "rows": rows,
            "metrics": asdict(EpisodeMetrics(**info["episode_metrics"])),
        }
    finally:
        environment.close()


def _residual_rollout(track: Track, actor_action: np.ndarray) -> dict[str, Any]:
    environment = _environment(track)
    controller = FeedbackResidualController("CHUNK3_RESIDUAL_SAC")
    rows = []
    try:
        observation, _ = environment.reset(seed=20261023)
        controller.reset()
        ended = False
        while not ended:
            actor_observation = controller.prepare_native_step(observation)
            if actor_observation is not None:
                controller.begin_chunk(actor_action)
            action = controller.action(observation)
            decision = controller.last_decision
            next_observation, _, terminated, truncated, info = environment.step(
                [action]
            )
            controller.observe_executed_action(action)
            rows.append(
                {
                    "action": action,
                    "trace": _trace(info),
                    "mpc_decision": decision.mpc_decision,
                    "residual_decision": decision.as_dict(),
                }
            )
            observation = next_observation
            ended = bool(terminated or truncated)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        return {
            "rows": rows,
            "metrics": asdict(metrics),
            "feasible": is_feasible(metrics, CANONICAL_TASK),
        }
    finally:
        environment.close()


def _canonical_hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def zero_residual_parity() -> dict[str, Any]:
    results = []
    for name in PARITY_FIXTURES:
        baseline = _adaptive_rollout(physical_fixtures()[name])
        residual = _residual_rollout(physical_fixtures()[name], np.zeros(3))
        comparable_residual = {
            "rows": [
                {
                    "action": row["action"],
                    "trace": row["trace"],
                    "mpc_decision": row["mpc_decision"],
                }
                for row in residual["rows"]
            ],
            "metrics": residual["metrics"],
        }
        exact = comparable_residual == baseline
        results.append(
            {
                "fixture": name,
                "native_steps": len(baseline["rows"]),
                "exact": exact,
                "baseline_sha256": _canonical_hash(baseline),
                "residual_sha256": _canonical_hash(comparable_residual),
            }
        )
    return {"fixtures": results, "passed": all(row["exact"] for row in results)}


def influence_and_timing_fixture() -> dict[str, Any]:
    track = Track(np.array([0.0, 1000.0]), np.array([100.0, 100.0]) / 3.6)
    environment = _environment(track)
    controller = FeedbackResidualController("CHUNK3_RESIDUAL_SAC")
    requested = []
    influenced = []
    actor_boundaries = []
    try:
        observation, _ = environment.reset(seed=7)
        controller.reset()
        for _ in range(500):
            actor_observation = controller.prepare_native_step(observation)
            if actor_observation is not None:
                actor_boundaries.append(controller.native_step)
                controller.begin_chunk(np.array([-0.2, -0.4, -0.6]))
            action = controller.action(observation)
            decision = controller.last_decision
            requested.append((decision.chunk_phase, decision.requested_residual))
            if decision.residual_influence > 1.0e-15:
                influenced.append(
                    (
                        decision.chunk_phase,
                        decision.requested_residual,
                        action - decision.mpc_executed_action,
                    )
                )
            observation, _, terminated, truncated, _ = environment.step([action])
            controller.observe_executed_action(action)
            if terminated or truncated:
                break
    finally:
        environment.close()
    checks = {
        "actor_boundaries_exactly_30_native_steps": actor_boundaries
        == list(range(0, len(requested), 30)),
        "all_three_phases_requested": {row[0] for row in requested} == {0, 1, 2},
        "all_three_phases_influenced": {row[0] for row in influenced} == {0, 1, 2},
        "distinct_scaled_influence": {round(abs(row[2]), 12) for row in influenced}
        == {0.05, 0.1, 0.15},
        "residual_scale_frozen": RESIDUAL_SCALE == 0.25,
    }
    return {
        "native_steps": len(requested),
        "actor_boundaries": actor_boundaries,
        "influenced_native_steps": len(influenced),
        "checks": checks,
        "passed": all(checks.values()),
    }


def physical_safety_fixture() -> dict[str, Any]:
    rows = []
    for name in PARITY_FIXTURES:
        result = _residual_rollout(
            physical_fixtures()[name], np.array([-0.2, -0.4, -0.6])
        )
        metrics = result["metrics"]
        rows.append(
            {
                "fixture": name,
                "completed": metrics["completed"],
                "travel_time_s": metrics["travel_time_s"],
                "max_speed_violation_m_s": metrics["max_speed_violation_m_s"],
                "feasible": result["feasible"],
                "completion_bound_retained_every_step": all(
                    row["residual_decision"]["completion_bound_s"] is not None
                    for row in result["rows"]
                ),
            }
        )
    return {
        "rows": rows,
        "passed": all(
            row["feasible"]
            and row["max_speed_violation_m_s"] <= 0.0
            and row["completion_bound_retained_every_step"]
            for row in rows
        ),
    }


def interruption_fixtures() -> dict[str, Any]:
    km_h = 1.0 / 3.6
    visibility_track = Track(np.array([0.0, 516.0]), np.array([100.0, 60.0]) * km_h)
    visibility = _residual_rollout(visibility_track, np.zeros(3))
    visible_events = [
        row["residual_decision"]
        for row in visibility["rows"]
        if row["residual_decision"]["newly_visible_transition"]
        and row["residual_decision"]["chunk_phase"] > 0
    ]

    deadline_track = Track(np.array([0.0, 1000.0]), np.array([100.0, 100.0]) * km_h)
    environment = _environment(deadline_track)
    controller = FeedbackResidualController("CHUNK3_RESIDUAL_SAC")
    deadline_rows = []
    try:
        observation, _ = environment.reset(seed=3)
        controller.reset()
        for _ in range(30):
            actor_observation = controller.prepare_native_step(observation)
            if actor_observation is not None:
                controller.begin_chunk(np.array([0.0, -1.0, -1.0]))
            action = controller.action(observation)
            deadline_rows.append(controller.last_decision.as_dict())
            observation, _, terminated, truncated, _ = environment.step([action])
            controller.observe_executed_action(action)
            if terminated or truncated:
                break
    finally:
        environment.close()
    rejection = deadline_rows[10]
    remainder = deadline_rows[11:30]
    checks = {
        "new_visibility_mid_chunk_observed": bool(visible_events),
        "new_visibility_cancelled": bool(
            visible_events
            and all(
                row["chunk_cancelled"]
                and row["cancellation_reason"] == "NEWLY_VISIBLE_SPEED_LIMIT_TRANSITION"
                for row in visible_events
            )
        ),
        "phase_two_residual_rejected": bool(
            rejection["chunk_phase"] == 1
            and not rejection["residual_accepted"]
            and rejection["cancellation_reason"] == "RESIDUAL_DEADLINE_RECOVERABILITY"
        ),
        "no_mid_chunk_regeneration": all(
            not row["residual_attempted"]
            and row["executed_action"] == row["mpc_executed_action"]
            for row in remainder
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def smdp_fixture() -> dict[str, Any]:
    rows = []
    for action, duration in (
        (np.array([0.5]), 10),
        (np.array([-0.5, 0.0, 0.5]), 30),
    ):
        accumulator = MacroAccumulator(np.zeros(16), action, duration)
        native = np.linspace(-0.5, 0.75, duration)
        for value in native:
            accumulator.add_native_step(
                reward=float(value),
                speed_cost=max(0.0, float(value)),
                terminal_cost=0.0,
                intervention_cost=abs(float(value)) * 0.1,
            )
        transition = accumulator.finish(np.ones(16))
        independent = sum(
            (0.99 ** (index / 10.0)) * value for index, value in enumerate(native)
        )
        rows.append(
            {
                "duration": duration,
                "reward": transition.reward,
                "independent_reward": independent,
                "continuation_discount": transition.continuation_discount,
                "expected_discount": 0.99 ** (duration / 10.0),
                "passed": math.isclose(transition.reward, independent, abs_tol=1.0e-14)
                and math.isclose(
                    transition.continuation_discount,
                    0.99 ** (duration / 10.0),
                    abs_tol=1.0e-14,
                ),
            }
        )
    return {
        "gamma_native": GAMMA_NATIVE,
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }


def stability_fixture() -> dict[str, Any]:
    rows = []
    rng = np.random.default_rng(20261023)
    for method, dim in (("RESIDUAL1S_SAC", 1), ("CHUNK3_RESIDUAL_SAC", 3)):
        agent = ConstrainedResidualSAC(method, 11)
        batch = {
            "observation": rng.normal(size=(256, 16)).astype(np.float32),
            "action": rng.uniform(-1, 1, size=(256, dim)).astype(np.float32),
            "reward": rng.normal(size=(256, 1)).astype(np.float32),
            "speed_cost": rng.uniform(0, 0.01, size=(256, 1)).astype(np.float32),
            "terminal_cost": np.zeros((256, 1), dtype=np.float32),
            "intervention_cost": rng.uniform(0, 0.25, size=(256, 1)).astype(np.float32),
            "next_observation": rng.normal(size=(256, 16)).astype(np.float32),
            "continuation_discount": np.full((256, 1), 0.99**dim, dtype=np.float32),
        }
        diagnostics = []
        generator = torch.Generator().manual_seed(77)
        for _ in range(20):
            diagnostics.append(agent.update(batch, generator=generator))
        action = agent.act(np.zeros(16), deterministic=True)
        finite = np.isfinite(action).all() and all(
            np.isfinite(value) for row in diagnostics for value in row.values()
        )
        rows.append(
            {
                "method": method,
                "action_dim": dim,
                "target_entropy": agent.config.target_entropy,
                "updates": len(diagnostics),
                "finite": bool(finite),
                "final": diagnostics[-1],
            }
        )
    return {"rows": rows, "passed": all(row["finite"] for row in rows)}


def _synthetic_runner(
    method: str,
    *,
    attempt_id: str,
    native_budget: int,
    replay_capacity: int,
) -> TrainingRunner:
    environment = gym.wrappers.TimeLimit(LongiControlEnv(), max_episode_steps=1_800)
    return TrainingRunner(
        method=method,
        training_seed=11,
        attempt_id=attempt_id,
        configuration_sha256="synthetic-preflight",
        source_hashes={"synthetic": "preflight"},
        environment=environment,
        native_budget=native_budget,
        update_budget=0,
        warmup_native_transitions=native_budget - 1,
        batch_size=2,
        replay_capacity=replay_capacity,
        verify_track_content=False,
    )


def recovery_fixture(temporary: Path) -> dict[str, Any]:
    rows = []
    for checkpoint_step in (12, 22):
        runner = _synthetic_runner(
            "CHUNK3_RESIDUAL_SAC",
            attempt_id=f"recovery-{checkpoint_step}",
            native_budget=80,
            replay_capacity=128,
        )
        restored = None
        try:
            while runner.native_transition_count < checkpoint_step:
                runner.advance_native()
            path = temporary / f"recovery-{checkpoint_step}.pkl"
            metadata = runner.save(path)
            expected = [runner.advance_native() for _ in range(14)]
            restored = TrainingRunner.restore(
                path,
                expected_sha256=metadata["sha256"],
                configuration_sha256="synthetic-preflight",
                source_hashes={"synthetic": "preflight"},
                attempt_id=f"recovery-{checkpoint_step}",
            )
            observed = [restored.advance_native() for _ in range(14)]
            rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "chunk_phase": (checkpoint_step % 30) // 10,
                    "checkpoint_sha256": metadata["sha256"],
                    "exact_future_records": observed == expected,
                }
            )
        finally:
            runner.close()
            if restored is not None:
                restored.close()
    return {
        "rows": rows,
        "passed": all(row["exact_future_records"] for row in rows),
    }


def resource_fixture(temporary: Path) -> dict[str, Any]:
    measurements = []
    evidence_rows = []
    checkpoint = None
    for method in ("CHUNK3_RESIDUAL_SAC", "RESIDUAL1S_SAC"):
        runner = _synthetic_runner(
            method,
            attempt_id=f"resource-{method}",
            native_budget=1_000,
            replay_capacity=300_000,
        )
        wall_started = time.perf_counter()
        process_started = time.process_time()
        try:
            for _ in range(1_000):
                evidence_rows.append(runner.advance_native())
            wall = time.perf_counter() - wall_started
            process = time.process_time() - process_started
            row = {
                "method": method,
                "native_transitions": 1_000,
                "wall_time_s": wall,
                "process_time_s": process,
                "wall_time_per_native_s": wall / 1_000,
                "process_time_per_native_s": process / 1_000,
            }
            if method == "CHUNK3_RESIDUAL_SAC":
                checkpoint_path = temporary / "resource-checkpoint.pkl"
                checkpoint = runner.save(checkpoint_path)
                restore_started = time.perf_counter()
                restored = TrainingRunner.restore(
                    checkpoint_path,
                    expected_sha256=checkpoint["sha256"],
                    configuration_sha256="synthetic-preflight",
                    source_hashes={"synthetic": "preflight"},
                    attempt_id=f"resource-{method}",
                )
                checkpoint["restore_s"] = time.perf_counter() - restore_started
                restored.close()
            measurements.append(row)
        finally:
            runner.close()

    evidence_path = temporary / "resource-evidence.jsonl.gz"
    with gzip.open(evidence_path, "wt", encoding="utf-8", compresslevel=6) as stream:
        for row in evidence_rows:
            stream.write(
                json.dumps(
                    row,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            stream.write("\n")
    evidence_bytes_per_row = evidence_path.stat().st_size / len(evidence_rows)

    agent = ConstrainedResidualSAC("CHUNK3_RESIDUAL_SAC", 11)
    generator = torch.Generator().manual_seed(88)
    rng = np.random.default_rng(88)
    batch = {
        "observation": rng.normal(size=(256, 16)).astype(np.float32),
        "action": rng.uniform(-1, 1, size=(256, 3)).astype(np.float32),
        "reward": rng.normal(size=(256, 1)).astype(np.float32),
        "speed_cost": np.zeros((256, 1), dtype=np.float32),
        "terminal_cost": np.zeros((256, 1), dtype=np.float32),
        "intervention_cost": rng.uniform(0, 0.25, size=(256, 1)).astype(np.float32),
        "next_observation": rng.normal(size=(256, 16)).astype(np.float32),
        "continuation_discount": np.full((256, 1), 0.99**3, dtype=np.float32),
    }
    update_wall_started = time.perf_counter()
    update_process_started = time.process_time()
    for _ in range(50):
        agent.update(batch, generator=generator)
    update_wall = time.perf_counter() - update_wall_started
    update_process = time.process_time() - update_process_started

    inference_started = time.perf_counter()
    for _ in range(1_000):
        agent.act(np.zeros(16), deterministic=True)
    inference_time = time.perf_counter() - inference_started

    mean_native_wall = statistics.fmean(
        row["wall_time_per_native_s"] for row in measurements
    )
    mean_native_process = statistics.fmean(
        row["process_time_per_native_s"] for row in measurements
    )
    mean_update_wall = update_wall / 50
    mean_update_process = update_process / 50
    training_collection_s = mean_native_wall * 1_800_000
    training_updates_s = mean_update_wall * 180_000
    diagnostic_native_steps = 900 * 1_200
    diagnostic_s = mean_native_wall * diagnostic_native_steps
    checkpoint_count = 180
    checkpoint_s = float(checkpoint["write_s"]) * checkpoint_count
    raw_projection = (
        training_collection_s + training_updates_s + diagnostic_s + checkpoint_s
    )
    projected_active_s = raw_projection * 2.0
    scientific_checkpoint_bytes = int(checkpoint["size_bytes"]) * 36
    recovery_checkpoint_bytes = int(checkpoint["size_bytes"]) * 6
    evidence_bytes = evidence_bytes_per_row * (1_800_000 + diagnostic_native_steps)
    projected_disk_bytes = (
        scientific_checkpoint_bytes
        + recovery_checkpoint_bytes
        + evidence_bytes
        + 1_000_000_000
    )
    return {
        "measurements": measurements,
        "gradient_updates": {
            "count": 50,
            "wall_time_s": update_wall,
            "process_time_s": update_process,
            "mean_wall_time_s": mean_update_wall,
            "mean_process_time_s": mean_update_process,
        },
        "neural_inference": {
            "count": 1_000,
            "wall_time_s": inference_time,
            "mean_wall_time_s": inference_time / 1_000,
        },
        "checkpoint": checkpoint,
        "native_evidence": {
            "sample_rows": len(evidence_rows),
            "sample_sha256": file_sha256(evidence_path),
            "compressed_bytes_per_row": evidence_bytes_per_row,
        },
        "host": {
            "maximum_rss_platform_units": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "mean_native_process_time_s": mean_native_process,
            "mean_update_process_time_s": mean_update_process,
        },
        "projection": {
            "training_collection_s": training_collection_s,
            "training_updates_s": training_updates_s,
            "diagnostic_s": diagnostic_s,
            "checkpoint_s": checkpoint_s,
            "raw_total_s": raw_projection,
            "uncertainty_factor": 2.0,
            "projected_active_compute_hours": projected_active_s / 3600.0,
            "projected_retained_bytes": projected_disk_bytes,
            "projected_retained_gb": projected_disk_bytes / 1_000_000_000,
        },
        "limits": {"active_compute_hours": 24.0, "retained_artifacts_gb": 12.0},
        "runtime_gate_passed": projected_active_s <= 24.0 * 3600.0,
        "disk_gate_passed": projected_disk_bytes <= 12_000_000_000,
    }


def render_resource_report(resource_result: dict[str, Any]) -> str:
    projection = resource_result["projection"]
    active_hours = projection["projected_active_compute_hours"]
    retained_gb = projection["projected_retained_gb"]
    runtime_gate = "PASS" if resource_result["runtime_gate_passed"] else "FAIL"
    disk_gate = "PASS" if resource_result["disk_gate_passed"] else "FAIL"
    return "\n".join(
        [
            "# Resource Report",
            "",
            "Measured on synthetic unreserved fixtures before main training.",
            "Wall time and process time are recorded separately.",
            "",
            f"- Projected active compute: {active_hours:.3f} h.",
            f"- Limit: {resource_result['limits']['active_compute_hours']:.1f} h.",
            f"- Projected retained artifacts: {retained_gb:.3f} GB.",
            f"- Limit: {resource_result['limits']['retained_artifacts_gb']:.1f} GB.",
            f"- Runtime gate: {runtime_gate}.",
            f"- Disk gate: {disk_gate}.",
            "",
            "The projection includes 1.8M native training transitions, 180k",
            "updates, 900 Diagnostic episodes, all technical/scientific checkpoint",
            "writes, compressed native evidence, and a 2x runtime uncertainty factor.",
            "Host sleep is excluded from active process work and must be reported",
            "separately from later wall-clock execution.",
            "",
        ]
    )


def run_preflight() -> dict[str, Any]:
    verify_repository(require_clean=True)
    split = json.loads(SPLIT_LEDGER.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="feedback-chunk-preflight-") as name:
        temporary = Path(name)
        parity = zero_residual_parity()
        influence = influence_and_timing_fixture()
        safety = physical_safety_fixture()
        interruption = interruption_fixtures()
        smdp = smdp_fixture()
        stability = stability_fixture()
        recovery = recovery_fixture(temporary)
        resources = resource_fixture(temporary)
    split_checks = {
        "route_clean_verified": split["route_clean_verified"],
        "diagnostic_count_36": split["checks"]["diagnostic_count_36"],
        "validation_count_36": split["checks"]["validation_count_36"],
        "validation_performance_episodes_zero": split["validation_panel"][
            "performance_episode_count"
        ]
        == 0,
        "paper_uninspected": split["paper"]["state"] == "SEALED_UNINSPECTED"
        and split["paper"]["content_fingerprints_constructed"] == 0,
    }
    gates = {
        "A_zero_residual_parity": parity["passed"],
        "B_action_chunk_timing": influence["checks"][
            "actor_boundaries_exactly_30_native_steps"
        ],
        "C_distinct_phase_influence": influence["passed"],
        "D_physical_shield_unchanged": parity["passed"] and interruption["passed"],
        "E_exact_native_speed_compliance": safety["passed"],
        "F_new_speed_limit_response": interruption["checks"][
            "new_visibility_cancelled"
        ],
        "G_mid_chunk_intervention": interruption["checks"][
            "phase_two_residual_rejected"
        ]
        and interruption["checks"]["no_mid_chunk_regeneration"],
        "H_partial_chunk_recovery": recovery["passed"],
        "I_smdp_targets": smdp["passed"],
        "J_route_clean_split": all(split_checks.values()),
        "K_numerical_stability": stability["passed"],
        "L_resource_projection": resources["runtime_gate_passed"]
        and resources["disk_gate_passed"],
    }
    passed = all(gates.values())
    results = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "scope": (
            "synthetic/unreserved fixtures only; zero Diagnostic, Validation "
            "or Paper episodes"
        ),
        "gates": gates,
        "passed": passed,
        "zero_residual_parity": parity,
        "influence_and_timing": influence,
        "physical_safety": safety,
        "interruption": interruption,
        "smdp": smdp,
        "stability": stability,
        "recovery": recovery,
        "split_checks": split_checks,
        "scientific_activity": {
            "diagnostic_performance_episodes": 0,
            "validation_performance_episodes": 0,
            "paper_performance_episodes": 0,
            "paper_content_fingerprints_constructed": 0,
        },
    }
    atomic_json(PREFLIGHT_RESULTS, results)
    atomic_json(RESOURCE_MEASUREMENTS, resources)
    RESOURCE_REPORT.write_text(render_resource_report(resources), encoding="utf-8")
    if not passed:
        raise RuntimeError("Feedback Chunk RL preflight failed closed")
    status = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "ready_for_main_training": True,
        "preflight_passed": True,
        "resource_gate_passed": True,
        "canonical_sha256": sha256_file(CANONICAL),
        "split_ledger_sha256": sha256_file(SPLIT_LEDGER),
        "scientific_source_sha256": scientific_source_hashes(),
        "preflight_results_sha256": sha256_file(PREFLIGHT_RESULTS),
        "resource_measurements_sha256": sha256_file(RESOURCE_MEASUREMENTS),
        "authorization": {
            "main_training_authorized": True,
            "main_training_enabled": False,
            "diagnostic_authorized": False,
            "validation_authorized": False,
            "paper_authorized": False,
        },
    }
    atomic_json(PREPARATION_STATUS, status)
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    authorization.update(
        {
            "preflight_passed": True,
            "resource_gate_passed": True,
            "method_source_frozen": True,
            "main_training_enabled": False,
            "diagnostic_authorized": False,
        }
    )
    atomic_json(AUTHORIZATION, authorization)
    return results


if __name__ == "__main__":
    output = run_preflight()
    print(json.dumps({"passed": output["passed"], "gates": output["gates"]}, indent=2))
