"""Preparation-only exact-resume and information-boundary probes."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from benchmarks.hybrid_rl_design.controller import HybridTargetSpeedController
from benchmarks.physical_feasibility_reference.diagnostics import physical_fixtures
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from .checkpointing import file_sha256
from .runner import HybridRunner


class _FixedTrackGenerator:
    def __init__(self, track) -> None:
        self.track = track

    def __call__(self, _rng):
        return self.track


def _fixture_environment() -> gym.Env:
    track = physical_fixtures()["upward_changes"]
    return gym.wrappers.TimeLimit(
        LongiControlEnv(track_generator=_FixedTrackGenerator(track)),
        max_episode_steps=1_800,
    )


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _scientific_snapshot(runner: HybridRunner) -> dict[str, Any]:
    runtime = runner.runtime_state()
    runtime.pop("walltime_state")
    return {
        "runtime": runtime,
        "agent": runner.agent.state_dict(),
        "replay": runner.replay.state_dict(),
        "environment_state": runner.environment.unwrapped.state,
        "controller_state": runner.controller.state,
        "known_transitions": dict(runner.controller.reference._known_transitions),
        "controller_last_action": runner.controller.reference._last_action,
    }


def exact_resume_probe() -> dict[str, Any]:
    configuration_sha256 = "exact-resume-probe"
    source_hashes = {"probe": "unreserved-upward-changes"}
    runner = HybridRunner(
        training_seed=11,
        attempt_id="EXACT-RESUME-PROBE",
        configuration_sha256=configuration_sha256,
        source_hashes=source_hashes,
        environment=_fixture_environment(),
        native_budget=40,
        update_budget=4,
        warmup_native_transitions=10,
        batch_size=1,
        replay_capacity=64,
        verify_track_content=False,
    )
    for _ in range(3):
        runner.advance_native()
    partial = {
        "native_steps_in_accumulator": runner.macro_accumulator.native_steps,
        "remaining_native_steps_in_block": runner.remaining_native_steps_in_block,
        "held_strategy_action": runner.held_strategy_action,
    }
    with tempfile.TemporaryDirectory(prefix="hybrid-rl-resume-") as directory:
        checkpoint = Path(directory) / "checkpoint.pkl"
        metadata = runner.save(checkpoint)
        uninterrupted_records = [runner.advance_native() for _ in range(20)]
        uninterrupted = _scientific_snapshot(runner)
        restored = HybridRunner.restore(
            checkpoint,
            expected_sha256=file_sha256(checkpoint),
            configuration_sha256=configuration_sha256,
            source_hashes=source_hashes,
            attempt_id=runner.attempt_id,
        )
        restored_records = [restored.advance_native() for _ in range(20)]
        resumed = _scientific_snapshot(restored)
        records_exact = _equal(uninterrupted_records, restored_records)
        scientific_state_exact = _equal(uninterrupted, resumed)
        restored.close()
    runner.close()
    metadata.pop("path", None)
    return {
        "checkpoint": metadata,
        "partial_block_at_checkpoint": partial,
        "future_native_steps_compared": 20,
        "future_action_reward_cost_trace_exact": records_exact,
        "network_optimizer_replay_environment_controller_rng_exact": (
            scientific_state_exact
        ),
        "passed": bool(records_exact and scientific_state_exact),
        "preparation_interactions": {
            "real_native_transitions": 43,
            "gradient_updates": 4,
        },
    }


def causal_information_boundary() -> dict[str, Any]:
    source = inspect.getsource(HybridTargetSpeedController)
    action_parameters = tuple(
        inspect.signature(HybridTargetSpeedController.action).parameters
    )
    prepare_parameters = tuple(
        inspect.signature(HybridTargetSpeedController.prepare_strategy_step).parameters
    )
    forbidden_parameters = {
        "environment",
        "track",
        "simulator_state",
        "oracle_action",
        "future_state",
    }
    observed_parameters = set(action_parameters) | set(prepare_parameters)
    checks = {
        "action_signature": action_parameters
        == (
            "self",
            "observation",
            "strategy_action",
        ),
        "prepare_signature": prepare_parameters == ("self", "observation"),
        "no_forbidden_runtime_parameters": not (
            forbidden_parameters & observed_parameters
        ),
        "ordinary_observation_shape_enforced": "shape != (8,)" in source,
        "strategy_observation_shape_enforced": (
            "STRATEGY_OBSERVATION_DIMENSIONS" in source
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "controller_inputs": ["ordinary 8D observation", "scalar strategy action"],
        "hidden_route_input": False,
        "oracle_input": False,
    }
