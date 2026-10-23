"""Exact-budget collection and learning for both residual treatments."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .chunk_decoder import method_spec
from .macro_transition import MacroAccumulator
from .replay import MacroReplay
from .residual_controller import FeedbackResidualController
from .sac_agent import ConstrainedResidualSAC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STREAM_LEDGER = (
    ROOT / "benchmarks/constrained_v2_collector_replication/stream_ledger.json"
)
TRAINING_SEEDS = (11, 29, 47)
METHODS = ("CHUNK3_RESIDUAL_SAC", "RESIDUAL1S_SAC")
NATIVE_BUDGET = 300_000
GRADIENT_UPDATE_BUDGET = 30_000
WARMUP_NATIVE_TRANSITIONS = 10_000
BATCH_SIZE = 256
REPLAY_CAPACITY = 300_000
TIME_LIMIT_NATIVE_STEPS = 1_800
CANONICAL_TASK = TaskSpecification(140.0, 0.0)
WARMUP_LATENT_STANDARD_DEVIATION = 0.2

POLICY_TORCH_DOMAIN = 0x504F4C59
UPDATE_TORCH_DOMAIN = 0x55504454
WARMUP_NUMPY_DOMAIN = 0x5741524D
REPLAY_NUMPY_DOMAIN = 0x52504C59


def cumulative_update_target(native_transitions: int) -> int:
    if native_transitions <= WARMUP_NATIVE_TRANSITIONS:
        return 0
    return min(
        GRADIENT_UPDATE_BUDGET,
        ((int(native_transitions) - WARMUP_NATIVE_TRANSITIONS) * 30_000) // 290_000,
    )


def _seed_value(seed: int, domain: int) -> int:
    return int(np.random.SeedSequence([seed, domain]).generate_state(1)[0])


def _torch_generator(seed: int, domain: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(_seed_value(seed, domain))


def _numpy_generator(seed: int, domain: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, domain]))


class CanonicalTrainingStream:
    EXPECTED_HASHES = {
        11: "08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c",
        29: "ca8912845c1293e86fd326a7415901c9beb0c70ae6a32db5e78263fceb355312",
        47: "2b1323d664f8ca470975ce4238f33a1883bf6d88eb43d035681b5c1520b15255",
    }

    def __init__(self, seed: int, cursor: int = 0) -> None:
        if seed not in TRAINING_SEEDS:
            raise ValueError("training seed is outside the frozen matrix")
        payload = json.loads(STREAM_LEDGER.read_text(encoding="utf-8"))[str(seed)]
        if payload["ordered_sha256"] != self.EXPECTED_HASHES[seed]:
            raise RuntimeError("frozen training stream hash changed")
        if len(payload["accepted_tasks"]) != 865:
            raise RuntimeError("frozen training stream length changed")
        self.seed = int(seed)
        self.rows = tuple(payload["accepted_tasks"])
        self.cursor = int(cursor)
        if not 0 <= self.cursor <= len(self.rows):
            raise ValueError("invalid training stream cursor")

    def next_row(self) -> dict[str, Any]:
        if self.cursor >= len(self.rows):
            raise RuntimeError("frozen training stream exhausted")
        row = dict(self.rows[self.cursor])
        self.cursor += 1
        return row

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "cursor": self.cursor}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> CanonicalTrainingStream:
        return cls(int(state["seed"]), int(state["cursor"]))


class UpdateBudgetAccumulator:
    def __init__(
        self,
        *,
        warmup: int = WARMUP_NATIVE_TRANSITIONS,
        native_budget: int = NATIVE_BUDGET,
        update_budget: int = GRADIENT_UPDATE_BUDGET,
    ) -> None:
        if native_budget <= 0 or update_budget < 0 or not 0 <= warmup < native_budget:
            raise ValueError("invalid update budget")
        self.warmup = int(warmup)
        self.native_budget = int(native_budget)
        self.update_budget = int(update_budget)

    def cumulative_target(self, native_transitions: int) -> int:
        if native_transitions <= self.warmup or self.update_budget == 0:
            return 0
        target = ((int(native_transitions) - self.warmup) * self.update_budget) // (
            self.native_budget - self.warmup
        )
        return min(target, self.update_budget)

    def due(self, native_transitions: int, completed_updates: int) -> int:
        return max(0, self.cumulative_target(native_transitions) - completed_updates)


def make_training_environment() -> gym.Env:
    return gym.wrappers.TimeLimit(
        StochasticTrack(), max_episode_steps=TIME_LIMIT_NATIVE_STEPS
    )


class TrainingRunner:
    """Advance exactly one real native environment transition per call."""

    def __init__(
        self,
        *,
        method: str,
        training_seed: int,
        attempt_id: str | None,
        configuration_sha256: str,
        source_hashes: Mapping[str, str],
        environment: gym.Env | None = None,
        native_budget: int = NATIVE_BUDGET,
        update_budget: int = GRADIENT_UPDATE_BUDGET,
        warmup_native_transitions: int = WARMUP_NATIVE_TRANSITIONS,
        batch_size: int = BATCH_SIZE,
        replay_capacity: int = REPLAY_CAPACITY,
        verify_track_content: bool = True,
    ) -> None:
        self.spec = method_spec(method)
        if training_seed not in TRAINING_SEEDS:
            raise ValueError("training seed is outside the frozen matrix")
        self.method = self.spec.name
        self.training_seed = int(training_seed)
        self.attempt_id = attempt_id or str(uuid.uuid4())
        self.configuration_sha256 = str(configuration_sha256)
        self.source_hashes = dict(source_hashes)
        self.native_budget = int(native_budget)
        self.update_budget = int(update_budget)
        self.warmup_native_transitions = int(warmup_native_transitions)
        self.batch_size = int(batch_size)
        self.verify_track_content = bool(verify_track_content)
        self.update_budget_accumulator = UpdateBudgetAccumulator(
            warmup=self.warmup_native_transitions,
            native_budget=self.native_budget,
            update_budget=self.update_budget,
        )
        self.agent = ConstrainedResidualSAC(self.method, self.training_seed)
        self.replay = MacroReplay(replay_capacity, action_dim=self.spec.action_dim)
        self.task_stream = CanonicalTrainingStream(self.training_seed)
        self.environment = environment or make_training_environment()
        self.controller = FeedbackResidualController(self.method)
        self.policy_generator = _torch_generator(
            self.training_seed, POLICY_TORCH_DOMAIN
        )
        self.update_generator = _torch_generator(
            self.training_seed, UPDATE_TORCH_DOMAIN
        )
        self.warmup_rng = _numpy_generator(self.training_seed, WARMUP_NUMPY_DOMAIN)
        self.replay_rng = _numpy_generator(self.training_seed, REPLAY_NUMPY_DOMAIN)
        self.native_transition_count = 0
        self.macro_transition_count = 0
        self.gradient_update_count = 0
        self.actor_sample_count = 0
        self.episode_id = -1
        self.episode_native_step = 0
        self.pending_reset = True
        self.current_task: dict[str, Any] | None = None
        self.observation = np.zeros(8, dtype=np.float64)
        self.prepared_actor_observation: np.ndarray | None = None
        self.macro_accumulator: MacroAccumulator | None = None
        self.training_outcomes: list[dict[str, Any]] = []
        self.chunk_outcomes: list[dict[str, Any]] = []
        self.update_diagnostics: list[dict[str, float]] = []
        self.cancellation_counts: dict[str, int] = {}
        self.proposal_count = 0
        self.residual_attempt_count = 0
        self.residual_accept_count = 0
        self.influenced_native_count = 0
        self.last_native_record: dict[str, Any] | None = None
        self.completed_walltime_s = 0.0
        self.completed_process_time_s = 0.0
        self.segment_wall_started = time.perf_counter()
        self.segment_process_started = time.process_time()
        self._reset_to_next_task()

    def _reset_to_next_task(self) -> None:
        if not self.pending_reset:
            raise RuntimeError("episode reset is not pending")
        row = self.task_stream.next_row()
        observation, _ = self.environment.reset(seed=int(row["reset_seed"]))
        if self.verify_track_content:
            track = self.environment.unwrapped.track
            fingerprint = fingerprint_track(
                track, v2_semantics(), numeric_seed=int(row["reset_seed"])
            )
            if fingerprint.route_sha256 != row["route_sha256"]:
                raise RuntimeError("runtime route differs from frozen stream")
            if fingerprint.full_task_sha256 != row["full_task_sha256"]:
                raise RuntimeError("runtime task differs from frozen stream")
        self.controller.reset()
        self.episode_id += 1
        self.episode_native_step = 0
        self.current_task = row
        self.observation = np.asarray(observation, dtype=np.float64)
        self.prepared_actor_observation = None
        self.macro_accumulator = None
        self.pending_reset = False

    def _warmup_action(self) -> np.ndarray:
        latent = self.warmup_rng.normal(
            0.0, WARMUP_LATENT_STANDARD_DEVIATION, size=self.spec.action_dim
        )
        return np.tanh(latent).astype(np.float64)

    def _start_macro_block(self, strategy_observation: np.ndarray) -> None:
        if self.macro_accumulator is not None:
            raise RuntimeError("actor action changed inside a macro block")
        if self.native_transition_count < self.warmup_native_transitions:
            actor_action = self._warmup_action()
        else:
            actor_action = self.agent.act(
                strategy_observation,
                deterministic=False,
                generator=self.policy_generator,
            )
        self.actor_sample_count += 1
        self.controller.begin_chunk(actor_action)
        self.macro_accumulator = MacroAccumulator(
            strategy_observation,
            actor_action,
            self.spec.actor_interval_native_steps,
        )

    def _record_episode(self, info: Mapping[str, Any]) -> None:
        metrics = EpisodeMetrics(**info["episode_metrics"])
        feasible = is_feasible(metrics, CANONICAL_TASK)
        self.training_outcomes.append(
            {
                "episode_id": self.episode_id,
                "ending_native_transition": self.native_transition_count,
                "task": dict(self.current_task or {}),
                "feasible": feasible,
                "deadline_compliant": metrics.travel_time_s <= 140.0,
                "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
                "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
                **asdict(metrics),
            }
        )

    def _perform_due_updates(self) -> None:
        due = self.update_budget_accumulator.due(
            self.native_transition_count, self.gradient_update_count
        )
        if due and len(self.replay) < self.batch_size:
            raise RuntimeError("warm-up did not provide one SAC minibatch")
        for _ in range(due):
            batch = self.replay.sample(self.batch_size, self.replay_rng)
            diagnostics = self.agent.update(batch, generator=self.update_generator)
            self.gradient_update_count += 1
            if (
                self.gradient_update_count == 1
                or self.gradient_update_count % 100 == 0
                or self.gradient_update_count == self.update_budget
            ):
                self.update_diagnostics.append(
                    {
                        "native_transition_count": float(self.native_transition_count),
                        "gradient_update_count": float(self.gradient_update_count),
                        **diagnostics,
                    }
                )

    def _prepare_actor_boundary(self) -> np.ndarray:
        if self.prepared_actor_observation is not None:
            result = self.prepared_actor_observation
            self.prepared_actor_observation = None
            return result
        result = self.controller.prepare_native_step(self.observation)
        if result is None:
            raise RuntimeError("new macro block is not at an actor boundary")
        return result

    def advance_native(self) -> dict[str, Any]:
        if self.native_transition_count >= self.native_budget:
            raise RuntimeError("frozen native transition budget is exhausted")
        if self.pending_reset:
            self._reset_to_next_task()
        if self.macro_accumulator is None:
            self._start_macro_block(self._prepare_actor_boundary())
        else:
            actor_observation = self.controller.prepare_native_step(self.observation)
            if actor_observation is not None:
                raise RuntimeError("actor was offered a mid-block resample")
        assert self.macro_accumulator is not None
        pre_state = self.controller.state
        action = self.controller.action(self.observation)
        residual_decision = self.controller.last_decision
        if residual_decision is None:
            raise RuntimeError("residual safety decision was not recorded")
        next_observation, _, terminated, environment_truncated, info = (
            self.environment.step([action])
        )
        self.controller.observe_executed_action(action)
        self.native_transition_count += 1
        self.episode_native_step += 1
        dataset_truncated = bool(
            self.native_transition_count == self.native_budget
            and not terminated
            and not environment_truncated
        )
        task_ended = bool(terminated or environment_truncated)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        terminal_failure = bool(
            task_ended and (not metrics.completed or metrics.travel_time_s > 140.0)
        )
        speed_cost = max(0.0, float(info["velocity_m_s"] - info["speed_limit_m_s"]))
        self.macro_accumulator.add_native_step(
            reward=-1000.0 * float(info["step_energy_kwh"]),
            speed_cost=speed_cost,
            terminal_cost=float(terminal_failure),
            intervention_cost=residual_decision.intervention_deviation,
            terminated=bool(terminated),
            environment_truncated=bool(environment_truncated),
            dataset_truncated=dataset_truncated,
        )
        self.proposal_count += 1
        self.residual_attempt_count += int(residual_decision.residual_attempted)
        self.residual_accept_count += int(residual_decision.residual_accepted)
        self.influenced_native_count += int(
            residual_decision.residual_influence > 1.0e-15
        )
        if residual_decision.cancellation_reason is not None:
            reason = residual_decision.cancellation_reason
            self.cancellation_counts[reason] = (
                self.cancellation_counts.get(reason, 0) + 1
            )
        self.observation = np.asarray(next_observation, dtype=np.float64)
        record = {
            "native_transition_count": self.native_transition_count,
            "method": self.method,
            "training_seed": self.training_seed,
            "episode_id": self.episode_id,
            "episode_native_step": self.episode_native_step,
            "task_reset_seed": int((self.current_task or {})["reset_seed"]),
            "pre_state": asdict(pre_state),
            "actor_action": self.macro_accumulator.action.tolist(),
            "native_action": float(action),
            "step_energy_kwh": float(info["step_energy_kwh"]),
            "post_state": {
                "position_m": float(info["position_m"]),
                "velocity_m_s": float(info["velocity_m_s"]),
                "acceleration_m_s2": float(info["acceleration_m_s2"]),
                "elapsed_time_s": float(info["elapsed_time_s"]),
                "total_energy_kwh": float(info["total_energy_kwh"]),
                "speed_limit_m_s": float(info["speed_limit_m_s"]),
            },
            "speed_cost_m_s": speed_cost,
            "terminal_cost": float(terminal_failure),
            "intervention_deviation": residual_decision.intervention_deviation,
            "terminated": bool(terminated),
            "environment_truncated": bool(environment_truncated),
            "dataset_truncated": dataset_truncated,
            "episode_metrics": dict(info["episode_metrics"]),
            "residual_decision": residual_decision.as_dict(),
        }
        self.last_native_record = record
        if self.macro_accumulator.complete:
            chunk = self.controller.active_chunk
            if chunk is None:
                raise RuntimeError("completed transition lost its residual chunk")
            self.chunk_outcomes.append(
                {
                    "episode_id": self.episode_id,
                    "chunk_id": chunk.chunk_id,
                    "ending_native_transition": self.native_transition_count,
                    "native_steps": self.macro_accumulator.native_steps,
                    "maximum_native_steps": self.macro_accumulator.maximum_native_steps,
                    "full_scheduled_execution": bool(
                        self.macro_accumulator.native_steps
                        == self.macro_accumulator.maximum_native_steps
                        and not chunk.cancelled
                    ),
                    "cancelled": chunk.cancelled,
                    "cancellation_reason": chunk.cancellation_reason,
                    "cancellation_native_step": chunk.cancellation_native_step,
                    "residual_influence_native_steps": (
                        chunk.residual_influence_native_steps
                    ),
                    "actor_action": chunk.action.tolist(),
                }
            )
            if task_ended:
                next_strategy_observation = np.zeros(16, dtype=np.float64)
                self._record_episode(info)
                self.pending_reset = self.native_transition_count < self.native_budget
            elif dataset_truncated:
                next_strategy_observation = np.zeros(16, dtype=np.float64)
            else:
                prepared = self.controller.prepare_native_step(self.observation)
                if prepared is None:
                    raise RuntimeError("macro completion missed actor boundary")
                self.prepared_actor_observation = prepared
                next_strategy_observation = prepared
            transition = self.macro_accumulator.finish(next_strategy_observation)
            self.replay.append(transition)
            self.macro_transition_count += 1
            self.macro_accumulator = None
        self._perform_due_updates()
        return record

    def run_to_budget(self) -> None:
        while self.native_transition_count < self.native_budget:
            self.advance_native()
        if self.gradient_update_count != self.update_budget:
            raise RuntimeError("exact gradient-update budget was not consumed")

    def walltime_s(self) -> float:
        return (
            self.completed_walltime_s + time.perf_counter() - self.segment_wall_started
        )

    def process_time_s(self) -> float:
        return (
            self.completed_process_time_s
            + time.process_time()
            - self.segment_process_started
        )

    def runtime_state(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "training_seed": self.training_seed,
            "attempt_id": self.attempt_id,
            "native_budget": self.native_budget,
            "update_budget": self.update_budget,
            "warmup_native_transitions": self.warmup_native_transitions,
            "batch_size": self.batch_size,
            "verify_track_content": self.verify_track_content,
            "native_transition_count": self.native_transition_count,
            "macro_transition_count": self.macro_transition_count,
            "gradient_update_count": self.gradient_update_count,
            "actor_sample_count": self.actor_sample_count,
            "episode_id": self.episode_id,
            "episode_native_step": self.episode_native_step,
            "pending_reset": self.pending_reset,
            "current_task": dict(self.current_task or {}),
            "observation": self.observation.copy(),
            "prepared_actor_observation": (
                None
                if self.prepared_actor_observation is None
                else self.prepared_actor_observation.copy()
            ),
            "macro_accumulator": (
                None
                if self.macro_accumulator is None
                else self.macro_accumulator.state_dict()
            ),
            "task_stream": self.task_stream.state_dict(),
            "policy_generator_state": self.policy_generator.get_state(),
            "update_generator_state": self.update_generator.get_state(),
            "warmup_rng_state": self.warmup_rng.bit_generator.state,
            "replay_rng_state": self.replay_rng.bit_generator.state,
            "training_outcomes": list(self.training_outcomes),
            "chunk_outcomes": list(self.chunk_outcomes),
            "update_diagnostics": list(self.update_diagnostics),
            "cancellation_counts": dict(self.cancellation_counts),
            "proposal_count": self.proposal_count,
            "residual_attempt_count": self.residual_attempt_count,
            "residual_accept_count": self.residual_accept_count,
            "influenced_native_count": self.influenced_native_count,
            "last_native_record": self.last_native_record,
            "walltime_state": {"completed_segments_s": self.walltime_s()},
            "process_time_state": {"completed_segments_s": self.process_time_s()},
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        from .checkpointing import save_checkpoint

        return save_checkpoint(
            path,
            agent_state=self.agent.state_dict(),
            replay=self.replay,
            environment=self.environment,
            controller=self.controller,
            runtime=self.runtime_state(),
            configuration_sha256=self.configuration_sha256,
            source_hashes=self.source_hashes,
        )

    @classmethod
    def restore(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None,
        configuration_sha256: str,
        source_hashes: Mapping[str, str],
        attempt_id: str,
    ) -> TrainingRunner:
        from .checkpointing import load_checkpoint, restore_global_rng_states

        loaded = load_checkpoint(
            path,
            expected_sha256=expected_sha256,
            expected_configuration_sha256=configuration_sha256,
            expected_source_hashes=source_hashes,
            expected_attempt_id=attempt_id,
        )
        runtime = loaded.runtime
        self = cls.__new__(cls)
        self.spec = method_spec(runtime["method"])
        self.method = self.spec.name
        self.training_seed = int(runtime["training_seed"])
        self.attempt_id = loaded.attempt_id
        self.configuration_sha256 = configuration_sha256
        self.source_hashes = dict(source_hashes)
        for field in (
            "native_budget",
            "update_budget",
            "warmup_native_transitions",
            "batch_size",
        ):
            setattr(self, field, int(runtime[field]))
        self.verify_track_content = bool(runtime["verify_track_content"])
        self.update_budget_accumulator = UpdateBudgetAccumulator(
            warmup=self.warmup_native_transitions,
            native_budget=self.native_budget,
            update_budget=self.update_budget,
        )
        self.agent = ConstrainedResidualSAC(self.method, self.training_seed)
        self.agent.load_state_dict(loaded.agent_state)
        self.replay = loaded.replay
        self.environment = loaded.environment
        self.controller = loaded.controller
        self.task_stream = CanonicalTrainingStream.from_state_dict(
            runtime["task_stream"]
        )
        self.policy_generator = _torch_generator(
            self.training_seed, POLICY_TORCH_DOMAIN
        )
        self.policy_generator.set_state(runtime["policy_generator_state"])
        self.update_generator = _torch_generator(
            self.training_seed, UPDATE_TORCH_DOMAIN
        )
        self.update_generator.set_state(runtime["update_generator_state"])
        self.warmup_rng = _numpy_generator(self.training_seed, WARMUP_NUMPY_DOMAIN)
        self.warmup_rng.bit_generator.state = runtime["warmup_rng_state"]
        self.replay_rng = _numpy_generator(self.training_seed, REPLAY_NUMPY_DOMAIN)
        self.replay_rng.bit_generator.state = runtime["replay_rng_state"]
        for field in (
            "native_transition_count",
            "macro_transition_count",
            "gradient_update_count",
            "actor_sample_count",
            "episode_id",
            "episode_native_step",
            "proposal_count",
            "residual_attempt_count",
            "residual_accept_count",
            "influenced_native_count",
        ):
            setattr(self, field, int(runtime[field]))
        self.pending_reset = bool(runtime["pending_reset"])
        self.current_task = dict(runtime["current_task"])
        self.observation = np.asarray(runtime["observation"], dtype=np.float64)
        prepared = runtime["prepared_actor_observation"]
        self.prepared_actor_observation = (
            None if prepared is None else np.asarray(prepared, dtype=np.float64)
        )
        accumulator = runtime["macro_accumulator"]
        self.macro_accumulator = (
            None
            if accumulator is None
            else MacroAccumulator.from_state_dict(accumulator)
        )
        self.training_outcomes = list(runtime["training_outcomes"])
        self.chunk_outcomes = list(runtime["chunk_outcomes"])
        self.update_diagnostics = list(runtime["update_diagnostics"])
        self.cancellation_counts = dict(runtime["cancellation_counts"])
        self.last_native_record = runtime["last_native_record"]
        self.completed_walltime_s = float(
            runtime["walltime_state"]["completed_segments_s"]
        )
        self.completed_process_time_s = float(
            runtime["process_time_state"]["completed_segments_s"]
        )
        self.segment_wall_started = time.perf_counter()
        self.segment_process_started = time.process_time()
        restore_global_rng_states(loaded.global_rng_states)
        return self

    def close(self) -> None:
        self.environment.close()
