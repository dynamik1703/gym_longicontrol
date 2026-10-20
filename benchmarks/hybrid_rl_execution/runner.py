"""Exact-budget Hybrid SMDP collection and learning state machine."""

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
from benchmarks.hybrid_rl_design.controller import HybridTargetSpeedController
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .agent import ConstrainedSAC
from .macro_transition import DECISION_INTERVAL_NATIVE_STEPS, MacroAccumulator
from .replay import MacroReplay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STREAM_LEDGER = (
    ROOT / "benchmarks/constrained_v2_collector_replication/stream_ledger.json"
)
TRAINING_SEEDS = (11, 29, 47)
NATIVE_BUDGET = 300_000
GRADIENT_UPDATE_BUDGET = 30_000
WARMUP_NATIVE_TRANSITIONS = 10_000
BATCH_SIZE = 256
REPLAY_CAPACITY = 300_000
TIME_LIMIT_NATIVE_STEPS = 1_800
CANONICAL_TASK = TaskSpecification(140.0, 0.0)

POLICY_TORCH_DOMAIN = 0x504F4C59
UPDATE_TORCH_DOMAIN = 0x55504454
WARMUP_NUMPY_DOMAIN = 0x5741524D
REPLAY_NUMPY_DOMAIN = 0x52504C59


def cumulative_update_target(native_transitions: int) -> int:
    """Return the frozen Revision-3 optimizer-update target at native step N."""

    if native_transitions <= WARMUP_NATIVE_TRANSITIONS:
        return 0
    return min(
        GRADIENT_UPDATE_BUDGET,
        ((int(native_transitions) - WARMUP_NATIVE_TRANSITIONS) * GRADIENT_UPDATE_BUDGET)
        // (NATIVE_BUDGET - WARMUP_NATIVE_TRANSITIONS),
    )


def _seed_value(seed: int, domain: int) -> int:
    return int(np.random.SeedSequence([seed, domain]).generate_state(1)[0])


def _torch_generator(seed: int, domain: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(_seed_value(seed, domain))


def _numpy_generator(seed: int, domain: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, domain]))


class CanonicalTrainingStream:
    """Cursor over the already frozen training stream for one policy seed."""

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
    """Linearly spread the update budget over post-warm-up native steps."""

    def __init__(
        self,
        *,
        warmup: int = WARMUP_NATIVE_TRANSITIONS,
        native_budget: int = NATIVE_BUDGET,
        update_budget: int = GRADIENT_UPDATE_BUDGET,
    ) -> None:
        if native_budget <= 0 or update_budget <= 0 or not 0 <= warmup < native_budget:
            raise ValueError("invalid update budget")
        self.warmup = int(warmup)
        self.native_budget = int(native_budget)
        self.update_budget = int(update_budget)

    def cumulative_target(self, native_transitions: int) -> int:
        if native_transitions <= self.warmup:
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


class HybridRunner:
    """Advance exactly one native environment step per ``advance_native`` call."""

    def __init__(
        self,
        *,
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
        if training_seed not in TRAINING_SEEDS:
            raise ValueError("training seed is outside the frozen matrix")
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
        self.agent = ConstrainedSAC(self.training_seed)
        self.replay = MacroReplay(replay_capacity)
        self.task_stream = CanonicalTrainingStream(self.training_seed)
        self.environment = environment or make_training_environment()
        self.controller = HybridTargetSpeedController(strategy_enabled=True)
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
        self.prepared_strategy_observation: np.ndarray | None = None
        self.held_strategy_action: float | None = None
        self.remaining_native_steps_in_block = 0
        self.macro_accumulator: MacroAccumulator | None = None
        self.training_outcomes: list[dict[str, Any]] = []
        self.update_diagnostics: list[dict[str, float]] = []
        self.intervention_counts: dict[str, int] = {}
        self.proposal_count = 0
        self.override_count = 0
        self.influenced_native_count = 0
        self.last_native_record: dict[str, Any] | None = None
        self.completed_walltime_s = 0.0
        self.segment_started = time.perf_counter()
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
        self.prepared_strategy_observation = None
        self.held_strategy_action = None
        self.remaining_native_steps_in_block = 0
        self.macro_accumulator = None
        self.pending_reset = False

    def _start_macro_block(self) -> None:
        if self.macro_accumulator is not None:
            raise RuntimeError("strategy action changed inside a macro block")
        if self.prepared_strategy_observation is None:
            strategy_observation = self.controller.prepare_strategy_step(
                self.observation
            )
        else:
            strategy_observation = self.prepared_strategy_observation
            self.prepared_strategy_observation = None
        if self.native_transition_count < self.warmup_native_transitions:
            strategy_action = float(self.warmup_rng.uniform(-1.0, 1.0))
        else:
            strategy_action = self.agent.act(
                strategy_observation,
                deterministic=False,
                generator=self.policy_generator,
            )
        self.actor_sample_count += 1
        self.held_strategy_action = strategy_action
        self.macro_accumulator = MacroAccumulator(strategy_observation, strategy_action)
        self.remaining_native_steps_in_block = min(
            DECISION_INTERVAL_NATIVE_STEPS,
            self.native_budget - self.native_transition_count,
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

    def advance_native(self) -> dict[str, Any]:
        if self.native_transition_count >= self.native_budget:
            raise RuntimeError("frozen native transition budget is exhausted")
        if self.pending_reset:
            self._reset_to_next_task()
        if self.macro_accumulator is None:
            self._start_macro_block()
        assert self.macro_accumulator is not None
        assert self.held_strategy_action is not None
        action = self.controller.action(self.observation, self.held_strategy_action)
        next_observation, _, terminated, env_truncated, info = self.environment.step(
            [action]
        )
        self.controller.observe_executed_action(action)
        self.native_transition_count += 1
        self.episode_native_step += 1
        self.remaining_native_steps_in_block -= 1
        budget_truncated = bool(
            self.native_transition_count == self.native_budget
            and not terminated
            and not env_truncated
        )
        ended = bool(terminated or env_truncated)
        metrics = EpisodeMetrics(**info["episode_metrics"])
        terminal_failure = bool(
            ended and (not metrics.completed or metrics.travel_time_s > 140.0)
        )
        decision = self.controller.last_decision
        if decision is None:
            raise RuntimeError("shield decision was not recorded")
        intervention = abs(decision.proposed_native_action - decision.executed_action)
        speed_cost = max(0.0, float(info["velocity_m_s"] - info["speed_limit_m_s"]))
        self.macro_accumulator.add_native_step(
            reward=-1000.0 * float(info["step_energy_kwh"]),
            speed_cost=speed_cost,
            terminal_cost=float(terminal_failure),
            intervention_cost=intervention,
            terminated=bool(terminated),
            truncated=bool(env_truncated or budget_truncated),
        )
        self.proposal_count += 1
        self.override_count += int(decision.overridden)
        self.influenced_native_count += int(decision.native_action_influence > 0.0)
        reason = decision.intervention_reason
        self.intervention_counts[reason] = self.intervention_counts.get(reason, 0) + 1
        self.observation = np.asarray(next_observation, dtype=np.float64)
        record = {
            "native_transition_count": self.native_transition_count,
            "episode_id": self.episode_id,
            "episode_native_step": self.episode_native_step,
            "strategy_action": self.held_strategy_action,
            "native_action": float(action),
            "step_energy_kwh": float(info["step_energy_kwh"]),
            "speed_cost_m_s": speed_cost,
            "terminal_cost": float(terminal_failure),
            "intervention_deviation": intervention,
            "terminated": bool(terminated),
            "truncated": bool(env_truncated),
            "budget_truncated": budget_truncated,
            "shield": decision.as_dict(),
        }
        self.last_native_record = record
        if self.macro_accumulator.complete:
            if ended:
                next_strategy_observation = np.zeros(16, dtype=np.float64)
                self._record_episode(info)
                self.pending_reset = self.native_transition_count < self.native_budget
            elif budget_truncated:
                next_strategy_observation = np.zeros(16, dtype=np.float64)
            else:
                self.prepared_strategy_observation = (
                    self.controller.prepare_strategy_step(self.observation)
                )
                next_strategy_observation = self.prepared_strategy_observation
            transition = self.macro_accumulator.finish(next_strategy_observation)
            self.replay.append(transition)
            self.macro_transition_count += 1
            self.macro_accumulator = None
            self.held_strategy_action = None
            self.remaining_native_steps_in_block = 0
        # Revision 3 evaluates the cumulative optimizer budget at every native
        # step. Replay still contains only completed SMDP macro-transitions.
        self._perform_due_updates()
        return record

    def run_to_budget(self) -> None:
        while self.native_transition_count < self.native_budget:
            self.advance_native()
        if self.gradient_update_count != self.update_budget:
            raise RuntimeError("exact gradient-update budget was not consumed")

    def walltime_s(self) -> float:
        return self.completed_walltime_s + time.perf_counter() - self.segment_started

    def runtime_state(self) -> dict[str, Any]:
        return {
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
            "prepared_strategy_observation": (
                None
                if self.prepared_strategy_observation is None
                else self.prepared_strategy_observation.copy()
            ),
            "held_strategy_action": self.held_strategy_action,
            "remaining_native_steps_in_block": self.remaining_native_steps_in_block,
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
            "update_diagnostics": list(self.update_diagnostics),
            "intervention_counts": dict(self.intervention_counts),
            "proposal_count": self.proposal_count,
            "override_count": self.override_count,
            "influenced_native_count": self.influenced_native_count,
            "last_native_record": self.last_native_record,
            "walltime_state": {"completed_segments_s": self.walltime_s()},
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
    ) -> HybridRunner:
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
        self.training_seed = int(runtime["training_seed"])
        self.attempt_id = loaded.attempt_id
        self.configuration_sha256 = configuration_sha256
        self.source_hashes = dict(source_hashes)
        self.native_budget = int(runtime["native_budget"])
        self.update_budget = int(runtime["update_budget"])
        self.warmup_native_transitions = int(runtime["warmup_native_transitions"])
        self.batch_size = int(runtime["batch_size"])
        self.verify_track_content = bool(runtime["verify_track_content"])
        self.update_budget_accumulator = UpdateBudgetAccumulator(
            warmup=self.warmup_native_transitions,
            native_budget=self.native_budget,
            update_budget=self.update_budget,
        )
        self.agent = ConstrainedSAC(self.training_seed)
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
            "remaining_native_steps_in_block",
            "proposal_count",
            "override_count",
            "influenced_native_count",
        ):
            setattr(self, field, int(runtime[field]))
        self.pending_reset = bool(runtime["pending_reset"])
        self.current_task = dict(runtime["current_task"])
        self.observation = np.asarray(runtime["observation"], dtype=np.float64)
        prepared = runtime["prepared_strategy_observation"]
        self.prepared_strategy_observation = (
            None if prepared is None else np.asarray(prepared, dtype=np.float64)
        )
        held = runtime["held_strategy_action"]
        self.held_strategy_action = None if held is None else float(held)
        accumulator = runtime["macro_accumulator"]
        self.macro_accumulator = (
            None
            if accumulator is None
            else MacroAccumulator.from_state_dict(accumulator)
        )
        self.training_outcomes = list(runtime["training_outcomes"])
        self.update_diagnostics = list(runtime["update_diagnostics"])
        self.intervention_counts = dict(runtime["intervention_counts"])
        self.last_native_record = runtime["last_native_record"]
        self.completed_walltime_s = float(
            runtime["walltime_state"]["completed_segments_s"]
        )
        self.segment_started = time.perf_counter()
        restore_global_rng_states(loaded.global_rng_states)
        return self

    def exact_snapshot(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_state(),
            "agent": self.agent.state_dict(),
            "replay": self.replay.state_dict(),
            "environment_state": self.environment.unwrapped.state,
            "controller_state": self.controller.state,
        }

    def close(self) -> None:
        self.environment.close()
