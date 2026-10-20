"""State-machine runner for exact-resumable canonical GCSL collection."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .checkpointing import load_checkpoint, restore_rng_states, save_checkpoint
from .config import CHECKPOINTS, TRAINING_SEEDS, GCSLConfig, should_update
from .diagnostics import DiagnosticsAccumulator, learning_diagnostics
from .evaluation import failure_mode, make_environment, raw_outcome
from .goal_adapter import (
    OutcomeScales,
    augment_policy_state,
    canonical_command,
    project_outcome,
    projected_goal_is_canonical,
)
from .learner import GCSLLearner
from .replay import TrajectoryReplay
from .split import CanonicalTaskStream, verify_runtime_track

CANONICAL_TASK = TaskSpecification(140.0, 0.0)
POLICY_RNG_DOMAIN = 0x504F4C59
REPLAY_RNG_DOMAIN = 0x52504C59


def _numpy_rng(seed: int, domain: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, domain]))


def _torch_generator(seed: int, domain: int) -> torch.Generator:
    value = int(np.random.SeedSequence([seed, domain]).generate_state(1)[0])
    return torch.Generator(device="cpu").manual_seed(value)


class PolicyRunner:
    """One policy attempt with explicit collection/reset/update boundaries."""

    def __init__(
        self,
        *,
        training_seed: int,
        attempt_id: str | None,
        configuration_sha256: str,
        source_hashes: Mapping[str, str],
        environment=None,
        verify_track_content: bool = True,
    ):
        if training_seed not in TRAINING_SEEDS:
            raise ValueError("training seed is outside the frozen matrix")
        self.training_seed = int(training_seed)
        self.attempt_id = attempt_id or str(uuid.uuid4())
        self.configuration_sha256 = configuration_sha256
        self.source_hashes = dict(source_hashes)
        self.config = GCSLConfig()
        self.learner = GCSLLearner(self.config, seed=training_seed)
        if self.learner.parameter_count != 270_338:
            raise RuntimeError("Frozen actor parameter count changed")
        self.replay = TrajectoryReplay(
            self.config.replay_capacity,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
        )
        self.task_stream = CanonicalTaskStream(training_seed)
        self.policy_action_rng = _torch_generator(training_seed, POLICY_RNG_DOMAIN)
        self.replay_rng = _numpy_rng(training_seed, REPLAY_RNG_DOMAIN)
        self.environment = environment or make_environment()
        self.verify_track_content = bool(verify_track_content)
        self.current_task: dict[str, Any] | None = None
        self.training_exposures: list[dict[str, Any]] = []
        self.transition_count = 0
        self.update_count = 0
        self.episode_id = 0
        self.episode_step = 0
        self.pending_update = False
        self.pending_reset = False
        self.diagnostics = DiagnosticsAccumulator()
        self.training_outcomes: list[dict[str, Any]] = []
        self.diagnostic_checkpoints_completed: list[int] = []
        self.first_canonical_support_transition: int | None = None
        self.first_canonical_success_transition: int | None = None
        self.completed_walltime_s = 0.0
        self.segment_started = time.perf_counter()
        self._reset_to_next_task(initial=True)

    def _reset_to_next_task(self, *, initial: bool = False) -> None:
        if not initial:
            if not self.pending_reset:
                raise RuntimeError("episode reset was not pending")
            self.episode_id += 1
            self.episode_step = 0
        row = self.task_stream.next_row()
        observation, info = self.environment.reset(seed=int(row["reset_seed"]))
        if self.verify_track_content:
            verify_runtime_track(row, self.environment.unwrapped.track)
        self.current_task = row
        self.observation = np.asarray(observation, dtype=np.float64)
        self.current_outcome = raw_outcome(info, float(info["position_m"]))
        self.pending_reset = False

    def _policy_state(self) -> np.ndarray:
        state = augment_policy_state(
            self.observation, self.current_outcome, OutcomeScales()
        )
        if state.shape != (self.config.state_dim,):
            raise RuntimeError("policy state dimension changed")
        return state

    def _sample_action(self, state: np.ndarray) -> np.ndarray:
        return self.learner.sample_action(
            state[None, :],
            canonical_command()[None, :],
            generator=self.policy_action_rng,
        )[0].astype(np.float64)

    def _record_episode(self, info: Mapping[str, Any]) -> None:
        metrics = EpisodeMetrics(**info["episode_metrics"])
        feasible = is_feasible(metrics, CANONICAL_TASK)
        if feasible and self.first_canonical_success_transition is None:
            self.first_canonical_success_transition = self.transition_count
        self.training_outcomes.append(
            {
                "episode_id": self.episode_id,
                "ending_transition": self.transition_count,
                "task": dict(self.current_task or {}),
                "canonical_success": feasible,
                "failure_mode": failure_mode(metrics),
                "deadline_compliant": metrics.travel_time_s <= 140.0,
                "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
                "final_progress": min(float(info["position_m"]) / 1000.0, 1.0),
                "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
                **asdict(metrics),
            }
        )

    def collect_only(self) -> None:
        """Execute one action and stop before reset or supervised update."""

        if self.pending_reset or self.pending_update:
            raise RuntimeError("resolve pending boundary before another action")
        if self.transition_count >= 300_000:
            raise RuntimeError("the frozen transition budget is exhausted")
        if self.episode_step == 0:
            self.training_exposures.append(dict(self.current_task or {}))
        state = self._policy_state()
        action = self._sample_action(state)
        previous_position = float(self.current_outcome[0])
        observation, reward, terminated, truncated, info = self.environment.step(action)
        outcome = raw_outcome(info, previous_position)
        self.replay.append(
            state=state,
            action=action,
            source_outcome=self.current_outcome,
            outcome=outcome,
            historical_reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            episode_id=self.episode_id,
            episode_step=self.episode_step,
        )
        self.transition_count += 1
        self.episode_step += 1
        self.observation = np.asarray(observation, dtype=np.float64)
        self.current_outcome = outcome
        if (
            self.first_canonical_support_transition is None
            and projected_goal_is_canonical(
                project_outcome(
                    outcome,
                    terminated=np.asarray(bool(terminated), dtype=np.bool_),
                    task=CANONICAL_TASK,
                )
            )
        ):
            self.first_canonical_support_transition = self.transition_count
        ended = bool(terminated or truncated)
        if ended:
            self._record_episode(info)
            self.pending_reset = self.transition_count < 300_000
        self.pending_update = should_update(self.transition_count)

    def reset_pending_episode(self) -> None:
        self._reset_to_next_task()

    def perform_pending_update(self) -> dict[str, Any]:
        if not self.pending_update:
            raise RuntimeError("supervised update was not pending")
        sampled = self.replay.sample(
            self.config.batch_size,
            gamma=self.config.future_discount,
            rng=self.replay_rng,
            task=CANONICAL_TASK,
        )
        next_update = self.update_count + 1
        detailed = bool(
            next_update == 1
            or next_update % 100 == 0
            or self.transition_count in CHECKPOINTS
        )
        if detailed:
            learning = learning_diagnostics(self.learner, sampled)
        metrics = self.learner.update(sampled.batch)
        if not detailed:
            learning = {
                "transformed_action_nll": metrics["action_nll"],
                "gradient_norm": metrics["gradient_norm"],
                "parameter_norm": metrics["parameter_norm"],
                "nonfinite_value_count": int(
                    not all(np.isfinite(value) for value in metrics.values())
                ),
                "detailed_diagnostic": False,
            }
        else:
            learning["detailed_diagnostic"] = True
            learning["post_update"] = metrics
        self.update_count = next_update
        self.pending_update = False
        return self.diagnostics.record(
            transition_count=self.transition_count,
            update_cycle=self.update_count,
            sampled=sampled,
            learning=learning,
        )

    def advance(self) -> None:
        """Complete pending work, collect once, then honor the frozen ordering."""

        if self.pending_reset:
            self.reset_pending_episode()
        if self.pending_update:
            self.perform_pending_update()
        self.collect_only()
        # Preserve the old GCSL ordering: reset after terminal, then update.
        if self.pending_reset:
            self.reset_pending_episode()
        if self.pending_update:
            self.perform_pending_update()

    def walltime_s(self) -> float:
        return self.completed_walltime_s + time.perf_counter() - self.segment_started

    def runtime_state(self) -> dict[str, Any]:
        return {
            "training_seed": self.training_seed,
            "attempt_id": self.attempt_id,
            "transition_count": self.transition_count,
            "update_count": self.update_count,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "pending_update": self.pending_update,
            "pending_reset": self.pending_reset,
            "observation": np.asarray(self.observation, dtype=np.float64),
            "current_outcome": np.asarray(self.current_outcome, dtype=np.float64),
            "policy_action_rng_state": self.policy_action_rng.get_state(),
            "replay_rng_state": self.replay_rng.bit_generator.state,
            "task_stream_state": self.task_stream.state_dict(),
            "current_task": dict(self.current_task or {}),
            "training_exposures": list(self.training_exposures),
            "diagnostics": self.diagnostics.to_state(),
            "training_outcomes": list(self.training_outcomes),
            "diagnostic_checkpoints_completed": list(
                self.diagnostic_checkpoints_completed
            ),
            "first_canonical_support_transition": (
                self.first_canonical_support_transition
            ),
            "first_canonical_success_transition": (
                self.first_canonical_success_transition
            ),
            "walltime_state": {"completed_segments_s": self.walltime_s()},
        }

    def save(self, path: str | Path) -> dict[str, Any]:
        return save_checkpoint(
            path,
            learner_state=self.learner.state_dict(),
            replay=self.replay,
            environment=self.environment,
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
        verify_track_content: bool = True,
    ) -> PolicyRunner:
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
        self.config = GCSLConfig()
        self.learner = GCSLLearner(self.config, seed=self.training_seed)
        self.learner.load_state_dict(loaded.learner_state)
        self.replay = loaded.replay
        self.environment = loaded.environment
        self.verify_track_content = bool(verify_track_content)
        self.task_stream = CanonicalTaskStream.from_state(runtime["task_stream_state"])
        self.policy_action_rng = _torch_generator(self.training_seed, POLICY_RNG_DOMAIN)
        self.policy_action_rng.set_state(runtime["policy_action_rng_state"])
        self.replay_rng = _numpy_rng(self.training_seed, REPLAY_RNG_DOMAIN)
        self.replay_rng.bit_generator.state = runtime["replay_rng_state"]
        self.current_task = dict(runtime["current_task"])
        self.training_exposures = list(runtime["training_exposures"])
        self.transition_count = int(runtime["transition_count"])
        self.update_count = int(runtime["update_count"])
        self.episode_id = int(runtime["episode_id"])
        self.episode_step = int(runtime["episode_step"])
        self.pending_update = bool(runtime["pending_update"])
        self.pending_reset = bool(runtime["pending_reset"])
        self.observation = np.asarray(runtime["observation"], dtype=np.float64)
        self.current_outcome = np.asarray(runtime["current_outcome"], dtype=np.float64)
        self.diagnostics = DiagnosticsAccumulator.from_state(runtime["diagnostics"])
        self.training_outcomes = list(runtime["training_outcomes"])
        self.diagnostic_checkpoints_completed = list(
            runtime["diagnostic_checkpoints_completed"]
        )
        self.first_canonical_support_transition = runtime[
            "first_canonical_support_transition"
        ]
        self.first_canonical_success_transition = runtime[
            "first_canonical_success_transition"
        ]
        self.completed_walltime_s = float(
            runtime["walltime_state"]["completed_segments_s"]
        )
        self.segment_started = time.perf_counter()
        restore_rng_states(loaded.rng_states)
        return self

    def outcomes_summary(self) -> dict[str, Any]:
        successes = [row for row in self.training_outcomes if row["canonical_success"]]
        modes = sorted({row["failure_mode"] for row in self.training_outcomes})
        return {
            "completed_training_episodes": len(self.training_outcomes),
            "canonical_training_successes": len(successes),
            "first_canonical_success_transition": (
                self.first_canonical_success_transition
            ),
            "completion_count": sum(row["completed"] for row in self.training_outcomes),
            "exclusive_failure_mode_counts": {
                mode: sum(row["failure_mode"] == mode for row in self.training_outcomes)
                for mode in modes
            },
            "feasible_only_energy_kwh": [
                row["feasible_energy_kwh"] for row in successes
            ],
            "episodes": self.training_outcomes,
        }

    def exact_snapshot(self) -> dict[str, Any]:
        """Small comparison payload used only by recovery verification tests."""

        return {
            "transition_count": self.transition_count,
            "update_count": self.update_count,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "pending_update": self.pending_update,
            "pending_reset": self.pending_reset,
            "observation": self.observation.copy(),
            "current_outcome": self.current_outcome.copy(),
            "policy_rng": self.policy_action_rng.get_state().clone(),
            "replay_rng": json.dumps(
                self.replay_rng.bit_generator.state, sort_keys=True
            ),
            "task_stream": self.task_stream.state_dict(),
            "replay": self.replay.to_state(),
            "learner": self.learner.state_dict(),
        }

    def close(self) -> None:
        self.environment.close()
