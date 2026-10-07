"""End-to-end runner for the frozen projected-goal CRL depth study.

The command is intentionally disabled until a later commit explicitly sets
both main-training authorization flags. Importing or testing this module never
starts physical policy learning.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .checkpointing import file_sha256, load_checkpoint, save_checkpoint
from .config import (
    DEVELOPMENT_CHECKPOINTS,
    DEVELOPMENT_TRACKS,
    PLANNED_COMPLETE_UPDATE_CYCLES,
    PLANNED_DEPTHS,
    PLANNED_SEEDS,
    PLANNED_TRANSITIONS_PER_POLICY,
    VALIDATION_TRACKS,
    ReferenceCoreConfig,
)
from .diagnostics import DiagnosticsAccumulator, learning_batch_diagnostics
from .evaluation import evaluate_actor, make_environment
from .execution import (
    DEFAULT_OUTPUT_ROOT,
    EXPECTED_CONFIGURATION_SHA256,
    SCIENTIFIC_SHA256,
    ExclusiveStudyLock,
    TrainingTrackStream,
    atomic_json,
    initialize_study_root,
    interrupt_policy,
    permit_exact_resume,
    policy_key,
    should_update,
    start_policy,
    update_manifest,
    utc_now,
    validate_resume_provenance,
    validate_validation_gate,
    verify_preflight,
)
from .goal_adapter import (
    OutcomeScales,
    augment_policy_state,
    canonical_command,
    physical_outcome,
    project_outcome,
    projected_goal_is_canonical,
)
from .learner import ReferenceLearner
from .losses import tanh_gaussian_sample
from .projected_adapter import contrastive_batch_from_recorded
from .replay import TransitionReplay

CANONICAL_TASK = TaskSpecification(140.0, 0.0)
DIAGNOSTIC_UPDATE_INTERVAL = 100
ACTOR_RNG_DOMAIN = 0x41435452
UPDATE_RNG_DOMAIN = 0x55504454
FUTURE_RNG_DOMAIN = 0x46555452


def _rng(seed: int, domain: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), domain]))


def _jax_key(seed: int, domain: int):
    value = np.random.SeedSequence([seed, domain]).generate_state(1)[0]
    return jax.random.PRNGKey(value)


def _raw_outcome(info: Mapping[str, Any], previous_position_m: float) -> np.ndarray:
    return physical_outcome(
        position_m=float(info["position_m"]),
        previous_position_m=float(previous_position_m),
        elapsed_time_s=float(info["elapsed_time_s"]),
        max_speed_violation_m_s=float(info["max_speed_violation_m_s"]),
    )


def _failure_mode(metrics: EpisodeMetrics) -> str:
    failures = []
    if not metrics.completed:
        failures.append("incomplete")
    if metrics.travel_time_s > CANONICAL_TASK.max_time_s:
        failures.append("deadline")
    if metrics.max_speed_violation_m_s > CANONICAL_TASK.max_speed_violation_m_s:
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        array = np.asarray(value)
        if array.size == 1:
            result[key] = float(array.item())
    return result


def _observed_future_is_canonical(outcome: np.ndarray, *, terminated: bool) -> bool:
    projected = project_outcome(
        np.asarray(outcome, dtype=np.float64),
        terminated=np.asarray(terminated, dtype=np.bool_),
        task=CANONICAL_TASK,
    )
    return bool(projected_goal_is_canonical(projected))


class PolicyRunner:
    """One depth/seed process with checkpointable collection and updates."""

    def __init__(
        self,
        *,
        depth: int,
        training_seed: int,
        configuration_sha256: str = EXPECTED_CONFIGURATION_SHA256,
        scientific_source_sha256: Mapping[str, str] = SCIENTIFIC_SHA256,
        execution_source_sha256: Mapping[str, str],
        environment=None,
    ):
        if depth not in PLANNED_DEPTHS or training_seed not in PLANNED_SEEDS:
            raise ValueError("policy is outside the frozen depth/seed matrix")
        self.depth = int(depth)
        self.training_seed = int(training_seed)
        self.configuration_sha256 = configuration_sha256
        self.scientific_source_sha256 = dict(scientific_source_sha256)
        self.execution_source_sha256 = dict(execution_source_sha256)
        self.config = ReferenceCoreConfig(depth=depth)
        self.learner, self.learner_state = ReferenceLearner.create(
            self.config, seed=training_seed
        )
        self.replay = TransitionReplay(
            PLANNED_TRANSITIONS_PER_POLICY,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
        )
        self.track_stream = TrainingTrackStream(training_seed)
        self.future_rng = _rng(training_seed, FUTURE_RNG_DOMAIN)
        self.actor_rng_key = _jax_key(training_seed, ACTOR_RNG_DOMAIN)
        self.update_rng_key = _jax_key(training_seed, UPDATE_RNG_DOMAIN)
        self.environment = (
            environment if environment is not None else make_environment()
        )
        self.track_seeds_used: list[int] = []
        first_track = self._next_track_seed()
        self.observation, info = self.environment.reset(seed=first_track)
        self.current_outcome = _raw_outcome(info, float(info["position_m"]))
        self.transition_count = 0
        self.update_cycle_count = 0
        self.episode_id = 0
        self.episode_step = 0
        self.current_episode_ended = False
        self.diagnostics = DiagnosticsAccumulator()
        self.training_outcomes: list[dict[str, Any]] = []
        self.development_completed: list[int] = []
        self.first_canonical_positive_available_transition: int | None = None
        self.first_physical_canonical_success_transition: int | None = None
        self.started_at = time.perf_counter()

    def _next_track_seed(self) -> int:
        seed = self.track_stream.next_seed()
        self.track_seeds_used.append(seed)
        return seed

    def _policy_state(self) -> np.ndarray:
        state = augment_policy_state(
            self.observation, self.current_outcome, OutcomeScales()
        )
        if state.shape != (self.config.state_dim,):
            raise RuntimeError("policy state dimension changed")
        return state

    def _sample_action(self, state: np.ndarray) -> np.ndarray:
        self.actor_rng_key, sample_key = jax.random.split(self.actor_rng_key)
        state_goal = jnp.concatenate(
            (
                jnp.asarray(state[None, :], dtype=jnp.float32),
                jnp.asarray(canonical_command()[None, :], dtype=jnp.float32),
            ),
            axis=-1,
        )
        mean, log_std = self.learner.actor.apply(
            self.learner_state.actor.params, state_goal
        )
        noise = jax.random.normal(sample_key, mean.shape, dtype=mean.dtype)
        action, _log_probability = tanh_gaussian_sample(mean, log_std, noise)
        return np.asarray(action[0], dtype=np.float64)

    def _record_episode(self, info: Mapping[str, Any]) -> None:
        metrics = EpisodeMetrics(**info["episode_metrics"])
        feasible = is_feasible(metrics, CANONICAL_TASK)
        if feasible and self.first_physical_canonical_success_transition is None:
            self.first_physical_canonical_success_transition = self.transition_count
        row = {
            "episode_id": self.episode_id,
            "ending_transition": self.transition_count,
            "track_seed": self.track_seeds_used[-1],
            "canonical_success": feasible,
            "failure_mode": _failure_mode(metrics),
            "deadline_compliant": metrics.travel_time_s <= 140.0,
            "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
            "final_position_m": float(info["position_m"]),
            "final_route_progress": min(float(info["position_m"]) / 1000.0, 1.0),
            "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
            **asdict(metrics),
        }
        self.training_outcomes.append(row)

    def collect_transition(self) -> bool:
        """Collect one native transition and perform a scheduled update if due."""

        if self.transition_count >= PLANNED_TRANSITIONS_PER_POLICY:
            raise RuntimeError("frozen 300k transition budget is already exhausted")
        state = self._policy_state()
        action = self._sample_action(state)
        previous_position = float(self.current_outcome[0])
        observation, historical_reward, terminated, truncated, info = (
            self.environment.step(action)
        )
        outcome = _raw_outcome(info, previous_position)
        self.replay.append(
            state=state,
            action=action,
            source_outcome=self.current_outcome,
            outcome=outcome,
            historical_reward=float(historical_reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            episode_id=self.episode_id,
            episode_step=self.episode_step,
        )
        self.transition_count += 1
        self.episode_step += 1
        if (
            self.first_canonical_positive_available_transition is None
            and _observed_future_is_canonical(outcome, terminated=bool(terminated))
        ):
            self.first_canonical_positive_available_transition = self.transition_count
        ended = bool(terminated or truncated)
        self.current_episode_ended = ended
        if ended:
            self._record_episode(info)
            if self.transition_count < PLANNED_TRANSITIONS_PER_POLICY:
                self.episode_id += 1
                self.episode_step = 0
                self.current_episode_ended = False
                next_seed = self._next_track_seed()
                self.observation, reset_info = self.environment.reset(seed=next_seed)
                self.current_outcome = _raw_outcome(
                    reset_info, float(reset_info["position_m"])
                )
            else:
                self.observation = observation
                self.current_outcome = outcome
        else:
            self.observation = observation
            self.current_outcome = outcome
        if should_update(self.transition_count):
            self._update_once()
            return True
        return False

    def _update_once(self) -> None:
        sampled = self.replay.sample(
            self.config_batch_size,
            gamma=self.config.gamma,
            rng=self.future_rng,
        )
        batch = contrastive_batch_from_recorded(
            states=sampled.states,
            actions=sampled.actions,
            future_outcomes=sampled.future_outcomes,
            future_terminated=sampled.future_terminated,
            task=CANONICAL_TASK,
            historical_reward=sampled.historical_rewards,
        )
        self.update_rng_key, update_key = jax.random.split(self.update_rng_key)
        next_cycle = self.update_cycle_count + 1
        detailed = bool(
            next_cycle == 1
            or next_cycle % DIAGNOSTIC_UPDATE_INTERVAL == 0
            or self.transition_count in DEVELOPMENT_CHECKPOINTS
        )
        learning = (
            learning_batch_diagnostics(
                self.learner, self.learner_state, batch, update_key
            )
            if detailed
            else None
        )
        self.learner_state, update_metrics = self.learner.update(
            self.learner_state, batch, update_key
        )
        self.update_cycle_count = next_cycle
        if learning is None:
            scalars = _scalar_metrics(update_metrics)
            learning = {
                "optimization": {
                    "actor_loss": scalars["actor_loss"],
                    "critic_loss": scalars["critic_loss"],
                    "alpha_loss": scalars["alpha_loss"],
                    "log_alpha": float(self.learner_state.alpha.params["log_alpha"]),
                    "mean_log_probability": scalars["mean_log_probability"],
                    "estimated_entropy": -scalars["mean_log_probability"],
                    "gradient_norms_recorded": False,
                    "nonfinite_value_count": int(
                        sum(not np.isfinite(value) for value in scalars.values())
                    ),
                },
                "contrastive": {
                    "infonce_component": scalars["classification_loss"],
                    "logsumexp_regularization": scalars["logsumexp_regularizer"],
                    "detailed_scores_recorded": False,
                },
                "canonical_query": None,
            }
        projected = np.asarray(batch.critic_goals, dtype=np.float64)
        self.diagnostics.record(
            transition_count=self.transition_count,
            update_cycle=self.update_cycle_count,
            sampled=sampled,
            projected_goals=projected,
            dt_s=0.1,
            learning=learning,
        )

    @property
    def config_batch_size(self) -> int:
        return 256

    def runtime_state(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "training_seed": self.training_seed,
            "transition_count": self.transition_count,
            "update_cycle_count": self.update_cycle_count,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "current_episode_ended": self.current_episode_ended,
            "observation": np.asarray(self.observation, dtype=np.float64),
            "current_outcome": np.asarray(self.current_outcome, dtype=np.float64),
            "actor_rng_key": np.asarray(self.actor_rng_key),
            "update_rng_key": np.asarray(self.update_rng_key),
            "future_rng_state": self.future_rng.bit_generator.state,
            "track_rng_state": self.track_stream.state,
            "track_rng_provenance": self.track_stream.provenance,
            "track_seeds_used": list(self.track_seeds_used),
            "diagnostics": self.diagnostics.to_state(),
            "training_outcomes": list(self.training_outcomes),
            "development_completed": list(self.development_completed),
            "first_canonical_positive_available_transition": (
                self.first_canonical_positive_available_transition
            ),
            "first_physical_canonical_success_transition": (
                self.first_physical_canonical_success_transition
            ),
        }

    def save(self, path: str | Path) -> str:
        return save_checkpoint(
            path,
            learner_state=self.learner_state,
            replay=self.replay,
            environment=self.environment,
            runtime=self.runtime_state(),
            configuration_sha256=self.configuration_sha256,
            scientific_source_sha256=self.scientific_source_sha256,
            execution_source_sha256=self.execution_source_sha256,
        )

    @classmethod
    def restore(
        cls, path: str | Path, *, execution_source_sha256: Mapping[str, str]
    ) -> PolicyRunner:
        raw = __import__("pickle").loads(Path(path).read_bytes())
        runtime = raw["runtime"]
        depth = int(runtime["depth"])
        seed = int(runtime["training_seed"])
        config = ReferenceCoreConfig(depth=depth)
        learner, template = ReferenceLearner.create(config, seed=seed)
        loaded = load_checkpoint(
            path,
            learner_state_template=template,
            expected_configuration_sha256=EXPECTED_CONFIGURATION_SHA256,
            expected_scientific_source_sha256=SCIENTIFIC_SHA256,
            expected_execution_source_sha256=execution_source_sha256,
        )
        self = cls.__new__(cls)
        self.depth = depth
        self.training_seed = seed
        self.configuration_sha256 = loaded.configuration_sha256
        self.scientific_source_sha256 = loaded.scientific_source_sha256
        self.execution_source_sha256 = loaded.execution_source_sha256
        self.config = config
        self.learner = learner
        self.learner_state = loaded.learner_state
        self.replay = loaded.replay
        self.environment = loaded.environment
        self.transition_count = int(runtime["transition_count"])
        self.update_cycle_count = int(runtime["update_cycle_count"])
        self.episode_id = int(runtime["episode_id"])
        self.episode_step = int(runtime["episode_step"])
        self.current_episode_ended = bool(runtime["current_episode_ended"])
        self.observation = np.asarray(runtime["observation"], dtype=np.float64)
        self.current_outcome = np.asarray(runtime["current_outcome"], dtype=np.float64)
        self.actor_rng_key = jnp.asarray(runtime["actor_rng_key"])
        self.update_rng_key = jnp.asarray(runtime["update_rng_key"])
        self.future_rng = _rng(seed, FUTURE_RNG_DOMAIN)
        self.future_rng.bit_generator.state = runtime["future_rng_state"]
        self.track_stream = TrainingTrackStream(seed, runtime["track_rng_state"])
        self.track_seeds_used = list(runtime["track_seeds_used"])
        self.diagnostics = DiagnosticsAccumulator.from_state(runtime["diagnostics"])
        self.training_outcomes = list(runtime["training_outcomes"])
        self.development_completed = list(runtime["development_completed"])
        self.first_canonical_positive_available_transition = runtime[
            "first_canonical_positive_available_transition"
        ]
        self.first_physical_canonical_success_transition = runtime[
            "first_physical_canonical_success_transition"
        ]
        self.started_at = time.perf_counter()
        return self

    def development_evaluation(self) -> dict[str, Any]:
        before = (
            np.asarray(self.actor_rng_key).copy(),
            np.asarray(self.update_rng_key).copy(),
            json.dumps(self.future_rng.bit_generator.state, sort_keys=True),
            json.dumps(self.track_stream.state, sort_keys=True),
        )
        result = evaluate_actor(
            self.learner,
            self.learner_state.actor.params,
            DEVELOPMENT_TRACKS,
            split="development",
        )
        after = (
            np.asarray(self.actor_rng_key),
            np.asarray(self.update_rng_key),
            json.dumps(self.future_rng.bit_generator.state, sort_keys=True),
            json.dumps(self.track_stream.state, sort_keys=True),
        )
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        if before[2:] != after[2:]:
            raise RuntimeError("Development evaluation perturbed training RNG")
        return result

    def outcomes_summary(self) -> dict[str, Any]:
        rows = self.training_outcomes
        successes = [row for row in rows if row["canonical_success"]]
        modes = sorted({row["failure_mode"] for row in rows})
        incomplete_progress = [
            row["final_route_progress"] for row in rows if not row["completed"]
        ]
        partial = None
        if not self.current_episode_ended and self.episode_step:
            partial = {
                "episode_id": self.episode_id,
                "transition_count": self.episode_step,
                "position_m": float(self.current_outcome[0]),
                "elapsed_time_s": float(self.current_outcome[2]),
                "max_speed_violation_m_s": float(self.current_outcome[3]),
                "final_route_progress": min(
                    float(self.current_outcome[0]) / 1000.0, 1.0
                ),
            }
        return {
            "total_ended_training_episodes": len(rows),
            "canonical_successful_episodes": len(successes),
            "first_canonical_success_transition": (
                self.first_physical_canonical_success_transition
            ),
            "completion_count": sum(row["completed"] for row in rows),
            "completed_by_deadline_count": sum(
                row["completed"] and row["deadline_compliant"] for row in rows
            ),
            "speed_compliant_count": sum(row["speed_compliant"] for row in rows),
            "exclusive_failure_mode_counts": {
                mode: sum(row["failure_mode"] == mode for row in rows) for mode in modes
            },
            "incomplete_final_route_progress": incomplete_progress,
            "feasible_episode_energy_kwh": [
                row["feasible_energy_kwh"] for row in successes
            ],
            "partial_episode_at_training_boundary": partial,
            "episodes": rows,
        }

    def close(self) -> None:
        self.environment.close()


def _checkpoint_policy(
    runner: PolicyRunner,
    *,
    run_directory: Path,
    item: dict[str, Any],
    study_root: Path,
    manifest: dict[str, Any],
) -> None:
    transition = runner.transition_count
    final_directory = run_directory / f"step-{transition:06d}"
    if final_directory.exists():
        raise FileExistsError(f"checkpoint already exists: {final_directory}")
    development = runner.development_evaluation()
    manifest["actual_resources_across_attempts"][
        "development_simulator_transitions"
    ] += development["summary"]["evaluation_simulator_transitions"]
    checkpoint_dir = run_directory / (
        f".step-{transition:06d}.pending-{uuid.uuid4().hex}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "exact-resume.ckpt"
    runner.development_completed.append(transition)
    try:
        checkpoint_hash = runner.save(checkpoint_path)
        atomic_json(checkpoint_dir / "development-result.json", development)
        atomic_json(
            checkpoint_dir / "diagnostics.json",
            {
                "summary": {
                    **runner.diagnostics.summary(),
                    "first_canonical_positive_available_transition": (
                        runner.first_canonical_positive_available_transition
                    ),
                },
                "latest": (
                    runner.diagnostics.rows[-1] if runner.diagnostics.rows else None
                ),
            },
        )
        checkpoint_dir.replace(final_directory)
    except BaseException:
        runner.development_completed.remove(transition)
        raise
    checkpoint_dir = final_directory
    checkpoint_path = checkpoint_dir / "exact-resume.ckpt"
    record = {
        "transition_count": transition,
        "complete_update_cycles": runner.update_cycle_count,
        "checkpoint_path": str(checkpoint_path.relative_to(study_root)),
        "checkpoint_sha256": checkpoint_hash,
        "development_result_path": str(
            (checkpoint_dir / "development-result.json").relative_to(study_root)
        ),
        "development_simulator_transitions": development["summary"][
            "evaluation_simulator_transitions"
        ],
    }
    item["development_checkpoints"].append(record)
    item["native_transitions"] = transition
    item["complete_update_cycles"] = runner.update_cycle_count
    attempt = item["attempts"][-1]
    attempt["native_transitions"] = transition
    attempt["complete_update_cycles"] = runner.update_cycle_count
    _refresh_resource_totals(manifest)
    update_manifest(study_root, manifest)


def execute_policy(
    *,
    depth: int,
    training_seed: int,
    study_root: Path,
    manifest: dict[str, Any],
    resume_checkpoint: Path | None = None,
) -> None:
    key = policy_key(depth, training_seed)
    item = manifest["policies"][key]
    run_directory = study_root / f"depth-{depth}" / f"seed-{training_seed}"
    if resume_checkpoint is None:
        start_policy(manifest, depth, training_seed)
        run_directory.mkdir(parents=True, exist_ok=False)
        runner = PolicyRunner(
            depth=depth,
            training_seed=training_seed,
            execution_source_sha256=manifest["provenance"]["execution_source_sha256"],
        )
    else:
        if file_sha256(resume_checkpoint) != item.get("interruption_checkpoint_sha256"):
            raise RuntimeError("resume checkpoint hash differs from manifest")
        permit_exact_resume(item, item["interruption_checkpoint_sha256"])
        runner = PolicyRunner.restore(
            resume_checkpoint,
            execution_source_sha256=manifest["provenance"]["execution_source_sha256"],
        )
        if (
            runner.transition_count in DEVELOPMENT_CHECKPOINTS
            and runner.transition_count not in runner.development_completed
        ):
            _checkpoint_policy(
                runner,
                run_directory=run_directory,
                item=item,
                study_root=study_root,
                manifest=manifest,
            )
    update_manifest(study_root, manifest)
    try:
        while runner.transition_count < PLANNED_TRANSITIONS_PER_POLICY:
            runner.collect_transition()
            if runner.transition_count in DEVELOPMENT_CHECKPOINTS:
                _checkpoint_policy(
                    runner,
                    run_directory=run_directory,
                    item=item,
                    study_root=study_root,
                    manifest=manifest,
                )
        if runner.update_cycle_count != PLANNED_COMPLETE_UPDATE_CYCLES:
            raise RuntimeError("policy did not perform exactly 7,250 update cycles")
        final_checkpoint = run_directory / "step-300000" / "exact-resume.ckpt"
        item.update(
            {
                "status": "COMPLETED",
                "completed_at": utc_now(),
                "native_transitions": runner.transition_count,
                "complete_update_cycles": runner.update_cycle_count,
                "final_checkpoint_path": str(final_checkpoint.relative_to(study_root)),
                "final_checkpoint_sha256": file_sha256(final_checkpoint),
                "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
                "scientific_source_sha256": dict(SCIENTIFIC_SHA256),
                "execution_source_sha256": dict(
                    manifest["provenance"]["execution_source_sha256"]
                ),
                "training_track_stream": runner.track_stream.provenance,
                "training_track_seed_count": len(runner.track_seeds_used),
            }
        )
        attempt = item["attempts"][-1]
        attempt.update(
            {
                "status": "COMPLETED",
                "completed_at": utc_now(),
                "native_transitions": runner.transition_count,
                "complete_update_cycles": runner.update_cycle_count,
            }
        )
        atomic_json(run_directory / "training-outcomes.json", runner.outcomes_summary())
        atomic_json(
            run_directory / "crl-diagnostics.json",
            {
                "summary": {
                    **runner.diagnostics.summary(),
                    "first_canonical_positive_available_transition": (
                        runner.first_canonical_positive_available_transition
                    ),
                },
                "rows": runner.diagnostics.rows,
            },
        )
    except BaseException as error:
        checkpoint = run_directory / "interruption-exact-resume.ckpt"
        checkpoint_hash = runner.save(checkpoint)
        interrupt_policy(
            item,
            transition_count=runner.transition_count,
            update_cycles=runner.update_cycle_count,
            reason=f"{type(error).__name__}: {error}",
        )
        item["interruption_checkpoint_path"] = str(checkpoint.relative_to(study_root))
        item["interruption_checkpoint_sha256"] = checkpoint_hash
        _refresh_resource_totals(manifest)
        update_manifest(study_root, manifest)
        raise
    finally:
        runner.close()
    update_manifest(study_root, manifest)


def _refresh_resource_totals(manifest: dict[str, Any]) -> None:
    attempts = [
        attempt
        for item in manifest["policies"].values()
        for attempt in item["attempts"]
    ]
    manifest["actual_resources_across_attempts"]["native_simulator_transitions"] = sum(
        attempt["native_transitions"] for attempt in attempts
    )
    manifest["actual_resources_across_attempts"]["complete_update_cycles"] = sum(
        attempt["complete_update_cycles"] for attempt in attempts
    )


def run_study(repo_root: str | Path) -> Path:
    """Run all six policies only after a separate authorization commit."""

    root = Path(repo_root).resolve()
    provenance = verify_preflight(root, require_training_authorization=True)
    output = root / DEFAULT_OUTPUT_ROOT
    study_root, manifest = initialize_study_root(output, provenance)
    with ExclusiveStudyLock(study_root, manifest["study_id"]):
        manifest["status"] = "RUNNING"
        update_manifest(study_root, manifest)
        try:
            for depth in PLANNED_DEPTHS:
                for seed in PLANNED_SEEDS:
                    execute_policy(
                        depth=depth,
                        training_seed=seed,
                        study_root=study_root,
                        manifest=manifest,
                    )
                    _refresh_resource_totals(manifest)
                    update_manifest(study_root, manifest)
            manifest["status"] = "COMPLETED"
            manifest["completed_at"] = utc_now()
            update_manifest(study_root, manifest)
        except BaseException as error:
            _refresh_resource_totals(manifest)
            manifest["status"] = "INTERRUPTED"
            manifest["interruption"] = {
                "at": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
            }
            update_manifest(study_root, manifest)
            raise
    return study_root / "manifest.json"


def resume_study(repo_root: str | Path) -> Path:
    """Exactly continue one preserved interrupted policy, then the matrix."""

    root = Path(repo_root).resolve()
    current_provenance = verify_preflight(root, require_training_authorization=True)
    study_root = root / DEFAULT_OUTPUT_ROOT
    manifest_path = study_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("no preserved CRL study exists to resume")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_resume_provenance(manifest, current_provenance)
    if manifest.get("status") != "INTERRUPTED":
        raise RuntimeError("exact resume requires an INTERRUPTED study")
    interrupted = [
        item
        for item in manifest["policies"].values()
        if item["status"] == "INTERRUPTED"
    ]
    if len(interrupted) != 1:
        raise RuntimeError("exact resume requires exactly one interrupted policy")
    target = interrupted[0]
    checkpoint = study_root / target["interruption_checkpoint_path"]
    if not checkpoint.is_file():
        raise RuntimeError("preserved exact-resume checkpoint is missing")
    with ExclusiveStudyLock(study_root, manifest["study_id"]):
        manifest["status"] = "RUNNING"
        update_manifest(study_root, manifest)
        try:
            execute_policy(
                depth=target["depth"],
                training_seed=target["training_seed"],
                study_root=study_root,
                manifest=manifest,
                resume_checkpoint=checkpoint,
            )
            for depth in PLANNED_DEPTHS:
                for seed in PLANNED_SEEDS:
                    item = manifest["policies"][policy_key(depth, seed)]
                    if item["status"] == "COMPLETED":
                        continue
                    if item["status"] != "NOT_STARTED":
                        raise RuntimeError(
                            "resume found an unhandled active/partial policy"
                        )
                    execute_policy(
                        depth=depth,
                        training_seed=seed,
                        study_root=study_root,
                        manifest=manifest,
                    )
            _refresh_resource_totals(manifest)
            manifest["status"] = "COMPLETED"
            manifest["completed_at"] = utc_now()
            update_manifest(study_root, manifest)
        except BaseException as error:
            _refresh_resource_totals(manifest)
            manifest["status"] = "INTERRUPTED"
            manifest["interruption"] = {
                "at": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
            }
            update_manifest(study_root, manifest)
            raise
    return manifest_path


def validate_study(repo_root: str | Path) -> Path:
    """Run the single authorized final Validation pass after all six policies."""

    root = Path(repo_root).resolve()
    current_provenance = verify_preflight(root, require_training_authorization=True)
    study_root = root / DEFAULT_OUTPUT_ROOT
    manifest_path = study_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("no completed CRL study exists for Validation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_resume_provenance(manifest, current_provenance)
    validate_validation_gate(study_root, manifest)
    with ExclusiveStudyLock(study_root, manifest["study_id"]):
        opened_at = utc_now()
        manifest["validation_opened"] = True
        manifest["validation"] = {
            "status": "RUNNING",
            "opened_at": opened_at,
            "tracks": list(VALIDATION_TRACKS),
            "policy_results": [],
        }
        update_manifest(study_root, manifest)
        validation_episodes: list[dict[str, Any]] = []
        try:
            for depth in PLANNED_DEPTHS:
                for seed in PLANNED_SEEDS:
                    item = manifest["policies"][policy_key(depth, seed)]
                    checkpoint = study_root / item["final_checkpoint_path"]
                    runner = PolicyRunner.restore(
                        checkpoint,
                        execution_source_sha256=current_provenance[
                            "execution_source_sha256"
                        ],
                    )
                    try:
                        result = evaluate_actor(
                            runner.learner,
                            runner.learner_state.actor.params,
                            VALIDATION_TRACKS,
                            split="validation",
                        )
                    finally:
                        runner.close()
                    result = {
                        "depth": depth,
                        "training_seed": seed,
                        "final_checkpoint_sha256": item["final_checkpoint_sha256"],
                        **result,
                    }
                    result_path = (
                        study_root
                        / f"depth-{depth}"
                        / f"seed-{seed}"
                        / "validation-result.json"
                    )
                    atomic_json(result_path, result)
                    item["validation_result_path"] = str(
                        result_path.relative_to(study_root)
                    )
                    item["validation_result_sha256"] = file_sha256(result_path)
                    item["validation_episode_count"] = len(result["episodes"])
                    manifest["actual_resources_across_attempts"][
                        "validation_simulator_transitions"
                    ] += result["summary"]["evaluation_simulator_transitions"]
                    manifest["validation"]["policy_results"].append(
                        {
                            "depth": depth,
                            "training_seed": seed,
                            "path": item["validation_result_path"],
                            "sha256": item["validation_result_sha256"],
                            "episode_count": item["validation_episode_count"],
                        }
                    )
                    for episode in result["episodes"]:
                        validation_episodes.append(
                            {
                                "depth": depth,
                                "training_seed": seed,
                                **episode,
                            }
                        )
                    atomic_json(
                        study_root / "validation-episodes.json",
                        validation_episodes,
                    )
                    update_manifest(study_root, manifest)
            if len(validation_episodes) != 54:
                raise RuntimeError("final Validation did not produce 54 episodes")
            manifest["validation"].update(
                {
                    "status": "COMPLETED",
                    "completed_at": utc_now(),
                    "episode_count": len(validation_episodes),
                    "episodes_path": "validation-episodes.json",
                    "episodes_sha256": file_sha256(
                        study_root / "validation-episodes.json"
                    ),
                }
            )
            update_manifest(study_root, manifest)
        except BaseException as error:
            manifest["validation"].update(
                {
                    "status": "INTERRUPTED",
                    "interrupted_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            update_manifest(study_root, manifest)
            raise
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "run", "resume", "validate"),
        help="run/resume/validate are authorization-gated",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    if args.command == "preflight":
        print(
            json.dumps(
                verify_preflight(
                    repo_root,
                    require_clean=True,
                    require_training_authorization=False,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    operations = {
        "run": run_study,
        "resume": resume_study,
        "validate": validate_study,
    }
    operation = operations[args.command]
    print(operation(repo_root), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
