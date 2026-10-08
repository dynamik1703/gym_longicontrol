"""End-to-end GCSL V1 runner, disabled until a later authorization commit."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .checkpointing import file_sha256, load_checkpoint, save_checkpoint
from .config import (
    DEVELOPMENT_CHECKPOINTS,
    DEVELOPMENT_TRACKS,
    PLANNED_SEEDS,
    PLANNED_TRANSITIONS_PER_POLICY,
    PLANNED_UPDATE_CYCLES,
    VALIDATION_TRACKS,
    GCSLConfig,
    should_update,
)
from .diagnostics import DiagnosticsAccumulator, learning_diagnostics
from .evaluation import evaluate_policy, failure_mode, make_environment, raw_outcome
from .execution import (
    DEFAULT_OUTPUT_ROOT,
    ExclusiveStudyLock,
    TrainingTrackStream,
    atomic_json,
    initialize_study_root,
    interrupt_policy,
    permit_exact_resume,
    policy_key,
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
    project_outcome,
    projected_goal_is_canonical,
)
from .learner import GCSLLearner
from .replay import TrajectoryReplay

CANONICAL_TASK = TaskSpecification(140.0, 0.0)
POLICY_RNG_DOMAIN = 0x504F4C59
REPLAY_RNG_DOMAIN = 0x52504C59


def _numpy_rng(seed: int, domain: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, domain]))


def _torch_generator(seed: int, domain: int) -> torch.Generator:
    value = int(np.random.SeedSequence([seed, domain]).generate_state(1)[0])
    return torch.Generator(device="cpu").manual_seed(value)


def _observed_future_is_canonical(outcome, *, terminated: bool) -> bool:
    goal = project_outcome(
        np.asarray(outcome, dtype=np.float64),
        terminated=np.asarray(terminated, dtype=np.bool_),
        task=CANONICAL_TASK,
    )
    return bool(projected_goal_is_canonical(goal))


class PolicyRunner:
    """One seed with checkpointable collection, replay, RNGs, and optimizer."""

    def __init__(
        self,
        *,
        training_seed: int,
        configuration_sha256: str,
        scientific_source_sha256: Mapping[str, str],
        execution_source_sha256: Mapping[str, str],
        environment=None,
    ):
        if training_seed not in PLANNED_SEEDS:
            raise ValueError("training seed is outside the frozen matrix")
        self.training_seed = int(training_seed)
        self.configuration_sha256 = configuration_sha256
        self.scientific_source_sha256 = dict(scientific_source_sha256)
        self.execution_source_sha256 = dict(execution_source_sha256)
        self.config = GCSLConfig()
        self.learner = GCSLLearner(self.config, seed=training_seed)
        self.replay = TrajectoryReplay(
            self.config.replay_capacity,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
        )
        self.track_stream = TrainingTrackStream(training_seed)
        self.policy_action_rng = _torch_generator(training_seed, POLICY_RNG_DOMAIN)
        self.replay_rng = _numpy_rng(training_seed, REPLAY_RNG_DOMAIN)
        self.environment = environment or make_environment()
        self.track_seeds_used: list[int] = []
        observation, info = self.environment.reset(seed=self._next_track_seed())
        self.observation = observation
        self.current_outcome = raw_outcome(info, float(info["position_m"]))
        self.transition_count = 0
        self.update_cycle_count = 0
        self.episode_id = 0
        self.episode_step = 0
        self.current_episode_ended = False
        self.diagnostics = DiagnosticsAccumulator()
        self.training_outcomes: list[dict[str, Any]] = []
        self.development_completed: list[int] = []
        self.first_canonical_target_available_transition: int | None = None
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
        return self.learner.sample_action(
            state[None, :],
            canonical_command()[None, :],
            generator=self.policy_action_rng,
        )[0].astype(np.float64)

    def _record_episode(self, info: Mapping[str, Any]) -> None:
        metrics = EpisodeMetrics(**info["episode_metrics"])
        feasible = is_feasible(metrics, CANONICAL_TASK)
        if feasible and self.first_physical_canonical_success_transition is None:
            self.first_physical_canonical_success_transition = self.transition_count
        self.training_outcomes.append(
            {
                "episode_id": self.episode_id,
                "ending_transition": self.transition_count,
                "track_seed": self.track_seeds_used[-1],
                "canonical_success": feasible,
                "failure_mode": failure_mode(metrics),
                "deadline_compliant": metrics.travel_time_s <= 140.0,
                "speed_compliant": metrics.max_speed_violation_m_s <= 0.0,
                "final_progress": min(float(info["position_m"]) / 1000.0, 1.0),
                "feasible_energy_kwh": metrics.energy_kwh if feasible else None,
                **asdict(metrics),
            }
        )

    def collect_transition(self) -> bool:
        if self.transition_count >= PLANNED_TRANSITIONS_PER_POLICY:
            raise RuntimeError("the frozen 300,000-transition budget is exhausted")
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
        if (
            self.first_canonical_target_available_transition is None
            and _observed_future_is_canonical(outcome, terminated=bool(terminated))
        ):
            self.first_canonical_target_available_transition = self.transition_count
        ended = bool(terminated or truncated)
        self.current_episode_ended = ended
        if ended:
            self._record_episode(info)
            if self.transition_count < PLANNED_TRANSITIONS_PER_POLICY:
                self.episode_id += 1
                self.episode_step = 0
                self.current_episode_ended = False
                observation, reset_info = self.environment.reset(
                    seed=self._next_track_seed()
                )
                self.observation = observation
                self.current_outcome = raw_outcome(
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
            self.config.batch_size,
            gamma=self.config.future_discount,
            rng=self.replay_rng,
            task=CANONICAL_TASK,
        )
        next_cycle = self.update_cycle_count + 1
        detailed = bool(
            next_cycle == 1
            or next_cycle % 100 == 0
            or self.transition_count in DEVELOPMENT_CHECKPOINTS
        )
        learning = (
            learning_diagnostics(self.learner, sampled) if detailed else None
        )
        metrics = self.learner.update(sampled.batch)
        if learning is None:
            learning = {
                "action_nll": metrics["action_nll"],
                "gradient_norm": metrics["gradient_norm"],
                "parameter_norm": metrics["parameter_norm"],
                "nonfinite_value_count": int(
                    not all(np.isfinite(value) for value in metrics.values())
                ),
                "detailed_diagnostic": False,
                "goal_dependence": None,
            }
        else:
            learning["detailed_diagnostic"] = True
            for name, value in metrics.items():
                learning[f"post_update_{name}"] = value
        self.update_cycle_count = next_cycle
        self.diagnostics.record(
            transition_count=self.transition_count,
            update_cycle=self.update_cycle_count,
            sampled=sampled,
            learning=learning,
        )

    def runtime_state(self) -> dict[str, Any]:
        return {
            "training_seed": self.training_seed,
            "transition_count": self.transition_count,
            "update_cycle_count": self.update_cycle_count,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "current_episode_ended": self.current_episode_ended,
            "observation": np.asarray(self.observation, dtype=np.float64),
            "current_outcome": np.asarray(self.current_outcome, dtype=np.float64),
            "policy_action_rng_state": self.policy_action_rng.get_state(),
            "replay_rng_state": self.replay_rng.bit_generator.state,
            "track_rng_state": self.track_stream.state,
            "track_seeds_used": list(self.track_seeds_used),
            "diagnostics": self.diagnostics.to_state(),
            "training_outcomes": list(self.training_outcomes),
            "development_completed": list(self.development_completed),
            "first_canonical_target_available_transition": (
                self.first_canonical_target_available_transition
            ),
            "first_physical_canonical_success_transition": (
                self.first_physical_canonical_success_transition
            ),
        }

    def save(self, path: str | Path) -> str:
        return save_checkpoint(
            path,
            learner_state=self.learner.state_dict(),
            replay=self.replay,
            environment=self.environment,
            runtime=self.runtime_state(),
            configuration_sha256=self.configuration_sha256,
            scientific_source_sha256=self.scientific_source_sha256,
            execution_source_sha256=self.execution_source_sha256,
        )

    @classmethod
    def restore(
        cls,
        path: str | Path,
        *,
        configuration_sha256: str,
        scientific_source_sha256: Mapping[str, str],
        execution_source_sha256: Mapping[str, str],
    ) -> PolicyRunner:
        loaded = load_checkpoint(
            path,
            expected_configuration_sha256=configuration_sha256,
            expected_scientific_source_sha256=scientific_source_sha256,
            expected_execution_source_sha256=execution_source_sha256,
        )
        runtime = loaded.runtime
        self = cls.__new__(cls)
        self.training_seed = int(runtime["training_seed"])
        self.configuration_sha256 = configuration_sha256
        self.scientific_source_sha256 = dict(scientific_source_sha256)
        self.execution_source_sha256 = dict(execution_source_sha256)
        self.config = GCSLConfig()
        self.learner = GCSLLearner(self.config, seed=self.training_seed)
        self.learner.load_state_dict(loaded.learner_state)
        self.replay = loaded.replay
        self.environment = loaded.environment
        self.track_stream = TrainingTrackStream(
            self.training_seed, runtime["track_rng_state"]
        )
        self.policy_action_rng = _torch_generator(
            self.training_seed, POLICY_RNG_DOMAIN
        )
        self.policy_action_rng.set_state(runtime["policy_action_rng_state"])
        self.replay_rng = _numpy_rng(self.training_seed, REPLAY_RNG_DOMAIN)
        self.replay_rng.bit_generator.state = runtime["replay_rng_state"]
        self.track_seeds_used = list(runtime["track_seeds_used"])
        self.transition_count = int(runtime["transition_count"])
        self.update_cycle_count = int(runtime["update_cycle_count"])
        self.episode_id = int(runtime["episode_id"])
        self.episode_step = int(runtime["episode_step"])
        self.current_episode_ended = bool(runtime["current_episode_ended"])
        self.observation = np.asarray(runtime["observation"], dtype=np.float64)
        self.current_outcome = np.asarray(
            runtime["current_outcome"], dtype=np.float64
        )
        self.diagnostics = DiagnosticsAccumulator.from_state(runtime["diagnostics"])
        self.training_outcomes = list(runtime["training_outcomes"])
        self.development_completed = list(runtime["development_completed"])
        self.first_canonical_target_available_transition = runtime[
            "first_canonical_target_available_transition"
        ]
        self.first_physical_canonical_success_transition = runtime[
            "first_physical_canonical_success_transition"
        ]
        self.started_at = time.perf_counter()
        return self

    def development_evaluation(self) -> dict[str, Any]:
        before = (
            self.policy_action_rng.get_state().clone(),
            json.dumps(self.replay_rng.bit_generator.state, sort_keys=True),
            json.dumps(self.track_stream.state, sort_keys=True),
        )
        result = evaluate_policy(
            self.learner, DEVELOPMENT_TRACKS, split="development"
        )
        if not torch.equal(before[0], self.policy_action_rng.get_state()):
            raise RuntimeError("Development evaluation perturbed policy RNG")
        if before[1:] != (
            json.dumps(self.replay_rng.bit_generator.state, sort_keys=True),
            json.dumps(self.track_stream.state, sort_keys=True),
        ):
            raise RuntimeError("Development evaluation perturbed training RNG")
        return result

    def outcomes_summary(self) -> dict[str, Any]:
        rows = self.training_outcomes
        successes = [row for row in rows if row["canonical_success"]]
        modes = sorted({row["failure_mode"] for row in rows})
        return {
            "total_ended_training_episodes": len(rows),
            "canonical_successful_episodes": len(successes),
            "first_canonical_success_transition": (
                self.first_physical_canonical_success_transition
            ),
            "completion_count": sum(row["completed"] for row in rows),
            "exclusive_failure_mode_counts": {
                mode: sum(row["failure_mode"] == mode for row in rows)
                for mode in modes
            },
            "feasible_episode_energy_kwh": [
                row["feasible_energy_kwh"] for row in successes
            ],
            "episodes": rows,
        }

    def close(self) -> None:
        self.environment.close()


def _checkpoint_policy(runner, run_directory, item, study_root, manifest) -> None:
    transition = runner.transition_count
    final_directory = run_directory / f"step-{transition:06d}"
    if final_directory.exists():
        raise FileExistsError(f"checkpoint already exists: {final_directory}")
    development = runner.development_evaluation()
    manifest["actual_resources_across_attempts"][
        "development_simulator_transitions"
    ] += development["summary"]["evaluation_simulator_transitions"]
    pending = run_directory / f".step-{transition:06d}.pending-{uuid.uuid4().hex}"
    pending.mkdir(parents=True, exist_ok=False)
    runner.development_completed.append(transition)
    try:
        checkpoint = pending / "exact-resume.ckpt"
        checkpoint_hash = runner.save(checkpoint)
        atomic_json(pending / "development-result.json", development)
        atomic_json(
            pending / "diagnostics.json",
            {
                "summary": {
                    **runner.diagnostics.summary(),
                    "first_canonical_target_available_transition": (
                        runner.first_canonical_target_available_transition
                    ),
                },
                "latest": runner.diagnostics.rows[-1],
            },
        )
        pending.replace(final_directory)
    except BaseException:
        runner.development_completed.remove(transition)
        raise
    checkpoint = final_directory / "exact-resume.ckpt"
    item["development_checkpoints"].append(
        {
            "transition_count": transition,
            "complete_update_cycles": runner.update_cycle_count,
            "checkpoint_path": str(checkpoint.relative_to(study_root)),
            "checkpoint_sha256": checkpoint_hash,
            "development_result_path": str(
                (final_directory / "development-result.json").relative_to(study_root)
            ),
        }
    )
    item["native_transitions"] = transition
    item["complete_update_cycles"] = runner.update_cycle_count
    update_manifest(study_root, manifest)


def _refresh_totals(manifest: dict[str, Any]) -> None:
    attempts = [
        attempt
        for item in manifest["policies"].values()
        for attempt in item["attempts"]
    ]
    manifest["actual_resources_across_attempts"][
        "native_simulator_transitions"
    ] = sum(item["native_transitions"] for item in attempts)
    manifest["actual_resources_across_attempts"]["complete_update_cycles"] = sum(
        item["complete_update_cycles"] for item in attempts
    )


def execute_policy(*, seed, study_root, manifest, resume_checkpoint=None) -> None:
    item = manifest["policies"][policy_key(seed)]
    run_directory = study_root / f"seed-{seed}"
    provenance = manifest["provenance"]
    if resume_checkpoint is None:
        start_policy(manifest, seed)
        run_directory.mkdir(parents=True, exist_ok=False)
        runner = PolicyRunner(
            training_seed=seed,
            configuration_sha256=provenance["configuration_sha256"],
            scientific_source_sha256=provenance["scientific_source_sha256"],
            execution_source_sha256=provenance["execution_source_sha256"],
        )
    else:
        if file_sha256(resume_checkpoint) != item["interruption_checkpoint_sha256"]:
            raise RuntimeError("resume checkpoint differs from preserved attempt")
        permit_exact_resume(item, item["interruption_checkpoint_sha256"])
        runner = PolicyRunner.restore(
            resume_checkpoint,
            configuration_sha256=provenance["configuration_sha256"],
            scientific_source_sha256=provenance["scientific_source_sha256"],
            execution_source_sha256=provenance["execution_source_sha256"],
        )
    update_manifest(study_root, manifest)
    try:
        while runner.transition_count < PLANNED_TRANSITIONS_PER_POLICY:
            runner.collect_transition()
            if runner.transition_count in DEVELOPMENT_CHECKPOINTS:
                _checkpoint_policy(
                    runner, run_directory, item, study_root, manifest
                )
        if runner.update_cycle_count != PLANNED_UPDATE_CYCLES:
            raise RuntimeError("policy did not perform exactly 7,250 updates")
        final_checkpoint = run_directory / "step-300000" / "exact-resume.ckpt"
        item.update(
            {
                "status": "COMPLETED",
                "completed_at": utc_now(),
                "native_transitions": runner.transition_count,
                "complete_update_cycles": runner.update_cycle_count,
                "final_checkpoint_path": str(final_checkpoint.relative_to(study_root)),
                "final_checkpoint_sha256": file_sha256(final_checkpoint),
                "training_track_stream": runner.track_stream.provenance,
            }
        )
        item["attempts"][-1].update(
            {
                "status": "COMPLETED",
                "completed_at": utc_now(),
                "native_transitions": runner.transition_count,
                "complete_update_cycles": runner.update_cycle_count,
            }
        )
        atomic_json(run_directory / "training-outcomes.json", runner.outcomes_summary())
        atomic_json(
            run_directory / "gcsl-diagnostics.json",
            {
                "summary": {
                    **runner.diagnostics.summary(),
                    "first_canonical_target_available_transition": (
                        runner.first_canonical_target_available_transition
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
        _refresh_totals(manifest)
        update_manifest(study_root, manifest)
        raise
    finally:
        runner.close()
    _refresh_totals(manifest)
    update_manifest(study_root, manifest)


def run_study(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    provenance = verify_preflight(root, require_training_authorization=True)
    study_root, manifest = initialize_study_root(root / DEFAULT_OUTPUT_ROOT, provenance)
    with ExclusiveStudyLock(study_root):
        manifest["status"] = "RUNNING"
        update_manifest(study_root, manifest)
        try:
            for seed in PLANNED_SEEDS:
                execute_policy(seed=seed, study_root=study_root, manifest=manifest)
            manifest["status"] = "COMPLETED"
            manifest["completed_at"] = utc_now()
            update_manifest(study_root, manifest)
        except BaseException as error:
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
    root = Path(repo_root).resolve()
    provenance = verify_preflight(root, require_training_authorization=True)
    study_root = root / DEFAULT_OUTPUT_ROOT
    manifest_path = study_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_resume_provenance(manifest, provenance)
    interrupted = [
        item
        for item in manifest["policies"].values()
        if item["status"] == "INTERRUPTED"
    ]
    if manifest["status"] != "INTERRUPTED" or len(interrupted) != 1:
        raise RuntimeError("exact resume requires one preserved interrupted policy")
    target = interrupted[0]
    checkpoint = study_root / target["interruption_checkpoint_path"]
    with ExclusiveStudyLock(study_root):
        manifest["status"] = "RUNNING"
        execute_policy(
            seed=target["training_seed"],
            study_root=study_root,
            manifest=manifest,
            resume_checkpoint=checkpoint,
        )
        for seed in PLANNED_SEEDS:
            if manifest["policies"][policy_key(seed)]["status"] == "NOT_STARTED":
                execute_policy(seed=seed, study_root=study_root, manifest=manifest)
        manifest["status"] = "COMPLETED"
        manifest["completed_at"] = utc_now()
        update_manifest(study_root, manifest)
    return manifest_path


def validate_study(repo_root: str | Path) -> Path:
    root = Path(repo_root).resolve()
    provenance = verify_preflight(root, require_training_authorization=True)
    study_root = root / DEFAULT_OUTPUT_ROOT
    manifest_path = study_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_resume_provenance(manifest, provenance)
    validate_validation_gate(study_root, manifest)
    episodes = []
    with ExclusiveStudyLock(study_root):
        manifest["validation_opened"] = True
        manifest["validation"] = {"status": "RUNNING", "opened_at": utc_now()}
        update_manifest(study_root, manifest)
        for seed in PLANNED_SEEDS:
            item = manifest["policies"][policy_key(seed)]
            runner = PolicyRunner.restore(
                study_root / item["final_checkpoint_path"],
                configuration_sha256=provenance["configuration_sha256"],
                scientific_source_sha256=provenance["scientific_source_sha256"],
                execution_source_sha256=provenance["execution_source_sha256"],
            )
            try:
                result = evaluate_policy(
                    runner.learner, VALIDATION_TRACKS, split="validation"
                )
            finally:
                runner.close()
            for episode in result["episodes"]:
                episodes.append({"training_seed": seed, **episode})
            atomic_json(study_root / f"seed-{seed}" / "validation-result.json", result)
        if len(episodes) != 27:
            raise RuntimeError("Validation must contain exactly 3 x 9 episodes")
        atomic_json(study_root / "validation-episodes.json", episodes)
        manifest["validation"] = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "episode_count": 27,
            "pooled_descriptive_successes": sum(
                episode["canonical_success"] for episode in episodes
            ),
        }
        update_manifest(study_root, manifest)
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "resume", "validate"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command == "preflight":
        print(
            json.dumps(
                verify_preflight(
                    root,
                    require_clean=False,
                    require_training_authorization=False,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    operation = {"run": run_study, "resume": resume_study, "validate": validate_study}[
        args.command
    ]
    print(operation(root), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
