"""Execute the preregistered goal-conditioned SAC versus SAC+HER study."""

from __future__ import annotations

import argparse
import json
import os
import random
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.logger import KVWriter, Logger

from .config import (
    DEFAULT_CONFIG_PATH,
    configuration_sha256,
    load_configuration,
)
from .diagnostics import DiagnosticGoalReplayBuffer
from .evaluation import evaluate_model
from .experiment import make_environment, sac_model_kwargs
from .replay_buffer import GoalReplayBuffer

PREPARATION_COMMIT = "f74eb1120dbdfec5107e511db9db79d035c75cd0"
EXPECTED_CONFIGURATION_SHA256 = (
    "87fee0a3f74b9f8365b9f4c5004e3b7bf880a3739d890462780fc1d70294f6ec"
)
DEFAULT_OUTPUT_ROOT = Path("runs/goal-conditioned-her-v1")
SCIENTIFIC_SHA256 = {
    "benchmarks/goal_conditioned/canonical.json": (
        "330b483fece9f707fc132cd516a8b98d57c0e87e490251cb8f38b7bc510311cc"
    ),
    "benchmarks/goal_conditioned/DESIGN.md": (
        "d278bf22dd91a9e3539c8f07202945c2fe33aa51cd131b4777fdbf386f3dddc2"
    ),
    "benchmarks/goal_conditioned/PROTOCOL.md": (
        "3b31a63e4cc71886aceab007156c1f982a71fbad6ec4cb87377d83b8b33a70ec"
    ),
    "benchmarks/goal_conditioned/goal.py": (
        "70743d8e488579eebb30e7c8a8b718f6811e90e8c43658225eac7c4bbcef7875"
    ),
    "benchmarks/goal_conditioned/environment.py": (
        "d91badfe7ebb70c56b3688bf44771b1579cef80620d3f371035516eb3f24f421"
    ),
    "benchmarks/goal_conditioned/replay_buffer.py": (
        "54c72aec91485f0867c64f646b24bb1cbf51e18e2a59b4e527a1144e3b6ef4a3"
    ),
}
LOGGER_KEYS = {
    "rollout/ep_rew_mean",
    "train/actor_loss",
    "train/critic_loss",
    "train/ent_coef",
    "train/ent_coef_loss",
    "train/n_updates",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _verify_frozen_artifacts(repo_root: Path) -> int:
    manifest = repo_root / "benchmarks/llm_reward/frozen_artifacts.sha256"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip(" *")
        if _sha256_file(repo_root / relative) != expected:
            raise RuntimeError(f"Frozen artifact changed: {relative}")
        checked += 1
    return checked


def verify_preflight(configuration, repo_root: str | Path) -> dict[str, Any]:
    """Verify the frozen science and return immutable run provenance."""

    import gymnasium
    import stable_baselines3
    import torch

    import gym_longicontrol

    root = Path(repo_root).resolve()
    if _git(root, "branch", "--show-current") != "research/goal-conditioned-her":
        raise RuntimeError("Runner must execute on research/goal-conditioned-her")
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", PREPARATION_COMMIT, "HEAD"),
        cwd=root,
        check=True,
    )
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("Tracked or untracked working-tree changes block training")
    head = _git(root, "rev-parse", "HEAD")
    if head == PREPARATION_COMMIT:
        raise RuntimeError("Execution runner must be committed before training")
    if configuration_sha256(configuration) != EXPECTED_CONFIGURATION_SHA256:
        raise RuntimeError("Scientific configuration hash changed")
    actual_scientific_hashes = {
        relative: _sha256_file(root / relative)
        for relative in SCIENTIFIC_SHA256
    }
    if actual_scientific_hashes != SCIENTIFIC_SHA256:
        raise RuntimeError("Frozen Goal/HER scientific sources changed")

    history_path = root / "benchmarks/llm_reward/search_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if history["status"] != "CLOSED" or history["winner"] is None:
        raise RuntimeError("Completed LLM reward search is no longer CLOSED")
    frozen_count = _verify_frozen_artifacts(root)

    actual_versions = {
        "gym_longicontrol": gym_longicontrol.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    expected_versions = asdict(configuration.dependencies)
    if actual_versions != expected_versions:
        raise RuntimeError(
            f"Dependency versions changed: {actual_versions} != {expected_versions}"
        )
    runner_path = Path(__file__).resolve()
    return {
        "preparation_commit": PREPARATION_COMMIT,
        "execution_runner_commit": head,
        "branch": "research/goal-conditioned-her",
        "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "scientific_source_sha256": actual_scientific_hashes,
        "execution_source_sha256": {
            str(runner_path.relative_to(root)): _sha256_file(runner_path),
            "benchmarks/goal_conditioned/diagnostics.py": _sha256_file(
                root / "benchmarks/goal_conditioned/diagnostics.py"
            ),
            "benchmarks/goal_conditioned/evaluation.py": _sha256_file(
                root / "benchmarks/goal_conditioned/evaluation.py"
            ),
        },
        "dependencies": actual_versions,
        "llm_reward_search": {
            "status": history["status"],
            "winner_candidate_id": history["winner"]["candidate_id"],
            "history_sha256": _sha256_file(history_path),
            "frozen_artifact_count_verified": frozen_count,
        },
    }


class DiagnosticWriter(KVWriter):
    def __init__(self):
        self.latest: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []

    def write(
        self,
        key_values: Mapping[str, Any],
        key_excluded: Mapping[str, Any],
        step: int = 0,
    ) -> None:
        del key_excluded
        row: dict[str, Any] = {"training_steps": int(step)}
        for key, value in key_values.items():
            try:
                numeric = float(np.asarray(value).item())
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                row[key] = numeric
                self.latest[key] = numeric
        self.history.append(row)

    def close(self) -> None:
        return None


def _capture_rng_state() -> dict[str, Any]:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": None,
        "torch_mps": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            state["torch_mps"] = torch.mps.get_rng_state().clone()
        except RuntimeError:
            pass
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state["torch_mps"] is not None:
        torch.mps.set_rng_state(state["torch_mps"])


def _evaluate_without_rng_perturbation(model, configuration, seeds):
    state = _capture_rng_state()
    try:
        return evaluate_model(model, configuration, seeds)
    finally:
        _restore_rng_state(state)


def _build_diagnostic_model(configuration, condition_id, environment, seed, device):
    from stable_baselines3 import SAC

    arguments = sac_model_kwargs(configuration, condition_id)
    if arguments["replay_buffer_class"] is not GoalReplayBuffer:
        raise RuntimeError("Frozen replay buffer factory changed")
    arguments["replay_buffer_class"] = DiagnosticGoalReplayBuffer
    arguments["replay_buffer_kwargs"] = {
        **arguments["replay_buffer_kwargs"],
        "diagnostics_enabled": True,
    }
    return SAC(
        env=environment,
        seed=seed,
        device=device,
        verbose=0,
        **arguments,
    )


def _failure_mode(metrics, configuration) -> str:
    failures = []
    if not metrics["completed"]:
        failures.append("incomplete")
    if metrics["travel_time_s"] > configuration.task.max_time_s:
        failures.append("deadline")
    if metrics["max_speed_violation_m_s"] > configuration.task.max_speed_violation_m_s:
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def _model_sha256(path: Path) -> str:
    if not path.exists():
        raise RuntimeError(f"Missing saved model: {path}")
    return _sha256_file(path)


def _policy_key(condition_id: str, seed: int) -> str:
    return f"{condition_id}:seed-{seed}"


def _new_manifest(configuration, provenance, device: str) -> dict[str, Any]:
    order = [
        {"condition_id": condition.condition_id, "training_seed": seed}
        for condition in configuration.conditions
        for seed in configuration.training_seeds
    ]
    return {
        "schema_version": 1,
        "study_id": configuration.name,
        "status": "INITIALIZED",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "validation_opened": False,
        "paper_test_tracks_opened": False,
        "requested_device": device,
        "provenance": provenance,
        "execution_order": order,
        "policies": {
            _policy_key(item["condition_id"], item["training_seed"]): {
                **item,
                "status": "PENDING",
                "training_transitions": 0,
                "gradient_updates": 0,
                "development_checkpoints": [],
                "validation_status": "SEALED",
            }
            for item in order
        },
        "totals": {
            "training_transitions": 0,
            "development_evaluation_transitions": 0,
            "validation_evaluation_transitions": 0,
        },
    }


def initialize_study_root(
    output_root: str | Path,
    configuration,
    provenance: Mapping[str, Any],
    device: str,
) -> tuple[Path, dict[str, Any]]:
    """Atomically reserve a brand-new study root and write its active lock."""

    root = Path(output_root)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            "Goal/HER study root already exists; refusing duplicate/partial "
            f"run: {root}"
        ) from error
    lock_payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": _utc_now(),
    }
    lock_path = root / "ACTIVE.lock"
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(lock_payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    manifest = _new_manifest(configuration, provenance, device)
    _write_json(root / "manifest.json", manifest)
    return root, manifest


def _update_manifest(root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _write_json(root / "manifest.json", manifest)


def validate_validation_gate(root: Path, manifest, configuration) -> None:
    if manifest["validation_opened"]:
        raise RuntimeError("Validation was already opened")
    if manifest["paper_test_tracks_opened"]:
        raise RuntimeError("Paper-test tracks must remain sealed")
    for condition in configuration.conditions:
        for seed in configuration.training_seeds:
            item = manifest["policies"][_policy_key(condition.condition_id, seed)]
            if item["status"] != "TRAINING_COMPLETE":
                raise RuntimeError("Validation is sealed until all six policies finish")
            if item["training_transitions"] != configuration.total_training_steps:
                raise RuntimeError("A policy has an incorrect transition count")
            model_path = root / item["final_model_path"]
            if _model_sha256(model_path) != item["final_model_sha256"]:
                raise RuntimeError("A final model hash changed before Validation")
            if len(item["development_checkpoints"]) != len(
                configuration.development_evaluation_steps
            ):
                raise RuntimeError("Development evaluation is incomplete")
    current_sources = {
        relative: _sha256_file(Path(__file__).parents[2] / relative)
        for relative in SCIENTIFIC_SHA256
    }
    if current_sources != manifest["provenance"]["scientific_source_sha256"]:
        raise RuntimeError("Scientific sources changed before Validation")


class GoalStudyCallback:
    """Factory-backed SB3 callback with post-update checkpoint semantics."""

    def __new__(
        cls,
        *,
        configuration,
        condition_id,
        training_seed,
        model,
        run_directory,
        manifest,
        study_root,
        diagnostics_writer,
        started_at,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.remaining = list(configuration.checkpoint_steps[:-1])
                self.outcomes: list[dict[str, Any]] = []
                self.last_info: dict[str, Any] | None = None
                self.last_episode_end_step = 0
                self.evaluation_time_s = 0.0

            def _record_transition(self) -> None:
                info = self.locals["infos"][0]
                self.last_info = dict(info)
                done = bool(np.asarray(self.locals["dones"]).reshape(-1)[0])
                if not done:
                    return
                if "terminal_observation" not in info:
                    raise RuntimeError("Automatic reset lost terminal_observation")
                terminal = info["terminal_observation"]
                terminal_position = (
                    float(terminal["achieved_goal"][0])
                    * configuration.goal_scales.route_length_m
                )
                if not np.isclose(terminal_position, float(info["position_m"])):
                    raise RuntimeError("Stored terminal observation has wrong position")
                metrics = info["episode_metrics"]
                success = bool(info["goal_success"])
                reward = float(np.asarray(self.locals["rewards"]).reshape(-1)[0])
                if reward != float(success):
                    raise RuntimeError("Real sparse reward and goal success disagree")
                self.outcomes.append(
                    {
                        "episode_index": len(self.outcomes) + 1,
                        "training_step": int(self.num_timesteps),
                        "success": success,
                        "failure_mode": _failure_mode(metrics, configuration),
                        "completed": bool(metrics["completed"]),
                        "deadline_compliant": (
                            float(metrics["travel_time_s"])
                            <= configuration.task.max_time_s
                        ),
                        "speed_compliant": (
                            float(metrics["max_speed_violation_m_s"])
                            <= configuration.task.max_speed_violation_m_s
                        ),
                        "travel_time_s": float(metrics["travel_time_s"]),
                        "energy_kwh": float(metrics["energy_kwh"]),
                        "max_speed_violation_m_s": float(
                            metrics["max_speed_violation_m_s"]
                        ),
                        "integrated_speed_violation_m": float(
                            metrics["integrated_speed_violation_m"]
                        ),
                        "final_position_m": float(info["position_m"]),
                    }
                )
                self.last_episode_end_step = int(self.num_timesteps)

            def _checkpoint(self, milestone: int, *, final: bool) -> None:
                expected_timestep = milestone if final else milestone + 1
                if self.num_timesteps != expected_timestep:
                    raise RuntimeError(
                        f"Checkpoint {milestone} expected callback timestep "
                        f"{expected_timestep}, got {self.num_timesteps}"
                    )
                directory = run_directory / f"step-{milestone:06d}"
                directory.mkdir(parents=True, exist_ok=False)
                model_path = directory / "model.zip"
                actual_counter = model.num_timesteps
                try:
                    model.num_timesteps = milestone
                    model.save(model_path)
                finally:
                    model.num_timesteps = actual_counter
                evaluation_started = time.perf_counter()
                result = _evaluate_without_rng_perturbation(
                    model,
                    configuration,
                    configuration.track_splits.development_calibration,
                )
                self.evaluation_time_s += time.perf_counter() - evaluation_started
                checkpoint_payload = {
                    "schema_version": 1,
                    "condition_id": condition_id,
                    "training_seed": training_seed,
                    "training_transitions": milestone,
                    "gradient_updates": int(getattr(model, "_n_updates", 0)),
                    "checkpoint_boundary": (
                        "after the milestone gradient update; intermediate "
                        "capture occurs before storing transition milestone+1"
                    ),
                    "model_sha256": _model_sha256(model_path),
                    "development": result,
                    "replay_diagnostics": (
                        model.replay_buffer.replay_diagnostics.to_dict()
                    ),
                    "optimizer_diagnostics": {
                        key: value
                        for key, value in diagnostics_writer.latest.items()
                        if key in LOGGER_KEYS
                    },
                    "training_wall_time_s": (
                        time.perf_counter() - started_at - self.evaluation_time_s
                    ),
                }
                _write_json(directory / "development-result.json", checkpoint_payload)
                key = _policy_key(condition_id, training_seed)
                item = manifest["policies"][key]
                item["development_checkpoints"].append(
                    {
                        "training_transitions": milestone,
                        "gradient_updates": checkpoint_payload["gradient_updates"],
                        "model_path": str(model_path.relative_to(study_root)),
                        "model_sha256": checkpoint_payload["model_sha256"],
                        "result_path": str(
                            (directory / "development-result.json").relative_to(
                                study_root
                            )
                        ),
                        "evaluation_simulator_transitions": result["summary"][
                            "evaluation_simulator_transitions"
                        ],
                    }
                )
                item["training_transitions"] = milestone
                item["gradient_updates"] = checkpoint_payload["gradient_updates"]
                manifest["totals"]["development_evaluation_transitions"] += result[
                    "summary"
                ]["evaluation_simulator_transitions"]
                _update_manifest(study_root, manifest)

            def _on_step(self) -> bool:
                self._record_transition()
                if self.remaining and self.num_timesteps == self.remaining[0] + 1:
                    self._checkpoint(self.remaining.pop(0), final=False)
                return True

            def _on_training_end(self) -> None:
                if self.remaining:
                    raise RuntimeError("Training ended before intermediate checkpoints")
                final_step = configuration.total_training_steps
                self._checkpoint(final_step, final=True)
                successes = [row for row in self.outcomes if row["success"]]
                partial = None
                if self.last_episode_end_step < final_step:
                    if self.last_info is None:
                        raise RuntimeError("Missing final partial-episode state")
                    partial = {
                        "transition_count": final_step - self.last_episode_end_step,
                        "position_m": float(self.last_info["position_m"]),
                        "elapsed_time_s": float(self.last_info["elapsed_time_s"]),
                        "max_speed_violation_m_s": float(
                            self.last_info["max_speed_violation_m_s"]
                        ),
                    }
                outcome_payload = {
                    "schema_version": 1,
                    "condition_id": condition_id,
                    "training_seed": training_seed,
                    "training_transitions": final_step,
                    "completed_training_episode_count": len(self.outcomes),
                    "canonical_successful_episode_count": len(successes),
                    "first_canonical_success_step": (
                        successes[0]["training_step"] if successes else None
                    ),
                    "failure_mode_counts": {
                        mode: sum(row["failure_mode"] == mode for row in self.outcomes)
                        for mode in sorted(
                            {row["failure_mode"] for row in self.outcomes}
                        )
                    },
                    "partial_episode_at_training_boundary": partial,
                    "episodes": self.outcomes,
                }
                _write_json(run_directory / "training-outcomes.json", outcome_payload)
                _write_json(
                    run_directory / "replay-diagnostics.json",
                    model.replay_buffer.replay_diagnostics.to_dict(),
                )
                _write_json(
                    run_directory / "optimizer-diagnostics.json",
                    diagnostics_writer.history,
                )

        return _Callback()


def _run_policy(
    *,
    configuration,
    condition_id: str,
    training_seed: int,
    study_root: Path,
    manifest: dict[str, Any],
    device: str,
) -> None:
    key = _policy_key(condition_id, training_seed)
    item = manifest["policies"][key]
    if item["status"] != "PENDING":
        raise RuntimeError(f"Policy is not pending: {key}")
    run_directory = study_root / condition_id / f"training-seed-{training_seed}"
    run_directory.mkdir(parents=True, exist_ok=False)
    item["status"] = "TRAINING"
    item["started_at"] = _utc_now()
    _update_manifest(study_root, manifest)

    environment = make_environment(configuration)
    started_at = time.perf_counter()
    try:
        model = _build_diagnostic_model(
            configuration, condition_id, environment, training_seed, device
        )
        item["resolved_device"] = str(model.device)
        writer = DiagnosticWriter()
        model.set_logger(Logger(folder=None, output_formats=[writer]))
        callback = GoalStudyCallback(
            configuration=configuration,
            condition_id=condition_id,
            training_seed=training_seed,
            model=model,
            run_directory=run_directory,
            manifest=manifest,
            study_root=study_root,
            diagnostics_writer=writer,
            started_at=started_at,
        )
        model.learn(
            total_timesteps=configuration.total_training_steps,
            callback=callback,
            log_interval=1,
            progress_bar=False,
        )
        if model.num_timesteps != configuration.total_training_steps:
            raise RuntimeError("Policy did not consume exactly 300,000 transitions")
        final_directory = run_directory / (
            f"step-{configuration.total_training_steps:06d}"
        )
        final_model = final_directory / "model.zip"
        final_development = final_directory / "development-result.json"
        if not final_model.exists() or not final_development.exists():
            raise RuntimeError("Final checkpoint is incomplete")
        final_payload = json.loads(final_development.read_text(encoding="utf-8"))
        item.update(
            {
                "status": "TRAINING_COMPLETE",
                "completed_at": _utc_now(),
                "training_transitions": model.num_timesteps,
                "gradient_updates": int(getattr(model, "_n_updates", 0)),
                "final_model_path": str(final_model.relative_to(study_root)),
                "final_model_sha256": _model_sha256(final_model),
                "training_outcomes_path": str(
                    (run_directory / "training-outcomes.json").relative_to(study_root)
                ),
                "replay_diagnostics_path": str(
                    (run_directory / "replay-diagnostics.json").relative_to(
                        study_root
                    )
                ),
                "training_wall_time_s": final_payload["training_wall_time_s"],
            }
        )
        manifest["totals"]["training_transitions"] += model.num_timesteps
        _update_manifest(study_root, manifest)
    finally:
        environment.close()


def _run_validation(root: Path, manifest, configuration, device: str) -> None:
    from stable_baselines3 import SAC

    validate_validation_gate(root, manifest, configuration)
    manifest["validation_opened"] = True
    manifest["validation_opened_at"] = _utc_now()
    manifest["status"] = "VALIDATING"
    for item in manifest["policies"].values():
        item["validation_status"] = "PENDING"
    _update_manifest(root, manifest)
    for entry in manifest["execution_order"]:
        key = _policy_key(entry["condition_id"], entry["training_seed"])
        item = manifest["policies"][key]
        result_path = (
            root
            / entry["condition_id"]
            / f"training-seed-{entry['training_seed']}"
            / "validation-result.json"
        )
        if result_path.exists():
            raise FileExistsError(f"Validation result already exists: {result_path}")
        item["validation_status"] = "RUNNING"
        _update_manifest(root, manifest)
        loading_environment = make_environment(configuration)
        try:
            model = SAC.load(
                root / item["final_model_path"],
                env=loading_environment,
                device=device,
            )
            if model.num_timesteps != configuration.total_training_steps:
                raise RuntimeError("Loaded final policy has incorrect timestep count")
            result = _evaluate_without_rng_perturbation(
                model, configuration, configuration.track_splits.validation
            )
        finally:
            loading_environment.close()
        payload = {
            "schema_version": 1,
            "condition_id": entry["condition_id"],
            "training_seed": entry["training_seed"],
            "training_transitions": model.num_timesteps,
            "gradient_updates": item["gradient_updates"],
            "final_model_sha256": item["final_model_sha256"],
            "evaluation_split": "validation-3000-3008-v1",
            **result,
        }
        _write_json(result_path, payload)
        item["validation_status"] = "COMPLETE"
        item["validation_result_path"] = str(result_path.relative_to(root))
        item["validation_episode_count"] = len(result["episodes"])
        manifest["totals"]["validation_evaluation_transitions"] += result[
            "summary"
        ]["evaluation_simulator_transitions"]
        _update_manifest(root, manifest)


def run_study(
    *,
    configuration,
    output_root: str | Path,
    repo_root: str | Path,
    device: str = "auto",
) -> Path:
    provenance = verify_preflight(configuration, repo_root)
    root, manifest = initialize_study_root(
        output_root, configuration, provenance, device
    )
    manifest["status"] = "TRAINING"
    _update_manifest(root, manifest)
    try:
        for condition in configuration.conditions:
            for seed in configuration.training_seeds:
                _run_policy(
                    configuration=configuration,
                    condition_id=condition.condition_id,
                    training_seed=seed,
                    study_root=root,
                    manifest=manifest,
                    device=device,
                )
        manifest["status"] = "TRAINING_COMPLETE"
        _update_manifest(root, manifest)
        _run_validation(root, manifest, configuration, device)
        manifest["status"] = "COMPLETE"
        manifest["completed_at"] = _utc_now()
        _update_manifest(root, manifest)
        (root / "ACTIVE.lock").unlink()
    except BaseException as error:
        manifest["status"] = "INTERRUPTED"
        manifest["interruption"] = {
            "at": _utc_now(),
            "type": type(error).__name__,
            "message": str(error),
        }
        _update_manifest(root, manifest)
        raise
    return root / "manifest.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args(argv)
    configuration = load_configuration(args.config)
    repo_root = Path(__file__).resolve().parents[2]
    print(
        run_study(
            configuration=configuration,
            output_root=args.output_root,
            repo_root=repo_root,
            device=args.device,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
