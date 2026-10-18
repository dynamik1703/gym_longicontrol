"""Materialize the forensic audit evidence without simulator transitions."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.constrained_rl_v2.config import configuration_sha256
from benchmarks.constrained_rl_v2.config import load_configuration as load_v2

from .harness import (
    TRAINING_SEEDS,
    collector_buffer_fixture,
    collector_reset_counts,
    current_track_fingerprints,
    divergence_ladder,
    historical_track_fingerprints,
    initial_policy_fingerprints,
    replay_fixture,
    summarize_track_overlap,
    update_schedule_fixture,
)

ARCHIVE_COMMIT = "56d0bc94341ffcfda9cee29596489977be8b82ea"
AUDIT_START_COMMIT = "ebb4402f1bf5f457b265533a0ef95707c51f2a9b"
CURRENT_TRAINING_COMMIT = "7dc95d9d6d4aee3d6a4183387eb53d9fd8287b75"
FSRL_COMMIT = "e056fc9498d5d037869533da7cf976acf462f918"

SHARED_SCIENTIFIC_PATHS = (
    "benchmarks/constrained_rl/adapter.py",
    "benchmarks/constrained_rl_v2/adapter.py",
    "benchmarks/constrained_rl_v2/canonical.json",
    "benchmarks/constrained_rl_v2/config.py",
    "benchmarks/constrained_rl_v2/costs.py",
    "benchmarks/constrained_rl_v2/experiment.py",
    "benchmarks/scalar_sac/evaluation.py",
    "benchmarks/scalar_sac/experiment.py",
    "src/gym_longicontrol/domain/dynamics.py",
    "src/gym_longicontrol/domain/metrics.py",
    "src/gym_longicontrol/domain/observation.py",
    "src/gym_longicontrol/domain/task.py",
    "src/gym_longicontrol/domain/track.py",
    "src/gym_longicontrol/domain/vehicle.py",
    "src/gym_longicontrol/envs/longicontrol.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob(repository: Path, commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def source_comparison(repository: Path) -> dict[str, Any]:
    rows = {}
    for relative in SHARED_SCIENTIFIC_PATHS:
        archived = hashlib.sha256(
            _git_blob(repository, ARCHIVE_COMMIT, relative)
        ).hexdigest()
        current = sha256_file(repository / relative)
        rows[relative] = {
            "archive_sha256": archived,
            "audit_start_sha256": current,
            "identical": archived == current,
        }
    return {
        "archive_commit": ARCHIVE_COMMIT,
        "audit_start_commit": AUDIT_START_COMMIT,
        "all_identical": all(row["identical"] for row in rows.values()),
        "files": rows,
    }


def dependency_source_hashes() -> dict[str, Any]:
    modules = (
        "fsrl.agent.sac_lag_agent",
        "fsrl.data.fast_collector",
        "fsrl.policy.base_policy",
        "fsrl.policy.lagrangian_base",
        "fsrl.policy.sac_lag",
        "tianshou.data.buffer.base",
        "tianshou.data.collector",
    )
    rows = {}
    for name in modules:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        rows[name] = {"path": str(path), "sha256": sha256_file(path)}
    return rows


def _counts_from_episodes(episodes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "requirement_satisfaction": sum(bool(row["feasible"]) for row in episodes),
        "completion": sum(bool(row["completed"]) for row in episodes),
        "deadline_compliance": sum(
            bool(row["completed"] and row["travel_time_s"] <= 140.0) for row in episodes
        ),
        "speed_compliance": sum(
            float(row["max_speed_violation_m_s"]) == 0.0 for row in episodes
        ),
    }


def historical_evidence(root: Path) -> dict[str, Any]:
    policies = {}
    for seed in TRAINING_SEEDS:
        run = root / f"training-seed-{seed}"
        config_path = run / "run-config.json"
        diagnostics_path = run / "training-diagnostics.json"
        checkpoint_path = run / "target-300000" / "policy.pt"
        validation_path = run / "target-300000" / "validation-3000-3008-result.json"
        config = load_json(config_path)
        diagnostics = load_json(diagnostics_path)
        validation = load_json(validation_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        policies[str(seed)] = {
            "run_config_path": str(config_path),
            "run_config_sha256": sha256_file(config_path),
            "training_diagnostics_path": str(diagnostics_path),
            "training_diagnostics_sha256": sha256_file(diagnostics_path),
            "final_checkpoint_path": str(checkpoint_path),
            "final_checkpoint_sha256": sha256_file(checkpoint_path),
            "final_checkpoint_metadata": checkpoint["metadata"],
            "final_validation_path": str(validation_path),
            "final_validation_sha256": sha256_file(validation_path),
            "validation_counts": _counts_from_episodes(validation["episodes"]),
            "validation_summary": validation["summary"],
            "completed_training_episodes": len(diagnostics),
            "pid_updates": len(diagnostics),
            "first_100_episode_lengths": [
                int(row["episode_steps"]) for row in diagnostics[:100]
            ],
            "final_training_state": diagnostics[-1],
            "recorded_runtime": config["runtime"],
            "recorded_versions": {
                "python": config["runtime"]["python"],
                "fsrl": config["fsrl_version"],
                "fsrl_git_commit": config["fsrl_git_commit"],
                "tianshou": config["tianshou_version"],
                "numpy": "UNKNOWN",
                "torch": "UNKNOWN",
                "gymnasium": "UNKNOWN",
            },
        }
    return {
        "raw_root": str(root),
        "run_producing_commit": "UNKNOWN",
        "run_producing_branch": "UNKNOWN",
        "earliest_archive_commit": ARCHIVE_COMMIT,
        "earliest_archive_branch": "research/task-specification-benchmarks",
        "policies": policies,
    }


def _find_seed_wrapper(environment: Any) -> Any:
    current = environment
    while not hasattr(current, "seeds_used"):
        if not hasattr(current, "env"):
            raise RuntimeError("Checkpoint has no TrainingSeedWrapper")
        current = current.env
    return current


def _current_first_episode_lengths(root: Path, run_id: str) -> list[int]:
    terminal_transition_ids: set[int] = set()
    for step in (100_000, 150_000):
        checkpoint = root / run_id / f"step-{step:06d}" / "checkpoint.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        buffer = payload["real_replay"]
        transition_ids = np.asarray(buffer.info.real_transition_id).reshape(-1)
        done = np.asarray(buffer.done).reshape(-1)
        terminal_transition_ids.update(
            int(transition_id)
            for transition_id, is_done in zip(transition_ids, done, strict=True)
            if is_done
        )
        del payload, buffer
    ordered = sorted(terminal_transition_ids)
    lengths = np.diff(np.asarray([-1, *ordered], dtype=np.int64)).tolist()
    if len(lengths) < 100:
        raise RuntimeError("Retained checkpoints do not cover 100 current episodes")
    return [int(value) for value in lengths[:100]]


def current_evidence(root: Path, repository: Path) -> dict[str, Any]:
    results = load_json(repository / "benchmarks/multistep_physics_mbrl/results.json")
    main = load_json(
        repository / "benchmarks/multistep_physics_mbrl/main_training_results.json"
    )
    manifest = load_json(root / "manifest.json")
    main_rows = {row["run"]: row for row in main["policies"]}
    policies = {}
    checkpoint_source_hashes = []
    for seed in TRAINING_SEEDS:
        run_id = f"matched_real_only-seed-{seed}"
        checkpoint_path = root / run_id / "step-300000" / "checkpoint.pt"
        first_episode_lengths = _current_first_episode_lengths(root, run_id)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        wrapper = _find_seed_wrapper(payload["environment_state"])
        validation = results["validation"]["per_policy"][run_id]
        stability = main_rows[run_id]["stability"]
        checkpoint_source_hashes.append(payload["source_hashes"])
        policies[str(seed)] = {
            "run_id": run_id,
            "final_checkpoint_path": str(checkpoint_path),
            "final_checkpoint_sha256": sha256_file(checkpoint_path),
            "final_checkpoint_metadata": load_json(
                checkpoint_path.with_name("checkpoint.pt.metadata.json")
            ),
            "configuration_sha256": payload["configuration_sha256"],
            "counters": payload["counters"],
            "pid_updates": payload["pid_real_episode_updates"],
            "first_100_episode_lengths": first_episode_lengths,
            "reset_seed_count": len(wrapper.seeds_used),
            "first_100_observed_reset_seeds": list(wrapper.seeds_used[:100]),
            "runtime": payload["runtime_versions"],
            "validation_counts": validation["metrics"]["counts"],
            "validation_raw_result_sha256": validation["raw_result_sha256"],
            "validation_actor_unchanged": validation["actor_unchanged"],
            "final_training_state": {
                "alpha": stability["alpha"][-1],
                "lagrange_speed": stability["lambda_speed"][-1],
                "lagrange_deadline": stability["lambda_deadline"][-1],
            },
            "attempt": manifest["runs"][run_id]["attempts"][0],
        }
        del payload
    return {
        "raw_root": str(root),
        "training_authorization_commit": CURRENT_TRAINING_COMMIT,
        "completed_training_commit": results["completed_main_training_commit"],
        "validation_commit": AUDIT_START_COMMIT,
        "training_branch": "research/multistep-physics-mbrl",
        "recorded_training_threads": 4,
        "checkpoint_source_hashes_identical_across_seeds": (
            checkpoint_source_hashes[0]
            == checkpoint_source_hashes[1]
            == checkpoint_source_hashes[2]
        ),
        "checkpoint_source_hashes": checkpoint_source_hashes[0],
        "policies": policies,
    }


def rng_evidence(
    historical_provenance: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    by_seed = {}
    for seed in TRAINING_SEEDS:
        historical_tracks = historical_track_fingerprints(seed, count=100)
        recent = current_track_fingerprints(seed, count=100)
        generated_seeds = [row["explicit_reset_seed"] for row in recent]
        observed_seeds = current["policies"][str(seed)][
            "first_100_observed_reset_seeds"
        ]
        historical_lengths = historical_provenance["policies"][str(seed)][
            "first_100_episode_lengths"
        ]
        current_lengths = current["policies"][str(seed)]["first_100_episode_lengths"]
        by_seed[str(seed)] = {
            "historical_reconstruction_scope": (
                "source-exact but runtime-contingent reset-only reconstruction"
            ),
            "historical_first_100_used_track_fingerprints": historical_tracks,
            "current_first_100_used_track_fingerprints": recent,
            "current_generated_seeds_match_checkpoint": (
                generated_seeds == observed_seeds
            ),
            "historical_first_100_episode_lengths": historical_lengths,
            "current_first_100_episode_lengths": current_lengths,
            "episode_length_comparison": {
                "same_at_same_index": sum(
                    left == right
                    for left, right in zip(
                        historical_lengths, current_lengths, strict=True
                    )
                ),
                "first_divergence_index": next(
                    index
                    for index, (left, right) in enumerate(
                        zip(historical_lengths, current_lengths, strict=True)
                    )
                    if left != right
                ),
            },
            "overlap": summarize_track_overlap(historical_tracks, recent),
        }
    return {
        "global_seeding": {
            "both_agent_constructors": (
                "FSRL seed_all seeds Python random, legacy NumPy global RNG, "
                "Torch CPU/CUDA, and sets cuDNN flags"
            ),
            "historical_environment": (
                "first reset seed=training_seed, later reset seed=None; one "
                "continuous Gymnasium np_random stream"
            ),
            "current_environment": (
                "dedicated NumPy Generator seeded by SeedSequence([training_seed, "
                "0x4D564531]); each reset re-seeds the base environment"
            ),
            "replay_sampling": "Tianshou ReplayBuffer uses legacy np.random.choice",
            "actor_sampling": "Torch distribution sampling from global Torch RNG",
        },
        "collector_reset_fixture": collector_reset_counts(),
        "by_seed": by_seed,
    }


def config_evidence() -> dict[str, Any]:
    v2 = load_v2()
    return {
        "configuration_sha256": configuration_sha256(v2),
        "canonical_configuration": load_json(
            Path(__file__).parents[1] / "constrained_rl_v2" / "canonical.json"
        ),
        "scientific_parameter_differences": [],
        "current_mve_frozen_reference": {
            "configuration_sha256": configuration_sha256(v2),
            "fsrl_git_commit": FSRL_COMMIT,
            "tianshou_version": "0.5.1",
            "n_step": 2,
            "batch_size": 256,
            "update_per_step": 0.1,
        },
        "conclusion": (
            "No task, reward, cost, limit, network, optimizer, replay capacity, "
            "n-step, gamma, tau, entropy, PID, seed-list, or Validation-split "
            "parameter mismatch was found."
        ),
    }


def build_results(
    repository: Path, historical_root: Path, current_root: Path
) -> dict[str, Any]:
    historical = historical_evidence(historical_root)
    current = current_evidence(current_root, repository)
    initial = {str(seed): initial_policy_fingerprints(seed) for seed in TRAINING_SEEDS}
    return {
        "schema_version": 1,
        "audit": "constrained_v2_reproducibility",
        "audit_start_commit": AUDIT_START_COMMIT,
        "diagnostic_only": True,
        "longicontrol_simulator_transitions_consumed": 0,
        "new_scientific_training_runs": 0,
        "reserved_split_evaluations": {"development": 0, "validation": 0, "paper": 0},
        "headline": {
            "historical_validation_rsr": [21, 27],
            "current_matched_real_only_validation_rsr": [10, 27],
            "primary_classification": "C",
            "primary_label": "RNG/trajectory mismatch",
            "confidence_in_primary": "HIGH",
            "confidence_as_complete_causal_explanation_of_11_episode_gap": "MEDIUM",
            "validity": (
                "Both reported evaluations are valid for their recorded policies; "
                "they are not an exact seed-paired reproduction comparison."
            ),
        },
        "classifications": {
            "A": {"support": "REJECTED", "confidence": "HIGH"},
            "B": {"support": "STRONG_SECONDARY", "confidence": "HIGH"},
            "C": {"support": "PRIMARY", "confidence": "HIGH"},
            "D": {"support": "SECONDARY_POSSIBLE", "confidence": "MEDIUM"},
            "E": {"support": "PROVENANCE_LIMITATION", "confidence": "HIGH"},
            "F": {"support": "REJECTED", "confidence": "HIGH"},
        },
        "provenance": {
            "historical": historical,
            "current": current,
            "shared_source_comparison": source_comparison(repository),
            "installed_dependency_source_hashes": dependency_source_hashes(),
        },
        "configuration": config_evidence(),
        "rng_and_tracks": rng_evidence(historical, current),
        "initialization": initial,
        "bounded_end_to_end_fingerprint_ladder": divergence_ladder(audit_seed=0),
        "replay_and_n_step_fixture": replay_fixture(),
        "collector_buffer_fixture": collector_buffer_fixture(),
        "update_schedule_fixture": update_schedule_fixture(),
        "implementation_differences": [
            {
                "area": "training track RNG",
                "historical": "continuous base-environment RNG from nominal seed",
                "current": "dedicated seed stream and per-reset base re-seeding",
                "first_divergence": "episode 0",
            },
            {
                "area": "collector reset consumption",
                "historical": "two resets after each collected episode; one discarded",
                "current": "one reset after terminal transition; next track retained",
                "first_divergence": "after episode 0 even if first track were paired",
            },
            {
                "area": "actor/data collection schedule",
                "historical": "policy fixed for a whole episode",
                "current": "policy may change during an episode after replay warm-up",
                "first_divergence": "transition 256 in a long first episode",
            },
            {
                "area": "learner schedule",
                "historical": "round(0.1 * episode_steps) bulk updates",
                "current": "global floor(real_transitions * 0.1) interleaved updates",
                "observed_final_updates": {
                    "historical": {"11": 29995, "29": 30003, "47": 30003},
                    "current": {"11": 30000, "29": 30000, "47": 30000},
                },
            },
            {
                "area": "PID timing",
                "historical": "PID before all learner updates for the episode",
                "current": (
                    "PID at terminal after preceding interleaved updates and before "
                    "the terminal-step learner update"
                ),
            },
            {
                "area": "runtime",
                "historical": "Torch threads=2; NumPy/Torch/Gymnasium unrecorded",
                "current": "Torch threads=4; full package versions recorded",
            },
        ],
        "evaluation_protocol": {
            "equivalent_scientific_semantics": True,
            "tracks": list(range(3000, 3009)),
            "deterministic_actor": True,
            "normalization": None,
            "action_mapping": "scaled and clipped identically",
            "termination_and_truncation": "shared evaluator/environment code",
            "historical_validation_reuse_note": (
                "Historical training evaluated Validation at every checkpoint; the "
                "current study opened it once at final checkpoints. This is a protocol "
                "difference, not an explanation for deterministic final-policy scores."
            ),
        },
        "within_current_m_zero_parity": {
            "status": "PASS",
            "meaning": (
                "M=0 calls the pinned policy.update on real replay and adds no model "
                "target; this proves within-current branch parity only."
            ),
            "does_not_mean": (
                "It does not reproduce the historical RNG stream, collector cadence, "
                "PID timing, update count, or runtime."
            ),
        },
        "minimal_future_reproduction": {
            "necessary_for_causal_attribution": True,
            "not_needed_to_validate_existing_evaluations": True,
            "proposal": (
                "After recovering or freezing exact dependency builds, run a seed-11 "
                "2x2 at 300k: historical/current reset stream crossed with historical/"
                "current collector-update schedule, with common recorded initial state "
                "and replay-sample fingerprints. Extend to all three seeds only if the "
                "seed-11 attribution is ambiguous."
            ),
        },
        "limitations": [
            "The historical run-producing commit and branch were not recorded.",
            "Historical NumPy, Torch, and Gymnasium versions were not recorded.",
            (
                "Historical initial weights, RNG states, replay, and environment "
                "state were not checkpointed."
            ),
            (
                "Historical first-100 tracks are source-exact current-runtime "
                "reconstructions, not recovered historical bytes."
            ),
            (
                "No new full training was authorized, so RNG and schedule effects "
                "were not causally isolated."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--historical-runs", type=Path, required=True)
    parser.add_argument("--current-runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_results(
        args.repository.resolve(),
        args.historical_runs.resolve(),
        args.current_runs.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
