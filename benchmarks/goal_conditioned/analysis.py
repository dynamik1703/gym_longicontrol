"""Build the compact, preregistered Goal/HER result artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .evaluation import summarize_episodes

CONDITIONS = ("sac-no-her", "sac-her")
RUN_DIRECTORY = Path("runs/goal-conditioned-her-v1-restart-1")
OUTPUT_DIRECTORY = Path("benchmarks/goal_conditioned")


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, payload: Any) -> Path:
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


def _sum_counts(rows: Iterable[dict[str, Any]], key: str) -> int:
    return sum(int(row[key]) for row in rows)


def _pooled_replay(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count_fields = (
        "total_sampled_rows",
        "real_rows",
        "virtual_rows",
        "eligible_original_target_rows",
        "eligible_virtual_rows",
        "fallback_virtual_rows",
        "relabeled_virtual_rows",
        "positive_real_reward_rows",
        "positive_virtual_reward_rows",
        "virtual_terminal_rows",
        "relabel_added_terminal_rows",
        "zero_virtual_deadline_failure_rows",
        "zero_virtual_violation_failure_rows",
    )
    pooled = {field: _sum_counts(rows, field) for field in count_fields}
    real = pooled["real_rows"]
    virtual = pooled["virtual_rows"]
    total = pooled["total_sampled_rows"]
    pooled["mean_virtual_target_distance_m"] = (
        sum(float(row["virtual_target_distance_m_sum"]) for row in rows) / virtual
        if virtual
        else None
    )
    pooled["rates"] = {
        "real_row_fraction": pooled["real_rows"] / total,
        "virtual_row_fraction": pooled["virtual_rows"] / total,
        "positive_real_reward_rate": (
            pooled["positive_real_reward_rows"] / real if real else None
        ),
        "eligible_virtual_rate": (
            pooled["eligible_virtual_rows"] / virtual if virtual else None
        ),
        "fallback_virtual_rate": (
            pooled["fallback_virtual_rows"] / virtual if virtual else None
        ),
        "relabeled_virtual_rate": (
            pooled["relabeled_virtual_rows"] / virtual if virtual else None
        ),
        "positive_virtual_reward_rate": (
            pooled["positive_virtual_reward_rows"] / virtual if virtual else None
        ),
        "virtual_terminal_rate": (
            pooled["virtual_terminal_rows"] / virtual if virtual else None
        ),
        "relabel_added_terminal_rate": (
            pooled["relabel_added_terminal_rows"] / virtual if virtual else None
        ),
        "zero_virtual_deadline_failure_rate": (
            pooled["zero_virtual_deadline_failure_rows"] / virtual
            if virtual
            else None
        ),
        "zero_virtual_violation_failure_rate": (
            pooled["zero_virtual_violation_failure_rows"] / virtual
            if virtual
            else None
        ),
    }
    pooled["denominators"] = {
        "real_reward_rate": "real_rows",
        "replay_row_fractions": "total_sampled_rows",
        "virtual_row_rates": "virtual_rows",
        "virtual_target_distance": "virtual_rows",
    }
    return pooled


def _pooled_training(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(row["failure_mode_counts"])
    successes = _sum_counts(rows, "canonical_successful_episode_count")
    completed = _sum_counts(rows, "completed_training_episode_count")
    first_steps = [
        int(row["first_canonical_success_step"])
        for row in rows
        if row["first_canonical_success_step"] is not None
    ]
    return {
        "completed_training_episode_count": completed,
        "canonical_successful_episode_count": successes,
        "training_episode_success_rate": successes / completed,
        "first_canonical_success_step": min(first_steps) if first_steps else None,
        "failure_mode_counts": dict(sorted(failures.items())),
    }


def _validate_manifest(manifest, configuration) -> None:
    expected_keys = {
        f"{condition}:seed-{seed}"
        for condition in CONDITIONS
        for seed in configuration.training_seeds
    }
    if manifest["status"] != "COMPLETE":
        raise ValueError("The execution manifest is not complete")
    if not manifest["validation_opened"]:
        raise ValueError("Validation was not opened")
    if manifest["paper_test_tracks_opened"]:
        raise ValueError("Reserved paper-test tracks were opened")
    if manifest["provenance"]["configuration_sha256"] != configuration_sha256(
        configuration
    ):
        raise ValueError("Configuration hash mismatch")
    if set(manifest["policies"]) != expected_keys:
        raise ValueError("The six preregistered policies are not present")
    if manifest["totals"]["training_transitions"] != 1_800_000:
        raise ValueError("Unexpected total training-transition count")
    for policy in manifest["policies"].values():
        if policy["status"] != "TRAINING_COMPLETE":
            raise ValueError("A policy did not finish training")
        if policy["validation_status"] != "COMPLETE":
            raise ValueError("A policy did not finish Validation")
        if policy["training_transitions"] != configuration.total_training_steps:
            raise ValueError("A policy has the wrong training budget")
        if policy["gradient_updates"] != 298_200:
            raise ValueError("A policy has the wrong update count")
        if len(policy["development_checkpoints"]) != 6:
            raise ValueError("A policy is missing Development checkpoints")


def analyze(run_directory: str | Path, configuration) -> tuple[dict, dict, dict]:
    root = Path(run_directory)
    manifest = _read(root / "manifest.json")
    _validate_manifest(manifest, configuration)
    expected_validation = set(configuration.track_splits.validation)
    reserved = set(configuration.track_splits.paper_final_test_reserved)

    validation_episodes: list[dict[str, Any]] = []
    development_curves: list[dict[str, Any]] = []
    final_by_condition: dict[str, Any] = {}
    training_by_condition: dict[str, Any] = {}
    replay_by_condition: dict[str, Any] = {}

    for condition in CONDITIONS:
        condition_episodes = []
        training_rows = []
        replay_rows = []
        final_by_seed = {}
        training_by_seed = {}
        replay_by_seed = {}
        for seed in configuration.training_seeds:
            directory = root / condition / f"training-seed-{seed}"
            validation = _read(directory / "validation-result.json")
            episodes = validation["episodes"]
            observed_tracks = {row["evaluation_seed"] for row in episodes}
            if observed_tracks != expected_validation or len(episodes) != 9:
                raise ValueError(
                    f"Wrong Validation tracks for {condition}, seed {seed}"
                )
            if observed_tracks & reserved:
                raise ValueError("Reserved paper-test track occurs in Validation")
            policy = manifest["policies"][f"{condition}:seed-{seed}"]
            if validation["final_model_sha256"] != policy["final_model_sha256"]:
                raise ValueError("Validation model hash does not match the manifest")
            if validation["training_transitions"] != 300_000:
                raise ValueError("Validation did not use the final model")
            final_by_seed[str(seed)] = validation["summary"]
            condition_episodes.extend(episodes)
            validation_episodes.extend(
                {"condition_id": condition, "training_seed": seed, **row}
                for row in episodes
            )

            outcomes = _read(directory / "training-outcomes.json")
            if outcomes["training_transitions"] != 300_000:
                raise ValueError("Training outcome budget mismatch")
            if outcomes["canonical_successful_episode_count"] != sum(
                row["success"] for row in outcomes["episodes"]
            ):
                raise ValueError("Training success counter mismatch")
            compact_outcomes = {
                key: value
                for key, value in outcomes.items()
                if key != "episodes"
            }
            compact_outcomes["training_episode_success_rate"] = (
                outcomes["canonical_successful_episode_count"]
                / outcomes["completed_training_episode_count"]
            )
            training_by_seed[str(seed)] = compact_outcomes
            training_rows.append(outcomes)

            replay = _read(directory / "replay-diagnostics.json")
            expected_rows = policy["gradient_updates"] * configuration.sac.batch_size
            if replay["total_sampled_rows"] != expected_rows:
                raise ValueError("Replay-row accounting does not match updates")
            replay_by_seed[str(seed)] = replay
            replay_rows.append(replay)

            for step in configuration.development_evaluation_steps:
                checkpoint = _read(
                    directory
                    / f"step-{step:06d}"
                    / "development-result.json"
                )
                development = checkpoint["development"]
                tracks = {row["evaluation_seed"] for row in development["episodes"]}
                if tracks != set(configuration.track_splits.development_calibration):
                    raise ValueError("Wrong Development tracks")
                if tracks & reserved:
                    raise ValueError("Reserved track occurs in Development")
                development_curves.append(
                    {
                        "condition_id": condition,
                        "training_seed": seed,
                        "training_transitions": step,
                        "gradient_updates": checkpoint["gradient_updates"],
                        **development["summary"],
                    }
                )

        final_by_condition[condition] = {
            "by_training_seed": final_by_seed,
            "overall": summarize_episodes(condition_episodes, configuration),
        }
        training_by_condition[condition] = {
            "by_training_seed": training_by_seed,
            "overall": _pooled_training(training_rows),
        }
        replay_by_condition[condition] = {
            "by_training_seed": replay_by_seed,
            "overall": _pooled_replay(replay_rows),
        }

    episode_index = {
        (row["condition_id"], row["training_seed"], row["evaluation_seed"]): row
        for row in validation_episodes
    }
    contingency = Counter()
    for seed in configuration.training_seeds:
        for track in configuration.track_splits.validation:
            no_her = episode_index[("sac-no-her", seed, track)]["feasible"]
            her = episode_index[("sac-her", seed, track)]["feasible"]
            label = (
                "both_succeed"
                if no_her and her
                else "her_alone_succeeds"
                if her
                else "no_her_alone_succeeds"
                if no_her
                else "neither_succeeds"
            )
            contingency[label] += 1

    no_her_rsr = final_by_condition["sac-no-her"]["overall"][
        "requirement_satisfaction_rate"
    ]
    her_rsr = final_by_condition["sac-her"]["overall"][
        "requirement_satisfaction_rate"
    ]
    positive_virtual = replay_by_condition["sac-her"]["overall"][
        "positive_virtual_reward_rows"
    ]
    if positive_virtual <= 0 or her_rsr > no_her_rsr:
        raise ValueError("Observed outcome no longer matches preregistered outcome B")

    results = {
        "schema_version": 1,
        "study_id": configuration.name,
        "configuration_sha256": configuration_sha256(configuration),
        "execution": {
            "status": manifest["status"],
            "created_at": manifest["created_at"],
            "completed_at": manifest["completed_at"],
            "preparation_commit": manifest["provenance"]["preparation_commit"],
            "execution_runner_commit": manifest["provenance"][
                "execution_runner_commit"
            ],
            "authorized_restart_from": manifest["provenance"][
                "authorized_restart_from"
            ],
            "totals": manifest["totals"],
            "final_model_sha256": {
                key: value["final_model_sha256"]
                for key, value in sorted(manifest["policies"].items())
            },
            "validation_opened": manifest["validation_opened"],
            "paper_test_tracks_opened": manifest["paper_test_tracks_opened"],
        },
        "final_validation": final_by_condition,
        "primary_comparison": {
            "estimand": "RSR(sac-her) - RSR(sac-no-her)",
            "sac_no_her_successes": final_by_condition["sac-no-her"]["overall"][
                "success_count"
            ],
            "sac_her_successes": final_by_condition["sac-her"]["overall"][
                "success_count"
            ],
            "episodes_per_condition": 27,
            "rsr_difference": her_rsr - no_her_rsr,
            "paired_seed_track_contingency": {
                key: contingency.get(key, 0)
                for key in (
                    "both_succeed",
                    "her_alone_succeeds",
                    "no_her_alone_succeeds",
                    "neither_succeeds",
                )
            },
            "preregistered_outcome": "B",
            "interpretation": (
                "HER produced positive virtual samples, but canonical Validation "
                "success did not improve."
            ),
        },
        "development_curves": development_curves,
        "training_outcomes": training_by_condition,
        "replay_diagnostics": replay_by_condition,
        "artifact_notes": {
            "representative_trajectory_plot": (
                "Unavailable: the one permitted Validation pass stored compact "
                "episode metrics, not transient trajectories; Validation was not "
                "rerun."
            ),
            "energy_scope": (
                "Energy is descriptive only for feasible episodes and was not "
                "optimized."
            ),
        },
    }
    compact_episodes = {
        "schema_version": 1,
        "evaluation_split": "validation-3000-3008-v1",
        "episode_count": len(validation_episodes),
        "episodes": validation_episodes,
    }
    return results, compact_episodes, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIRECTORY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    results, episodes, manifest = analyze(
        args.run_dir, load_configuration(args.config)
    )
    paths = (
        _write(args.output_dir / "results.json", results),
        _write(args.output_dir / "validation_episodes.json", episodes),
        _write(args.output_dir / "execution_manifest.json", manifest),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
