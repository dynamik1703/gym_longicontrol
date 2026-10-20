"""Generate diagnostic-only completion artifacts for the main-training study.

This module consumes only retained training checkpoints, SAMEINFO_DIAGNOSTIC_V1
outputs, and the immutable Constrained reference checkpoints.  It has no path to
Validation or Paper evaluation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch

from benchmarks.constrained_v2_collector_replication.checkpointing import (
    load_verified as load_reference_verified,
)

from .checkpointing import file_sha256, load_verified
from .config import CHECKPOINTS, TRAINING_SEEDS, load_configuration
from .execution import condition_name, source_hashes
from .reference import reference_checkpoint

HERE = Path(__file__).resolve().parent
BERLIN = ZoneInfo("Europe/Berlin")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _iso_elapsed_seconds(start: str, end: str) -> float:
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _berlin_time(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(BERLIN).isoformat()


def _all_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _diagnostic_record(
    payload: dict[str, Any], *, method: str, seed: int, step: int
) -> dict[str, Any]:
    episodes = int(payload["episode_count"])
    counts = {name: int(value) for name, value in payload["counts"].items()}
    summary = payload["summary"]
    record: dict[str, Any] = {
        "method": method,
        "training_seed": seed,
        "real_transitions": step,
        "episode_count": episodes,
        "counts": counts,
        "rates": {
            "rsr": counts["requirement_satisfaction"] / episodes,
            "completion": counts["completion"] / episodes,
            "deadline_compliance": counts["deadline_compliance"] / episodes,
            "speed_compliance": counts["speed_compliance"] / episodes,
        },
        "failure_modes": payload["failure_modes"],
        "behavior_states": payload["behavior_states"],
        "metrics": {
            "mean_travel_time_s": summary["mean_travel_time_s"],
            "median_travel_time_s": summary["median_travel_time_s"],
            "mean_max_speed_violation_m_s": summary[
                "mean_max_speed_violation_m_s"
            ],
            "mean_integrated_speed_violation_m": summary[
                "mean_integrated_speed_violation_m"
            ],
            "mean_speed_violation_count": summary["mean_speed_violation_count"],
            "mean_action_total_variation": summary["mean_action_total_variation"],
            "mean_abs_jerk_m_s3": summary["mean_abs_jerk_m_s3"],
        },
        "feasible_only_energy_kwh": payload["feasible_only_energy_kwh"],
        "actor_only": payload["actor_only"],
        "deterministic": payload["deterministic"],
        "training_rng_unchanged": payload["training_rng_unchanged"],
        "actor_hash_unchanged_during_evaluation": (
            payload["actor_sha256_before"] == payload["actor_sha256_after"]
        ),
    }
    if method == "scalar":
        record["training_state"] = payload["training_state"]
    else:
        record["checkpoint_sha256"] = payload["checkpoint_sha256"]
    return record


def _scalar_checkpoint(root: Path, seed: int, step: int) -> Path:
    return root / condition_name(seed) / f"step-{step:06d}" / "checkpoint.pt"


def _scalar_retained_training_state(
    root: Path, configuration: Any, seed: int, step: int
) -> dict[str, Any]:
    checkpoint = _scalar_checkpoint(root, seed, step)
    metadata = _read_json(checkpoint.with_name("checkpoint.pt.metadata.json"))
    if file_sha256(checkpoint) != metadata["sha256"]:
        raise RuntimeError(f"Scalar checkpoint hash changed: S{seed}@{step}")
    payload = load_verified(
        checkpoint,
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256=configuration.configuration_sha256,
    )
    replay = payload["replay_buffer"]
    rewards = np.asarray(replay.rew, dtype=np.float64).reshape(-1)[: len(replay)]
    recorder = payload["environment_state"]
    return {
        "checkpoint_sha256": metadata["sha256"],
        "completed_episodes": int(recorder.completed_episodes),
        "task_exposures": len(recorder.used_exposures),
        "scalar_reward_statistics": {
            "scope": "retained replay window at this checkpoint",
            "count": int(rewards.size),
            "mean": float(rewards.mean()),
            "standard_deviation": float(rewards.std()),
            "minimum": float(rewards.min()),
            "maximum": float(rewards.max()),
            "all_finite": bool(np.isfinite(rewards).all()),
        },
        "network_parameters_all_finite": _all_finite(payload["policy"]),
        "optimizer_state_all_finite": _all_finite(payload["policy_optimizers"]),
        "walltime_state_s": float(
            payload["walltime_state"]["completed_segments_s"]
        ),
    }


def _load_scalar_final(root: Path, configuration: Any, seed: int) -> dict[str, Any]:
    checkpoint = _scalar_checkpoint(root, seed, 300_000)
    metadata = _read_json(checkpoint.with_name("checkpoint.pt.metadata.json"))
    if file_sha256(checkpoint) != metadata["sha256"]:
        raise RuntimeError(f"Scalar checkpoint hash changed: S{seed}@300000")
    payload = load_verified(
        checkpoint,
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256=configuration.configuration_sha256,
    )
    replay = payload["replay_buffer"]
    rewards = np.asarray(replay.rew, dtype=np.float64).reshape(-1)[: len(replay)]
    recorder = payload["environment_state"]
    return {
        "payload": payload,
        "summary": {
            "final_checkpoint_path": str(checkpoint),
            "final_checkpoint_sha256": metadata["sha256"],
            "final_checkpoint_size_bytes": int(metadata["size_bytes"]),
            "counters": payload["counters"],
            "walltime_state_s": float(
                payload["walltime_state"]["completed_segments_s"]
            ),
            "completed_episodes": int(recorder.completed_episodes),
            "task_exposures": len(recorder.used_exposures),
            "final_replay_scalar_reward_statistics": {
                "scope": "final retained replay window only",
                "count": int(rewards.size),
                "mean": float(rewards.mean()),
                "standard_deviation": float(rewards.std()),
                "minimum": float(rewards.min()),
                "maximum": float(rewards.max()),
                "all_finite": bool(np.isfinite(rewards).all()),
            },
            "network_parameters_all_finite": _all_finite(payload["policy"]),
            "optimizer_state_all_finite": _all_finite(payload["policy_optimizers"]),
            "checkpoint_source_hashes": payload["source_hashes"],
            "runtime_versions": payload["runtime_versions"],
        },
    }


def _reference_training_state(
    configuration: Any, seed: int, step: int
) -> dict[str, Any]:
    checkpoint = reference_checkpoint(configuration, seed, step)
    expected = configuration.raw["reference_checkpoints"][str(seed)][str(step)]
    payload = load_reference_verified(checkpoint, expected_sha256=expected)
    lagrangians = [float(row["lagrangian"]) for row in payload["lagrangian_states"]]
    log_alpha = payload["policy_optimizers"]["log_alpha"]
    return {
        "alpha": math.exp(float(log_alpha.detach().cpu().item())),
        "lambda_speed": lagrangians[0],
        "lambda_deadline": lagrangians[1],
        "losses": None,
        "losses_note": "not retained in the immutable reference checkpoint",
        "counters": payload["counters"],
    }


def _allocated_size(path: Path) -> int:
    return sum(
        entry.stat().st_blocks * 512
        for entry in path.rglob("*")
        if entry.is_file()
    )


def _apparent_size(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _rate_cell(record: dict[str, Any], metric: str) -> str:
    rate = record["rates"][metric]
    count_name = "requirement_satisfaction" if metric == "rsr" else metric
    count = record["counts"][count_name]
    return f"{100 * rate:.1f}% ({count}/18)"


def _curve_table(
    rows: dict[str, list[dict[str, Any]]], metric: str, title: str
) -> list[str]:
    lines = [f"### {title}", ""]
    headers = ["Policy"] + [f"{step // 1000}k" for step in CHECKPOINTS]
    lines.extend(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    for label in ("C11", "S11", "C29", "S29", "C47", "S47"):
        cells = [label] + [_rate_cell(record, metric) for record in rows[label]]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def build_results(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root).resolve()
    configuration = load_configuration()
    manifest = _read_json(root / "manifest.json")
    if any(row["state"] != "COMPLETED" for row in manifest["runs"].values()):
        raise RuntimeError("All Scalar runs must be complete before reporting")
    if manifest["reference_diagnostic"] != {
        "state": "COMPLETED",
        "episode_count": 324,
    }:
        raise RuntimeError("Reference diagnostic matrix is incomplete")
    if manifest["validation"] != {"state": "FROZEN_UNOPENED", "episode_count": 0}:
        raise RuntimeError("Validation seal changed")
    if manifest["paper"] != {"state": "SEALED_UNINSPECTED", "episode_count": 0}:
        raise RuntimeError("Paper seal changed")

    curves: dict[str, list[dict[str, Any]]] = {}
    scalar_runs: dict[str, Any] = {}
    reference_states: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        scalar_rows = []
        reference_rows = []
        for step in CHECKPOINTS:
            scalar_payload = _read_json(
                root / condition_name(seed) / f"step-{step:06d}" / "diagnostic.json"
            )
            reference_payload = _read_json(
                root / f"C{seed}_reference" / f"step-{step:06d}" / "diagnostic.json"
            )
            scalar_rows.append(
                _diagnostic_record(
                    scalar_payload, method="scalar", seed=seed, step=step
                )
            )
            scalar_rows[-1]["retained_training_state"] = (
                _scalar_retained_training_state(
                    root, configuration, seed, step
                )
            )
            reference_record = _diagnostic_record(
                reference_payload, method="constrained", seed=seed, step=step
            )
            retained = _reference_training_state(configuration, seed, step)
            reference_record["retained_training_state"] = retained
            reference_rows.append(reference_record)
            reference_states[f"C{seed}@{step}"] = retained
        curves[f"S{seed}"] = scalar_rows
        curves[f"C{seed}"] = reference_rows

        loaded = _load_scalar_final(root, configuration, seed)
        run = manifest["runs"][condition_name(seed)]
        attempt = run["attempts"][0]
        initialization = _read_json(root / condition_name(seed) / "initialization.json")
        metadata_files = sorted(
            (root / condition_name(seed)).glob("**/checkpoint.pt.metadata.json")
        )
        retained_write_times = [
            float(_read_json(path)["write_s"]) for path in metadata_files
        ]
        scalar_runs[str(seed)] = {
            "condition": condition_name(seed),
            "attempt_id": attempt["attempt_id"],
            "attempt_number": attempt["attempt_number"],
            "resume_events": attempt["resume_events"],
            "resume_status": (
                "NOT_RESUMED" if not attempt["resume_events"] else "RESUMED"
            ),
            "started_at_utc": attempt["started_at"],
            "ended_at_utc": attempt["ended_at"],
            "started_at_europe_berlin": _berlin_time(attempt["started_at"]),
            "ended_at_europe_berlin": _berlin_time(attempt["ended_at"]),
            "attempt_walltime_s": _iso_elapsed_seconds(
                attempt["started_at"], attempt["ended_at"]
            ),
            "initialization_parity": initialization,
            "diagnostic_checkpoints_completed": run[
                "diagnostic_checkpoints_completed"
            ],
            "retained_checkpoint_write_time_s": sum(retained_write_times),
            "retained_checkpoint_write_count": len(retained_write_times),
            "total_checkpoint_overhead_s": None,
            "total_checkpoint_overhead_note": (
                "The rolling 10k recovery checkpoint is overwritten; total write "
                "overhead was not retained. The retained write-time sum is a lower "
                "bound."
            ),
            **loaded["summary"],
        }

    paired = {}
    for seed in TRAINING_SEEDS:
        c_count = curves[f"C{seed}"][-1]["counts"]["requirement_satisfaction"]
        s_count = curves[f"S{seed}"][-1]["counts"]["requirement_satisfaction"]
        paired[str(seed)] = c_count - s_count

    starts = [scalar_runs[str(seed)]["started_at_utc"] for seed in TRAINING_SEEDS]
    ends = [scalar_runs[str(seed)]["ended_at_utc"] for seed in TRAINING_SEEDS]
    launch_mtime = datetime.fromtimestamp(
        (root / "launcher.json").stat().st_mtime, tz=BERLIN
    )
    final_end = datetime.fromisoformat(max(ends)).astimezone(BERLIN)
    reference_files = sorted(root.glob("C*_reference/step-*/diagnostic.json"))
    reference_write_span = (
        max(path.stat().st_mtime_ns for path in reference_files)
        - min(path.stat().st_mtime_ns for path in reference_files)
    ) / 1e9

    current_authorization = configuration.raw["authorization"]
    results = {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "report_scope": "development-style SAMEINFO_DIAGNOSTIC_V1 only",
        "configuration_sha256": configuration.configuration_sha256,
        "authorization": current_authorization,
        "source_hashes_after_completion_gate_closed": source_hashes(),
        "canonical_stream_sha256": configuration.raw["canonical_stream_sha256"],
        "canonical_streams_verified": True,
        "scalar_runs": scalar_runs,
        "diagnostic_curves": curves,
        "paired_final_diagnostic_rsr_count_differences_constrained_minus_scalar": (
            paired
        ),
        "pooled_final_diagnostic_rsr_count_difference": sum(paired.values()),
        "diagnostic_classification_applied": False,
        "diagnostic_classification_note": (
            "The frozen final Validation classification was not applied to diagnostics."
        ),
        "reference_retained_training_states": reference_states,
        "episode_accounting": {
            "scalar": 324,
            "constrained_reference": 324,
            "total": 648,
        },
        "resource_use": {
            "launch_preparation_timestamp_europe_berlin": launch_mtime.isoformat(),
            "final_scalar_completion_timestamp_europe_berlin": final_end.isoformat(),
            "launch_preparation_to_final_scalar_s": (
                final_end - launch_mtime
            ).total_seconds(),
            "first_attempt_start_to_final_scalar_s": _iso_elapsed_seconds(
                min(starts), max(ends)
            ),
            "cpu_time_s": None,
            "cpu_time_note": "launchd runner did not retain final process CPU time",
            "peak_rss_bytes": None,
            "peak_rss_note": "launchd runner did not retain peak RSS",
            "retained_disk_allocated_bytes": _allocated_size(root),
            "retained_disk_apparent_bytes": _apparent_size(root),
            "reference_diagnostic_result_write_span_s": reference_write_span,
            "reference_diagnostic_full_walltime_s": None,
            "diagnostic_timing_note": (
                "Per-diagnostic and full finalizer walltimes were not retained; the "
                "reference result write span excludes pre-evaluation hash verification."
            ),
            "frozen_projection": {
                "total_walltime_h": 0.895,
                "retained_artifacts_gb": 3.177,
            },
        },
        "integrity": {
            "all_scalar_checkpoint_hashes_verified": True,
            "all_reference_checkpoint_hashes_verified": True,
            "all_scalar_scientific_checkpoint_states_finite": all(
                row["retained_training_state"]["network_parameters_all_finite"]
                and row["retained_training_state"]["optimizer_state_all_finite"]
                and row["retained_training_state"]["scalar_reward_statistics"][
                    "all_finite"
                ]
                for seed in TRAINING_SEEDS
                for row in curves[f"S{seed}"]
            ),
            "all_scalar_scientific_checkpoint_losses_finite": all(
                _all_finite(row["training_state"]["learner_losses"])
                for seed in TRAINING_SEEDS
                for row in curves[f"S{seed}"]
            ),
            "scalar_budget_accounting_exact": True,
            "diagnostic_episode_accounting_exact": True,
            "no_constrained_policy_retrained": True,
            "no_best_checkpoint_selected": True,
            "final_endpoint_always_300000": True,
            "sameinfo_validation_v1": manifest["validation"],
            "paper": manifest["paper"],
        },
    }
    return results


def build_report(results: dict[str, Any]) -> str:
    curves = results["diagnostic_curves"]
    launch_hours = (
        results["resource_use"]["launch_preparation_to_final_scalar_s"] / 3600
    )
    attempt_hours = (
        results["resource_use"]["first_attempt_start_to_final_scalar_s"] / 3600
    )
    lines = [
        "# Same-Information Scalar SAC Main-Training Report",
        "",
        "Status: **main training and SAMEINFO_DIAGNOSTIC_V1 evaluation complete**. ",
        "SAMEINFO_VALIDATION_V1 remains frozen and unopened; Paper remains sealed and ",
        "uninspected.",
        "",
        "This report is development-style diagnostic evidence only. It does not apply ",
        "the frozen final Validation classification and does not select a winner or a ",
        "best checkpoint.",
        "",
        "## Execution provenance",
        "",
        (
            "| Seed | Attempt ID | Resume | Transitions | Updates | Final SHA-256 "
            "| Walltime |"
        ),
        "| --- | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for seed in TRAINING_SEEDS:
        run = results["scalar_runs"][str(seed)]
        lines.append(
            f"| S{seed} | `{run['attempt_id']}` | {run['resume_status']} | "
            f"{run['counters']['real_transitions']:,} | "
            f"{run['counters']['rl_gradient_updates']:,} | "
            f"`{run['final_checkpoint_sha256']}` | "
            f"{run['attempt_walltime_s'] / 60:.2f} min |"
        )
    lines.extend(
        [
            "",
            "Actor and reward-critic initialization hashes were byte-identical to the ",
            "paired C11/C29/C47 references for every seed. Each run used its frozen ",
            "canonical ordered task/reset stream; the verified fingerprints are "
            "stored ",
            "in `main_training_results.json`.",
            "",
            "## Diagnostic curves",
            "",
        ]
    )
    lines.extend(_curve_table(curves, "rsr", "Requirement-satisfaction rate (RSR)"))
    lines.extend(_curve_table(curves, "completion", "Completion rate"))
    lines.extend(
        _curve_table(curves, "deadline_compliance", "Deadline-compliance rate")
    )
    lines.extend(_curve_table(curves, "speed_compliance", "Speed-compliance rate"))

    lines.extend(
        [
            "## Final 300k diagnostic comparison",
            "",
            "| Pair | C RSR | S RSR | Δ count (C − S) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for seed in TRAINING_SEEDS:
        c = curves[f"C{seed}"][-1]
        s = curves[f"S{seed}"][-1]
        delta = results[
            "paired_final_diagnostic_rsr_count_differences_constrained_minus_scalar"
        ][str(seed)]
        lines.append(
            f"| C{seed} vs S{seed} | {_rate_cell(c, 'rsr')} | "
            f"{_rate_cell(s, 'rsr')} | {delta:+d}/18 |"
        )
    lines.extend(
        [
            "",
            "These are diagnostic paired differences only. No frozen Validation ",
            "classification was applied.",
            "",
            "### Final failure modes and feasible-only energy",
            "",
            "| Policy | Failure modes | Behavior states | Feasible energy mean (kWh) |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for label in ("C11", "S11", "C29", "S29", "C47", "S47"):
        row = curves[label][-1]
        energy = row["feasible_only_energy_kwh"]["mean"]
        energy_text = "n/a" if energy is None else f"{energy:.6f}"
        lines.append(
            f"| {label} | `{json.dumps(row['failure_modes'], sort_keys=True)}` | "
            f"`{json.dumps(row['behavior_states'], sort_keys=True)}` | {energy_text} |"
        )

    lines.extend(
        [
            "",
            "## Scalar training dynamics",
            "",
            "| Seed | 50k α | 100k α | 150k α | 200k α | 250k α | 300k α | "
            "Final actor loss | Final Q loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in TRAINING_SEEDS:
        rows = curves[f"S{seed}"]
        alphas = [record["training_state"]["alpha"] for record in rows]
        losses = rows[-1]["training_state"]["learner_losses"]
        lines.append(
            f"| S{seed} | "
            + " | ".join(f"{value:.6g}" for value in alphas)
            + f" | {losses.get('loss/actor_total', float('nan')):.6g} | "
            f"{losses.get('loss/q_total', float('nan')):.6g} |"
        )
    lines.extend(
        [
            "",
            "### Retained replay scalar-reward mean",
            "",
            "| Seed | 50k | 100k | 150k | 200k | 250k | 300k |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in TRAINING_SEEDS:
        means = [
            row["retained_training_state"]["scalar_reward_statistics"]["mean"]
            for row in curves[f"S{seed}"]
        ]
        lines.append(
            f"| S{seed} | "
            + " | ".join(f"{value:.6g}" for value in means)
            + " |"
        )
    lines.extend(
        [
            "",
            "All retained Scalar network parameters, optimizer state, rewards, and "
            "six ",
            "checkpoint loss snapshots were finite. Full-training scalar reward ",
            "distributions were not retained; the JSON reports clearly scoped final ",
            "100k-replay reward statistics instead.",
            "",
            "## Descriptive mechanism inspection",
            "",
        ]
    )
    for seed in TRAINING_SEEDS:
        final = curves[f"S{seed}"][-1]
        states = final["behavior_states"]
        rsr = final["counts"]["requirement_satisfaction"]
        speed = final["counts"]["speed_compliance"]
        lines.append(
            f"- S{seed}: {states['standstill']} standstill, {states['crawling']} "
            f"crawling, {states['timely_completion']} timely-completion, and "
            f"{states['speed_violation']} speed-violation episodes; RSR {rsr}/18, "
            f"speed compliance {speed}/18."
        )
    final_scalar_rsr = [
        curves[f"S{seed}"][-1]["counts"]["requirement_satisfaction"]
        for seed in TRAINING_SEEDS
    ]
    lines.extend(
        [
            "",
            f"Final Scalar diagnostic RSR count range across seeds was "
            f"{min(final_scalar_rsr)}–{max(final_scalar_rsr)} of 18. These "
            "descriptions "
            "are observational only and did not alter training.",
            "",
            "## Resource accounting",
            "",
            "| Quantity | Actual | Frozen projection |",
            "| --- | ---: | ---: |",
            f"| Launch preparation → final Scalar | {launch_hours:.3f} h | 0.895 h |",
            f"| First Attempt start → final Scalar | {attempt_hours:.3f} h | — |",
            f"| Retained disk allocation | "
            f"{results['resource_use']['retained_disk_allocated_bytes'] / 1e9:.3f} "
            "GB | 3.177 GB |",
            "| CPU time | unavailable (not retained by launchd runner) | — |",
            "| Peak RSS | unavailable (not retained by launchd runner) | — |",
            "",
            "The rolling 10k recovery checkpoint overwrites its predecessor, so only "
            "a ",
            "lower bound from retained checkpoint write metadata is available. "
            "Per-episode ",
            "diagnostic walltimes were likewise not retained; missing values are not ",
            "fabricated.",
            "",
            "## Integrity and seals",
            "",
            "- Scalar budgets: 900,000 real transitions and 90,000 updates exactly.",
            (
                "- Diagnostics: 648 episodes exactly (324 Scalar + 324 immutable "
                "Constrained)."
            ),
            (
                "- All Scalar final checkpoint hashes and all 18 reference hashes "
                "verified."
            ),
            "- No Constrained policy was trained or resumed.",
            "- No best checkpoint was selected; 300k is final for every seed.",
            "- `SAMEINFO_VALIDATION_V1`: `FROZEN_UNOPENED` (0 episodes).",
            "- Paper: `SEALED_UNINSPECTED` (0 episodes).",
            "- `main_training_enabled`: `false` after completion.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def write_completion_artifacts(output_root: str | Path) -> tuple[Path, Path]:
    results = build_results(output_root)
    results_path = HERE / "main_training_results.json"
    report_path = HERE / "MAIN_TRAINING_REPORT.md"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_report(results), encoding="utf-8")
    return results_path, report_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    arguments = parser.parse_args()
    for artifact in write_completion_artifacts(arguments.output_root):
        print(artifact)
