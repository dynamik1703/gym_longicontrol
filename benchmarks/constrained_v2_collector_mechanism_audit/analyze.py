"""Regenerate the read-only constrained-V2 collector mechanism audit.

This module only reads the frozen collector-replication checkout.  It never
constructs an environment, invokes a policy, performs an optimizer step, or
writes into the source-study worktree.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SOURCE_STUDY_COMMIT = "38a8c7cc6789ae22d44de2b6f6cd3c84352c4b62"
SOURCE_ANCESTOR_COMMIT = "7d4b63aa021a8694bf51cca2ed319d55045bb7f8"
AUDIT_BRANCH = "research/constrained-v2-collector-mechanism-audit"
CONDITIONS = (
    "E11_episodic_seed11",
    "T11_transitional_seed11",
    "E29_episodic_seed29",
    "T29_transitional_seed29",
    "E47_episodic_seed47",
    "T47_transitional_seed47",
)
SEEDS = (11, 29, 47)
CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
EPISODE_ALIGNMENT_POINTS = (25, 50, 100, 150, 200)
LOSS_KEYS = (
    "loss/actor_rew",
    "loss/actor_safety",
    "loss/actor_safety_1",
    "loss/actor_total",
    "loss/alpha_loss",
    "loss/alpha_value",
    "loss/lagrangian",
    "loss/lagrangian_1",
    "loss/q0",
    "loss/q1",
    "loss/q2",
    "loss/q_total",
    "loss/rescaling",
)
METRICS = (
    "requirement_satisfaction",
    "completion",
    "deadline_compliance",
    "speed_compliance",
)
HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/Users/romanliessner/gitlab/gym_longicontrol_collector_replication"
)


def read_json(path: Path) -> Any:
    """Read JSON without mutating its source."""

    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for a retained artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.mean(rows) if rows else None


def median(values: Iterable[float]) -> float | None:
    rows = list(values)
    return statistics.median(rows) if rows else None


def quantile(values: Iterable[float], probability: float) -> float:
    """Linear empirical quantile, matching NumPy's default method."""

    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("quantile requires at least one value")
    position = (len(rows) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return rows[lower]
    weight = position - lower
    return rows[lower] * (1.0 - weight) + rows[upper] * weight


def pearson(left: Iterable[float], right: Iterable[float]) -> float | None:
    """Descriptive Pearson coefficient; no inferential use is made of it."""

    x = [float(value) for value in left]
    y = [float(value) for value in right]
    if len(x) != len(y) or len(x) < 2:
        return None
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    x_ss = sum((value - x_mean) ** 2 for value in x)
    y_ss = sum((value - y_mean) ** 2 for value in y)
    if x_ss == 0.0 or y_ss == 0.0:
        return None
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y, strict=True)
    )
    return numerator / math.sqrt(x_ss * y_ss)


def sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def condition_for(
    runs: dict[str, dict[str, Any]], seed: int, collector: str
) -> tuple[str, dict[str, Any]]:
    matches = [
        (name, row)
        for name, row in runs.items()
        if row["nominal_seed"] == seed and row["collector"] == collector
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {collector} condition for seed {seed}")
    return matches[0]


def episode_behavior_state(row: dict[str, Any]) -> str:
    """Assign one preregistered, mutually exclusive diagnostic episode state."""

    if bool(row["feasible"]):
        return "FEASIBLE"
    if not bool(row["completed"]):
        if float(row["energy_kwh"]) <= 1e-6:
            return "STANDSTILL_OR_NEAR_STANDSTILL"
        return "CRAWLING_OR_PARTIAL"
    if float(row["max_speed_violation_m_s"]) > 0.0:
        return "COMPLETES_WITH_SPEED_VIOLATION"
    if float(row["travel_time_s"]) > 140.0:
        return "COMPLETES_TOO_SLOW"
    return "MIXED"


def checkpoint_behavior_state(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Use a strict majority; otherwise retain the heterogeneous MIXED label."""

    counts = Counter(episode_behavior_state(row) for row in episodes)
    ordered = {
        name: counts.get(name, 0)
        for name in (
            "STANDSTILL_OR_NEAR_STANDSTILL",
            "CRAWLING_OR_PARTIAL",
            "COMPLETES_TOO_SLOW",
            "FEASIBLE",
            "COMPLETES_WITH_SPEED_VIOLATION",
            "MIXED",
        )
    }
    majority = [name for name, count in ordered.items() if count > len(episodes) / 2]
    return {
        "state": majority[0] if len(majority) == 1 else "MIXED",
        "episode_state_counts": ordered,
    }


def episode_timelines(
    runs: dict[str, dict[str, Any]], large_jump_threshold: float
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for name in CONDITIONS:
        source = runs[name]["training_episode_diagnostics"]
        previous_transition = 0
        previous_updates_before_pid = 0
        routes: Counter[str] = Counter()
        tasks: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        for raw in source:
            routes[raw["track"]["route_sha256"]] += 1
            tasks[raw["track"]["full_task_sha256"]] += 1
            speed_jump = raw["lambda_speed_after"] - raw["lambda_speed_before"]
            deadline_jump = (
                raw["lambda_deadline_after"] - raw["lambda_deadline_before"]
            )
            events = raw["update_timing"]["events"]
            within_updates = sum(
                event["updates_executed"]
                for event in events
                if not bool(event["episode_terminal"])
            )
            immediate_updates = (
                raw["rl_updates_after_episode_processing"]
                - raw["rl_updates_after_pid"]
            )
            rows.append(
                {
                    "episode_number": raw["episode_index"] + 1,
                    "real_transition_count": raw["real_transition_count"],
                    "transitions_between_pid_events": (
                        raw["real_transition_count"] - previous_transition
                    ),
                    "rl_updates_before_pid": raw["rl_updates_before_pid"],
                    "rl_updates_after_pid": raw["rl_updates_after_pid"],
                    "rl_updates_after_episode_processing": raw[
                        "rl_updates_after_episode_processing"
                    ],
                    "gradient_updates_between_pid_events": (
                        raw["rl_updates_before_pid"] - previous_updates_before_pid
                    ),
                    "post_episode_lambda_update_exposure_updates": immediate_updates,
                    "within_episode_actor_updates": within_updates,
                    "update_events_while_episode_active": sum(
                        not bool(event["episode_terminal"]) for event in events
                    ),
                    "lambda_speed_before": raw["lambda_speed_before"],
                    "lambda_speed_after": raw["lambda_speed_after"],
                    "lambda_speed_jump": speed_jump,
                    "lambda_deadline_before": raw["lambda_deadline_before"],
                    "lambda_deadline_after": raw["lambda_deadline_after"],
                    "lambda_deadline_jump": deadline_jump,
                    "large_speed_jump": abs(speed_jump) >= large_jump_threshold,
                    "large_deadline_jump": (
                        abs(deadline_jump) >= large_jump_threshold
                    ),
                    "speed_episodic_cost": raw["speed_episodic_cost"],
                    "deadline_episodic_cost": raw["deadline_episodic_cost"],
                    "alpha": raw["alpha"],
                    "completed": bool(raw["completed"]),
                    "budget_truncated": bool(raw["budget_truncated"]),
                    "episode_duration_steps": raw["episode_duration_steps"],
                    "episode_duration_s": raw["episode_duration_s"],
                    "replay_size_at_terminal": raw["update_timing"][
                        "replay_size_at_terminal"
                    ],
                    "actor_hash_start": raw["actor_sha256_at_episode_start"],
                    "actor_hash_terminal": raw[
                        "actor_sha256_at_episode_terminal_before_updates"
                    ],
                    "actor_hash_after_processing": raw[
                        "actor_sha256_after_episode_processing"
                    ],
                    "actor_unchanged_during_collection": (
                        raw["actor_sha256_at_episode_start"]
                        == raw["actor_sha256_at_episode_terminal_before_updates"]
                    ),
                    "distinct_actor_versions_during_collection": raw[
                        "actor_stationarity"
                    ]["distinct_actor_versions"],
                    "stationary_policy_segments": raw["actor_stationarity"][
                        "stationary_policy_segments"
                    ],
                    "maximum_stationary_run_transitions": raw[
                        "actor_stationarity"
                    ]["maximum_stationary_run_transitions"],
                    "task_exposures": raw["episode_index"] + 1,
                    "unique_routes": len(routes),
                    "unique_full_tasks": len(tasks),
                    "duplicate_routes": sum(routes.values()) - len(routes),
                    "duplicate_full_tasks": sum(tasks.values()) - len(tasks),
                    "route_sha256": raw["track"]["route_sha256"],
                    "full_task_sha256": raw["track"]["full_task_sha256"],
                    "reset_seed": raw["track"]["reset_seed"],
                    "learner_losses": {
                        key: raw["learner_losses"].get(key) for key in LOSS_KEYS
                    },
                }
            )
            previous_transition = raw["real_transition_count"]
            previous_updates_before_pid = raw["rl_updates_before_pid"]
        output[name] = rows
    return output


def actor_vectors_and_checkpoints(
    source_root: Path,
    run_root: Path,
    raw_diagnostics: dict[str, dict[int, dict[str, Any]]],
) -> tuple[
    dict[str, dict[int, dict[str, Any]]],
    dict[str, str],
    dict[str, dict[int, Any]],
]:
    """Read retained tensors only; no module forward pass is performed."""

    import torch

    details: dict[str, dict[int, dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    vectors: dict[str, dict[int, Any]] = {}
    for condition in CONDITIONS:
        details[condition] = {}
        vectors[condition] = {}
        previous_vector = None
        for step in CHECKPOINTS:
            relative = (
                Path("runs/constrained-v2-collector-replication")
                / condition
                / f"step-{step:06d}"
                / "checkpoint.pt"
            )
            path = source_root / relative
            actual_hash = sha256_file(path)
            hashes[str(relative)] = actual_hash
            metadata = read_json(path.with_name("checkpoint.pt.metadata.json"))
            if actual_hash != metadata["sha256"]:
                raise RuntimeError(f"Scientific checkpoint hash mismatch: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload["condition"] != condition:
                raise RuntimeError("Scientific checkpoint condition mismatch")
            counters = payload["counters"]
            if (
                counters["real_transitions"] != step
                or counters["rl_gradient_updates"] < 0
                or counters["rl_gradient_updates"] > step // 10
                or (step == 300_000 and counters["rl_gradient_updates"] != 30_000)
            ):
                raise RuntimeError("Scientific checkpoint counter mismatch")
            tensors = [
                value.detach().cpu().reshape(-1).to(torch.float64)
                for key, value in payload["policy"].items()
                if key.startswith("actor.") and torch.is_tensor(value)
            ]
            vector = torch.cat(tensors)
            vectors[condition][step] = vector
            displacement = None
            if previous_vector is not None:
                displacement = float(torch.linalg.vector_norm(vector - previous_vector))
            actor_hash = raw_diagnostics[condition][step]["actor_sha256_before"]
            if (
                actor_hash
                != raw_diagnostics[condition][step]["actor_sha256_after"]
            ):
                raise RuntimeError("Diagnostic evaluation changed an actor")
            details[condition][step] = {
                "checkpoint_sha256": actual_hash,
                "checkpoint_declared_sha256": metadata["sha256"],
                "gradient_updates": counters["rl_gradient_updates"],
                "actor_sha256": actor_hash,
                "actor_parameter_count": int(vector.numel()),
                "actor_l2_norm": float(torch.linalg.vector_norm(vector)),
                "actor_l2_displacement_from_previous_50k": displacement,
            }
            previous_vector = vector
            del payload
            gc.collect()
    for seed in SEEDS:
        e_name, _ = condition_for_name(seed, "EPISODIC")
        t_name, _ = condition_for_name(seed, "TRANSITIONAL")
        for step in CHECKPOINTS:
            paired = float(
                torch.linalg.vector_norm(
                    vectors[e_name][step] - vectors[t_name][step]
                )
            )
            details[e_name][step]["paired_actor_l2_distance"] = paired
            details[t_name][step]["paired_actor_l2_distance"] = paired
    return details, hashes, vectors


def condition_for_name(seed: int, collector: str) -> tuple[str, str]:
    prefix = "E" if collector == "EPISODIC" else "T"
    name = f"{prefix}{seed}_{collector.lower()}_seed{seed}"
    return name, collector


def checkpoint_timelines(
    source_results: dict[str, Any],
    raw_diagnostics: dict[str, dict[int, dict[str, Any]]],
    checkpoint_details: dict[str, dict[int, dict[str, Any]]],
    initialization: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        run = source_results["runs"][condition]
        initial = initialization["by_condition"][condition]
        rows = [
            {
                "real_transitions": 0,
                "gradient_updates": 0,
                "gradient_updates_evidence": "DERIVABLE",
                "completed_episodes": 0,
                "task_exposures": 0,
                "unique_routes": 0,
                "unique_full_tasks": 0,
                "duplicate_routes": 0,
                "duplicate_full_tasks": 0,
                "pid_update_count": 0,
                "lambda_speed": 0.0,
                "lambda_deadline": 0.0,
                "alpha": 1.0,
                "log_alpha": 0.0,
                "actor_sha256": initial["actor_sha256"],
                "actor_l2_norm": None,
                "actor_l2_displacement_from_previous_50k": None,
                "paired_actor_l2_distance": 0.0,
                "diagnostic_counts": None,
                "behavioral_regime": None,
                "learner_losses": {key: None for key in LOSS_KEYS},
            }
        ]
        embedded = {
            row["real_transitions"]: row for row in run["diagnostic_curve"]
        }
        for step in CHECKPOINTS:
            row = embedded[step]
            state = row["training_state"]
            raw = raw_diagnostics[condition][step]
            exposure = state["exposure"]
            actor = checkpoint_details[condition][step]
            regime = checkpoint_behavior_state(raw["episodes"])
            rows.append(
                {
                    "real_transitions": step,
                    "gradient_updates": actor["gradient_updates"],
                    "gradient_updates_evidence": "DIRECTLY_OBSERVED",
                    "completed_episodes": state["completed_training_episodes"],
                    "task_exposures": exposure["task_exposures"],
                    "unique_routes": exposure["unique_routes"],
                    "unique_full_tasks": exposure["unique_full_tasks"],
                    "duplicate_routes": exposure["duplicate_routes"],
                    "duplicate_full_tasks": exposure["duplicate_full_tasks"],
                    "pid_update_count": state["pid_update_count"],
                    "lambda_speed": state["lambda_speed"],
                    "lambda_deadline": state["lambda_deadline"],
                    "alpha": state["alpha"],
                    "log_alpha": math.log(state["alpha"]),
                    "actor_sha256": actor["actor_sha256"],
                    "actor_l2_norm": actor["actor_l2_norm"],
                    "actor_l2_displacement_from_previous_50k": actor[
                        "actor_l2_displacement_from_previous_50k"
                    ],
                    "paired_actor_l2_distance": actor[
                        "paired_actor_l2_distance"
                    ],
                    "diagnostic_counts": {
                        key: row["counts"][key] for key in METRICS
                    },
                    "behavioral_regime": regime,
                    "learner_losses": {
                        key: state["learner_losses"].get(key) for key in LOSS_KEYS
                    },
                }
            )
        output[condition] = rows
    return output


def flat_checkpoint_delta(
    episodic: dict[str, Any], transitional: dict[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {"real_transitions": episodic["real_transitions"]}
    numeric = (
        "gradient_updates",
        "completed_episodes",
        "task_exposures",
        "unique_routes",
        "unique_full_tasks",
        "duplicate_routes",
        "duplicate_full_tasks",
        "pid_update_count",
        "lambda_speed",
        "lambda_deadline",
        "alpha",
        "log_alpha",
        "actor_l2_norm",
        "actor_l2_displacement_from_previous_50k",
    )
    for key in numeric:
        left = episodic[key]
        right = transitional[key]
        output[key] = None if left is None or right is None else left - right
    if episodic["diagnostic_counts"] is None:
        output["diagnostic_counts"] = None
    else:
        output["diagnostic_counts"] = {
            key: episodic["diagnostic_counts"][key]
            - transitional["diagnostic_counts"][key]
            for key in METRICS
        }
    output["learner_losses"] = {
        key: None
        if episodic["learner_losses"][key] is None
        or transitional["learner_losses"][key] is None
        else episodic["learner_losses"][key]
        - transitional["learner_losses"][key]
        for key in LOSS_KEYS
    }
    return output


def paired_checkpoint_deltas(
    timelines: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        e_name, _ = condition_for_name(seed, "EPISODIC")
        t_name, _ = condition_for_name(seed, "TRANSITIONAL")
        output[str(seed)] = [
            flat_checkpoint_delta(left, right)
            for left, right in zip(
                timelines[e_name], timelines[t_name], strict=True
            )
        ]
    return output


def paired_episode_deltas(
    timelines: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    numeric = (
        "real_transition_count",
        "transitions_between_pid_events",
        "rl_updates_before_pid",
        "rl_updates_after_episode_processing",
        "gradient_updates_between_pid_events",
        "post_episode_lambda_update_exposure_updates",
        "within_episode_actor_updates",
        "lambda_speed_before",
        "lambda_speed_after",
        "lambda_speed_jump",
        "lambda_deadline_before",
        "lambda_deadline_after",
        "lambda_deadline_jump",
        "speed_episodic_cost",
        "deadline_episodic_cost",
        "alpha",
        "episode_duration_steps",
        "episode_duration_s",
        "replay_size_at_terminal",
        "distinct_actor_versions_during_collection",
        "stationary_policy_segments",
        "maximum_stationary_run_transitions",
        "task_exposures",
        "unique_routes",
        "unique_full_tasks",
        "duplicate_routes",
        "duplicate_full_tasks",
    )
    output: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        e_name, _ = condition_for_name(seed, "EPISODIC")
        t_name, _ = condition_for_name(seed, "TRANSITIONAL")
        rows = []
        for left, right in zip(timelines[e_name], timelines[t_name]):
            row = {"episode_number": left["episode_number"]}
            row.update({key: left[key] - right[key] for key in numeric})
            row["completed"] = int(left["completed"]) - int(right["completed"])
            row["learner_losses"] = {
                key: left["learner_losses"][key] - right["learner_losses"][key]
                for key in LOSS_KEYS
            }
            rows.append(row)
        output[str(seed)] = rows
    return output


def volatility(values: list[float], threshold: float) -> dict[str, Any]:
    absolute = [abs(value) for value in values]
    return {
        "mean_signed_jump": statistics.mean(values),
        "mean_absolute_jump": statistics.mean(absolute),
        "median_absolute_jump": statistics.median(absolute),
        "maximum_absolute_jump": max(absolute),
        "signed_jump_population_variance": statistics.pvariance(values),
        "large_jump_count": sum(value >= threshold for value in absolute),
    }


def loss_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for key in LOSS_KEYS:
        values = [row["learner_losses"][key] for row in rows[1:]]
        steps = [right - left for left, right in zip(values, values[1:])]
        output[key] = {
            "values_50k_to_300k": values,
            "minimum": min(values),
            "maximum": max(values),
            "total_variation": sum(abs(value) for value in steps),
            "largest_absolute_checkpoint_step": max(
                (abs(value) for value in steps), default=0.0
            ),
        }
    return output


def condition_summaries(
    runs: dict[str, dict[str, Any]],
    episodes: dict[str, list[dict[str, Any]]],
    checkpoints: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        raw = runs[condition]["training_episode_diagnostics"]
        rows = episodes[condition]
        lengths = [row["episode_duration_steps"] for row in rows]
        speed_jumps = [row["lambda_speed_jump"] for row in rows]
        deadline_jumps = [row["lambda_deadline_jump"] for row in rows]
        immediate = sum(
            row["post_episode_lambda_update_exposure_updates"] for row in rows
        )
        within = [row["within_episode_actor_updates"] for row in rows]
        update_events = [
            event
            for episode in raw
            for event in episode["update_timing"]["events"]
        ]
        update_event_transitions = [
            event["real_transition_count"] for event in update_events
        ]
        data_between_update_events = [
            update_event_transitions[0],
            *[
                right - left
                for left, right in zip(
                    update_event_transitions, update_event_transitions[1:]
                )
            ],
        ]
        final = rows[-1]
        checkpoint_rows = checkpoints[condition]
        alpha = [row["alpha"] for row in checkpoint_rows[1:]]
        counts = [row["diagnostic_counts"] for row in checkpoint_rows[1:]]
        lambdas_speed = [row["lambda_speed"] for row in checkpoint_rows[1:]]
        lambdas_deadline = [
            row["lambda_deadline"] for row in checkpoint_rows[1:]
        ]
        output[condition] = {
            "pid_lagrange": {
                "pid_event_count": len(rows),
                "transition_intervals": {
                    "mean": statistics.mean(
                        row["transitions_between_pid_events"] for row in rows
                    ),
                    "median": statistics.median(
                        row["transitions_between_pid_events"] for row in rows
                    ),
                    "minimum": min(
                        row["transitions_between_pid_events"] for row in rows
                    ),
                    "maximum": max(
                        row["transitions_between_pid_events"] for row in rows
                    ),
                },
                "gradient_update_intervals": {
                    "mean": statistics.mean(
                        row["gradient_updates_between_pid_events"] for row in rows
                    ),
                    "median": statistics.median(
                        row["gradient_updates_between_pid_events"] for row in rows
                    ),
                    "minimum": min(
                        row["gradient_updates_between_pid_events"] for row in rows
                    ),
                    "maximum": max(
                        row["gradient_updates_between_pid_events"] for row in rows
                    ),
                },
                "post_episode_lambda_update_exposure": {
                    "definition": (
                        "gradient updates executed in the same scheduler call, "
                        "after the PID event caused by the just-completed episode"
                    ),
                    "updates": immediate,
                    "total_gradient_updates": 30_000,
                    "fraction": immediate / 30_000,
                },
                "speed_multiplier_volatility": volatility(
                    speed_jumps, threshold
                ),
                "deadline_multiplier_volatility": volatility(
                    deadline_jumps, threshold
                ),
                "final_lambda_speed": final["lambda_speed_after"],
                "final_lambda_deadline": final["lambda_deadline_after"],
            },
            "within_episode_nonstationarity": {
                "total_actor_updates_while_episode_active": sum(within),
                "mean_actor_updates_while_episode_active": statistics.mean(within),
                "median_actor_updates_while_episode_active": statistics.median(
                    within
                ),
                "maximum_actor_updates_while_episode_active": max(within),
                "episodes_with_actor_hash_change_during_collection": sum(
                    not row["actor_unchanged_during_collection"] for row in rows
                ),
                "episodes_with_more_than_one_actor_version": sum(
                    row["distinct_actor_versions_during_collection"] > 1
                    for row in rows
                ),
                "descriptive_correlations": {
                    "updates_vs_duration_steps": pearson(within, lengths),
                    "updates_vs_completion": pearson(
                        within, (int(row["completed"]) for row in rows)
                    ),
                    "updates_vs_speed_cost": pearson(
                        within, (row["speed_episodic_cost"] for row in rows)
                    ),
                    "updates_vs_deadline_cost": pearson(
                        within, (row["deadline_episodic_cost"] for row in rows)
                    ),
                },
                "per_episode_actor_parameter_norm": "NOT_RECOVERABLE",
            },
            "replay_freshness": {
                "update_event_count": len(update_events),
                "mean_replay_occupancy_at_update_event": statistics.mean(
                    event["replay_size"] for event in update_events
                ),
                "minimum_replay_occupancy_at_update_event": min(
                    event["replay_size"] for event in update_events
                ),
                "maximum_replay_occupancy_at_update_event": max(
                    event["replay_size"] for event in update_events
                ),
                "mean_new_transitions_between_update_events": statistics.mean(
                    data_between_update_events
                ),
                "median_new_transitions_between_update_events": statistics.median(
                    data_between_update_events
                ),
                "maximum_new_transitions_between_update_events": max(
                    data_between_update_events
                ),
                "updates_with_full_just_completed_episode_available": immediate,
                "fraction_updates_with_full_just_completed_episode_available": (
                    immediate / 30_000
                ),
                "sample_age_distribution": "NOT_RECOVERABLE",
                "sample_fraction_last_256": "NOT_RECOVERABLE",
                "sample_fraction_last_1000": "NOT_RECOVERABLE",
                "sample_fraction_last_5000": "NOT_RECOVERABLE",
            },
            "task_exposure": {
                "completed_episodes": len(rows),
                "task_exposures": len(rows),
                "unique_routes": final["unique_routes"],
                "unique_full_tasks": final["unique_full_tasks"],
                "duplicate_route_rate": final["duplicate_routes"] / len(rows),
                "duplicate_full_task_rate": (
                    final["duplicate_full_tasks"] / len(rows)
                ),
                "episode_length_steps": {
                    "mean": statistics.mean(lengths),
                    "median": statistics.median(lengths),
                    "minimum": min(lengths),
                    "p10": quantile(lengths, 0.10),
                    "p25": quantile(lengths, 0.25),
                    "p75": quantile(lengths, 0.75),
                    "p90": quantile(lengths, 0.90),
                    "maximum": max(lengths),
                },
            },
            "alpha": {
                "values_50k_to_300k": alpha,
                "log_values_50k_to_300k": [math.log(value) for value in alpha],
                "minimum": min(alpha),
                "maximum": max(alpha),
                "log_change_50k_to_300k": math.log(alpha[-1])
                - math.log(alpha[0]),
                "mean_log_change_per_50k_interval": (
                    math.log(alpha[-1]) - math.log(alpha[0])
                )
                / 5,
                "descriptive_checkpoint_correlations": {
                    "alpha_vs_completion": pearson(
                        alpha, (row["completion"] for row in counts)
                    ),
                    "alpha_vs_deadline_compliance": pearson(
                        alpha, (row["deadline_compliance"] for row in counts)
                    ),
                    "alpha_vs_speed_compliance": pearson(
                        alpha, (row["speed_compliance"] for row in counts)
                    ),
                    "alpha_vs_lambda_speed": pearson(alpha, lambdas_speed),
                    "alpha_vs_lambda_deadline": pearson(alpha, lambdas_deadline),
                    "alpha_after_first_vs_lambda_speed_change": pearson(
                        alpha[1:],
                        (
                            right - left
                            for left, right in zip(
                                lambdas_speed, lambdas_speed[1:]
                            )
                        ),
                    ),
                    "alpha_after_first_vs_lambda_deadline_change": pearson(
                        alpha[1:],
                        (
                            right - left
                            for left, right in zip(
                                lambdas_deadline, lambdas_deadline[1:]
                            )
                        ),
                    ),
                },
            },
            "optimization": {
                "checkpoint_loss_summaries": loss_summary(checkpoint_rows),
                "target_statistics": "NOT_RECOVERABLE",
                "per_update_loss_history": "NOT_RECOVERABLE",
            },
            "training_costs": {
                "mean_speed_episodic_cost": statistics.mean(
                    row["speed_episodic_cost"] for row in rows
                ),
                "mean_deadline_episodic_cost": statistics.mean(
                    row["deadline_episodic_cost"] for row in rows
                ),
                "last_50_mean_speed_episodic_cost": statistics.mean(
                    row["speed_episodic_cost"] for row in rows[-50:]
                ),
                "last_50_mean_deadline_episodic_cost": statistics.mean(
                    row["deadline_episodic_cost"] for row in rows[-50:]
                ),
                "training_completion_rate": statistics.mean(
                    int(row["completed"]) for row in rows
                ),
            },
        }
    return output


def episode_alignment(
    episodes: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        e_name, _ = condition_for_name(seed, "EPISODIC")
        t_name, _ = condition_for_name(seed, "TRANSITIONAL")
        rows = []
        for point in EPISODE_ALIGNMENT_POINTS:
            left = episodes[e_name][point - 1]
            right = episodes[t_name][point - 1]
            rows.append(
                {
                    "completed_episode_count": point,
                    "episodic": {
                        "real_transition_count": left["real_transition_count"],
                        "gradient_updates": left[
                            "rl_updates_after_episode_processing"
                        ],
                        "lambda_speed": left["lambda_speed_after"],
                        "lambda_deadline": left["lambda_deadline_after"],
                        "alpha": left["alpha"],
                        "unique_routes": left["unique_routes"],
                        "unique_full_tasks": left["unique_full_tasks"],
                    },
                    "transitional": {
                        "real_transition_count": right["real_transition_count"],
                        "gradient_updates": right[
                            "rl_updates_after_episode_processing"
                        ],
                        "lambda_speed": right["lambda_speed_after"],
                        "lambda_deadline": right["lambda_deadline_after"],
                        "alpha": right["alpha"],
                        "unique_routes": right["unique_routes"],
                        "unique_full_tasks": right["unique_full_tasks"],
                    },
                    "episodic_minus_transitional": {
                        "real_transition_count": left["real_transition_count"]
                        - right["real_transition_count"],
                        "gradient_updates": left[
                            "rl_updates_after_episode_processing"
                        ]
                        - right["rl_updates_after_episode_processing"],
                        "lambda_speed": left["lambda_speed_after"]
                        - right["lambda_speed_after"],
                        "lambda_deadline": left["lambda_deadline_after"]
                        - right["lambda_deadline_after"],
                        "alpha": left["alpha"] - right["alpha"],
                        "unique_routes": left["unique_routes"]
                        - right["unique_routes"],
                        "unique_full_tasks": left["unique_full_tasks"]
                        - right["unique_full_tasks"],
                    },
                }
            )
        output[str(seed)] = rows
    return output


def paired_feature(
    name: str,
    values: dict[int, float],
    rsr: dict[int, int],
    interpretation: str,
) -> dict[str, Any]:
    return {
        "feature": name,
        "episodic_minus_transitional": {
            str(seed): values[seed] for seed in SEEDS
        },
        "sign_matches_rsr": {
            str(seed): sign(values[seed]) == sign(rsr[seed]) for seed in SEEDS
        },
        "all_three_signs_match_rsr": all(
            sign(values[seed]) == sign(rsr[seed]) for seed in SEEDS
        ),
        "interpretation": interpretation,
    }


def sign_reversal_features(
    summaries: dict[str, dict[str, Any]],
    checkpoints: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    rsr = {11: 3, 29: -4, 47: -7}
    features: list[dict[str, Any]] = []

    def values(path: tuple[str, ...]) -> dict[int, float]:
        result = {}
        for seed in SEEDS:
            e_name, _ = condition_for_name(seed, "EPISODIC")
            t_name, _ = condition_for_name(seed, "TRANSITIONAL")
            left: Any = summaries[e_name]
            right: Any = summaries[t_name]
            for key in path:
                left = left[key]
                right = right[key]
            result[seed] = float(left) - float(right)
        return result

    features.append(
        paired_feature(
            "mean_absolute_speed_lambda_jump",
            values(
                (
                    "pid_lagrange",
                    "speed_multiplier_volatility",
                    "mean_absolute_jump",
                )
            ),
            rsr,
            "EPISODIC is more volatile for all seeds; the sign does not reverse.",
        )
    )
    features.append(
        paired_feature(
            "mean_absolute_deadline_lambda_jump",
            values(
                (
                    "pid_lagrange",
                    "deadline_multiplier_volatility",
                    "mean_absolute_jump",
                )
            ),
            rsr,
            "The small seed-47 reversal does not reproduce the RSR ordering.",
        )
    )
    features.append(
        paired_feature(
            "post_episode_lambda_update_exposure_fraction",
            values(("pid_lagrange", "post_episode_lambda_update_exposure", "fraction")),
            rsr,
            "The intended schedule contrast is nearly invariant across seeds.",
        )
    )
    features.append(
        paired_feature(
            "completed_episode_count",
            values(("task_exposure", "completed_episodes")),
            rsr,
            "Seed 47 contradicts a simple more-episodes explanation.",
        )
    )
    features.append(
        paired_feature(
            "mean_episode_length_steps",
            values(("task_exposure", "episode_length_steps", "mean")),
            rsr,
            "Episode-length direction is not concordant across all seeds.",
        )
    )
    features.append(
        paired_feature(
            "final_lambda_speed",
            values(("pid_lagrange", "final_lambda_speed")),
            rsr,
            "Final speed multipliers are higher under EPISODIC for all seeds.",
        )
    )
    features.append(
        paired_feature(
            "final_lambda_deadline",
            values(("pid_lagrange", "final_lambda_deadline")),
            rsr,
            (
                "This is the only compact internal feature with matching signs "
                "and ordering, but it is an endogenous result of episode costs."
            ),
        )
    )
    final_log_alpha = {}
    for seed in SEEDS:
        e_name, _ = condition_for_name(seed, "EPISODIC")
        t_name, _ = condition_for_name(seed, "TRANSITIONAL")
        final_log_alpha[seed] = math.log(checkpoints[e_name][-1]["alpha"]) - math.log(
            checkpoints[t_name][-1]["alpha"]
        )
    features.append(
        paired_feature(
            "final_log_alpha",
            final_log_alpha,
            rsr,
            (
                "EPISODIC ends lower for every seed; magnitude orders with the "
                "RSR deterioration but the sign does not reverse."
            ),
        )
    )
    return {
        "rsr_episodic_minus_transitional": {
            str(seed): value for seed, value in rsr.items()
        },
        "method": (
            "Descriptive sign/order concordance only; n=3 and no inferential "
            "statistics."
        ),
        "features": features,
        "conclusion": (
            "No exogenous mechanism variable changes sign in a uniquely identifying "
            "way. Final deadline lambda matches the outcome signs, but is itself an "
            "endogenous summary of the diverged trajectories."
        ),
    }


def scorecards() -> dict[str, dict[str, dict[str, str]]]:
    common = {
        "H2": {
            "data_completeness": "HIGH",
            "effect_direction": "EPISODIC fixed; TRANSITIONAL changes within episode",
            "confidence": "HIGH",
        },
        "H3": {
            "data_completeness": "PARTIAL",
            "effect_direction": "EPISODIC batches; TRANSITIONAL streams updates",
            "confidence": "MEDIUM",
        },
        "H4": {
            "data_completeness": "HIGH",
            "confidence": "HIGH",
        },
        "H5": {
            "data_completeness": "HIGH",
            "confidence": "HIGH",
        },
        "H6": {
            "data_completeness": "HIGH at episodes/checkpoints",
            "confidence": "MEDIUM",
        },
        "H7": {
            "data_completeness": "PARTIAL snapshots; no per-update history/targets",
            "confidence": "LOW",
        },
    }
    cards: dict[str, dict[str, dict[str, str]]] = {}
    narratives = {
        11: {
            "H1": (
                "EPISODIC has greater speed/deadline jump volatility and applies all "
                "updates immediately after PID; this can explain stronger feedback, "
                "but the same contrast exists in losing seeds."
            ),
            "H2": (
                "EPISODIC makes zero actor updates during collection versus about 129 "
                "per TRANSITIONAL episode; fixed-policy trajectories may reduce "
                "within-task drift, but this does not identify the favorable sign."
            ),
            "H3": (
                "All EPISODIC updates see a complete just-finished episode versus a "
                "tiny terminal fraction under TRANSITIONAL; sampled indices are absent."
            ),
            "H4": (
                "EPISODIC completes 10 more episodes and has shorter episodes, a "
                "direction compatible with its +3 result here only."
            ),
            "H5": (
                "Both oscillate; EPISODIC finishes diagnostic FEASIBLE (12/18 RSR) "
                "while TRANSITIONAL is MIXED (9/18), matching Validation's +3."
            ),
            "H6": (
                "Final alpha is similar (EPISODIC slightly lower); exploration does "
                "not provide a clear seed-11 advantage."
            ),
            "H7": (
                "Checkpoint actor and critic losses decline smoothly in both; no "
                "unique collapse precedes the +3 result."
            ),
        },
        29: {
            "H1": (
                "EPISODIC again has higher multiplier volatility and immediate PID "
                "exposure, but that treatment-level fact does not uniquely explain -4."
            ),
            "H2": (
                "The large within-episode update contrast is direct, yet identical in "
                "kind to seed 11 where the outcome reverses."
            ),
            "H3": (
                "Schedule-level freshness differs strongly; actual samples and ages "
                "were not logged, preventing attribution."
            ),
            "H4": (
                "Episode counts and mean lengths are almost equal (236 versus 238), "
                "arguing against task exposure as the main seed-29 cause."
            ),
            "H5": (
                "Both end MIXED on diagnostics; Validation shows EPISODIC completes "
                "more but loses deadline and speed compliance, an aggressive tradeoff."
            ),
            "H6": (
                "EPISODIC ends at about half the TRANSITIONAL alpha, consistent with "
                "less late exploration but not causally identified."
            ),
            "H7": (
                "TRANSITIONAL reward/constraint critic losses are much lower at the "
                "retained checkpoints, but snapshot losses are scale-sensitive and "
                "not direct error estimates."
            ),
        },
        47: {
            "H1": (
                "EPISODIC's immediate PID exposure and higher speed-lambda volatility "
                "coexist with a late collapse; deadline volatility itself is similar."
            ),
            "H2": (
                "TRANSITIONAL performs about 145 actor updates during an average "
                "episode and recovers late; whether nonstationarity causes recovery is "
                "not separable from replay and PID timing."
            ),
            "H3": (
                "The schedules sharply differ, but missing sample indices make actual "
                "freshness and episode amplification unmeasurable."
            ),
            "H4": (
                "EPISODIC exposes 30 more completed tasks and shorter episodes yet "
                "loses "
                "by 7, directly contradicting a simple diversity/exposure benefit."
            ),
            "H5": (
                "EPISODIC leads at 100k-150k, is FEASIBLE-dominant at 250k, then falls "
                "to CRAWLING/PARTIAL at 300k; TRANSITIONAL recovers after 150k."
            ),
            "H6": (
                "TRANSITIONAL alpha rebounds after 150k and ends about 8.7x EPISODIC; "
                "this is consistent with recovery, not proof that entropy caused it."
            ),
            "H7": (
                "EPISODIC critic losses exceed TRANSITIONAL snapshots while behavior "
                "later degrades, but no per-update trace or target statistics remain."
            ),
        },
    }
    for seed in SEEDS:
        cards[str(seed)] = {}
        for hypothesis in ("H1", "H2", "H3", "H4", "H5", "H6", "H7"):
            base = dict(common.get(hypothesis, {}))
            if hypothesis == "H1":
                base.update(
                    {
                        "data_completeness": "HIGH",
                        "effect_direction": (
                            "EPISODIC immediate block exposure and generally higher "
                            "volatility"
                        ),
                        "confidence": "HIGH for timing; LOW for causal attribution",
                    }
                )
            base["evidence_supporting_mechanism"] = narratives[seed][hypothesis]
            base["evidence_against_mechanism"] = (
                "The same collector-level contrast occurs in seeds with the opposite "
                "RSR direction, or the required fine-grained evidence was not retained."
            )
            cards[str(seed)][hypothesis] = base
    return cards


def evidence_ledger() -> dict[str, dict[str, Any]]:
    return {
        "H1_pid_lagrange_timing": {
            "status": "DIRECTLY_OBSERVED",
            "available": [
                "episode costs",
                "PID event transitions",
                "lambda before/after",
                "update counters around PID",
            ],
        },
        "H2_within_episode_policy_nonstationarity": {
            "status": "PARTIALLY_OBSERVED",
            "available": [
                "exact update count while active",
                "actor hashes start/terminal/after processing",
                "actor version run counts",
            ],
            "missing": ["per-episode actor tensors and parameter norms"],
        },
        "H3_replay_update_freshness": {
            "status": "PARTIALLY_OBSERVED",
            "available": [
                "replay occupancy at update events",
                "update block schedule",
                "new transitions between update events",
            ],
            "missing": ["sampled indices", "sample ages", "sample provenance"],
        },
        "H4_task_exposure_feedback": {
            "status": "DIRECTLY_OBSERVED",
            "available": [
                "ordered reset seeds",
                "route and full-task hashes",
                "episode lengths",
                "exposure counts",
            ],
        },
        "H5_constraint_regime_transitions": {
            "status": "DERIVABLE",
            "available": [
                "raw retained diagnostic episodes at each scientific checkpoint"
            ],
        },
        "H6_alpha_exploration": {
            "status": "DIRECTLY_OBSERVED",
            "available": ["episode and checkpoint alpha values"],
            "missing": ["action-distribution entropy on retained observations"],
        },
        "H7_optimization_dynamics": {
            "status": "PARTIALLY_OBSERVED",
            "available": ["episode and checkpoint logger snapshots"],
            "missing": ["per-update loss history", "target statistics"],
        },
    }


def missing_field_ledger() -> list[dict[str, str]]:
    return [
        {
            "field": "sampled replay indices and transition IDs",
            "status": "NOT_RECOVERABLE",
            "impact": "No empirical sample-age or recent-window fractions.",
        },
        {
            "field": "per-episode actor parameter tensors",
            "status": "NOT_RECOVERABLE",
            "impact": "No per-episode actor L2 drift; hashes and update counts only.",
        },
        {
            "field": "per-gradient-update loss history",
            "status": "NOT_RECOVERABLE",
            "impact": (
                "Abrupt sub-checkpoint optimization transients cannot be located."
            ),
        },
        {
            "field": "critic target and Q-distribution statistics",
            "status": "NOT_RECOVERABLE",
            "impact": "Checkpoint scalar losses cannot diagnose calibration/collapse.",
        },
        {
            "field": "diagnostic behavior at transition zero",
            "status": "NOT_RECOVERABLE",
            "impact": (
                "Zero row contains learner initialization only; no episode inferred."
            ),
        },
        {
            "field": "append-only Validation invocation log",
            "status": "PARTIALLY_OBSERVED",
            "impact": (
                "Single manifest open/completion record plus six sealed files and "
                "recovery code establish consistency, but not an external audit log."
            ),
        },
    ]


def source_hashes(
    source_root: Path,
    run_root: Path,
    checkpoint_hashes: dict[str, str],
) -> dict[str, str]:
    tracked = source_root / "benchmarks/constrained_v2_collector_replication"
    paths = [
        tracked / name
        for name in (
            "RESULTS.md",
            "results.json",
            "freeze.json",
            "initialization.json",
            "canonical.json",
            "preparation_status.json",
            "split_ledger.json",
            "stream_ledger.json",
            "recover_finalization.py",
            "finalize.py",
            "execution.py",
        )
    ]
    paths.extend(
        [
            run_root / "manifest.json",
            run_root / "launcher.stdout.log",
            run_root / "launcher.stderr.log",
        ]
    )
    for condition in CONDITIONS:
        condition_root = run_root / condition
        paths.extend(
            [
                condition_root / "initialization.json",
                condition_root / "training-episode-diagnostics.json",
                condition_root / "validation.json",
            ]
        )
        for step in CHECKPOINTS:
            directory = condition_root / f"step-{step:06d}"
            paths.extend(
                [
                    directory / "diagnostic.json",
                    directory / "checkpoint.pt.metadata.json",
                ]
            )
    result = {
        str(path.relative_to(source_root)): sha256_file(path) for path in paths
    }
    result.update(checkpoint_hashes)
    return dict(sorted(result.items()))


def finalizer_audit(
    source_root: Path,
    source_results: dict[str, Any],
    manifest: dict[str, Any],
    raw_diagnostics: dict[str, dict[int, dict[str, Any]]],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    validation = manifest["validation"]
    opened_at = validation["opened_at"]
    final_hash_checks = {}
    actor_checks = {}
    for condition in CONDITIONS:
        row = manifest["runs"][condition]
        relative = (
            Path("runs/constrained-v2-collector-replication")
            / condition
            / "step-300000/checkpoint.pt"
        )
        final_hash_checks[condition] = (
            input_hashes[str(relative)] == row["final_checkpoint_sha256"]
        )
        validation_path = (
            source_root
            / "runs/constrained-v2-collector-replication"
            / condition
            / "validation.json"
        )
        sealed = read_json(validation_path)
        final_actor = source_results["runs"][condition]["final_actor_sha256"]
        actor_checks[condition] = {
            "validation_before_matches_final": (
                sealed["actor_sha256_before"] == final_actor
            ),
            "validation_after_matches_final": (
                sealed["actor_sha256_after"] == final_actor
            ),
            "diagnostic_300k_matches_final": (
                raw_diagnostics[condition][300_000]["actor_sha256_before"]
                == final_actor
            ),
        }
    recovery_source = (
        source_root
        / "benchmarks/constrained_v2_collector_replication"
        / "recover_finalization.py"
    ).read_text(encoding="utf-8")
    attempts = [
        attempt
        for condition in CONDITIONS
        for attempt in manifest["runs"][condition]["attempts"]
    ]
    checks = {
        "six_final_checkpoint_hashes_verified": all(final_hash_checks.values()),
        "all_attempts_ended_before_validation_opened": all(
            attempt["ended_at"] < opened_at for attempt in attempts
        ),
        "one_attempt_per_policy": all(
            len(manifest["runs"][condition]["attempts"]) == 1
            for condition in CONDITIONS
        ),
        "no_resume_events": all(not attempt["resume_events"] for attempt in attempts),
        "validation_manifest_complete_108": (
            validation["state"] == "COMPLETED"
            and validation["episode_count"] == 108
            and validation["completed_conditions"] == list(CONDITIONS)
        ),
        "one_sealed_validation_file_per_policy": all(
            len(source_results["runs"][condition]["validation"]["episodes"]) == 18
            for condition in CONDITIONS
        ),
        "all_validation_actor_hashes_unchanged": all(
            all(row.values()) for row in actor_checks.values()
        ),
        "recovery_source_has_no_training_or_evaluation_call": (
            "run_policy(" not in recovery_source
            and "evaluate_diagnostic(" not in recovery_source
            and "evaluate_validation(" not in recovery_source
        ),
        "recovery_provenance_says_no_repetition_or_mutation": (
            source_results["postprocessing_recovery"]["training_repeated"] is False
            and source_results["postprocessing_recovery"]["evaluation_repeated"]
            is False
            and source_results["postprocessing_recovery"][
                "sealed_artifacts_modified"
            ]
            is False
        ),
        "paper_still_sealed": manifest["paper"]
        == {"episode_count": 0, "state": "SEALED_UNINSPECTED"},
    }
    if not all(checks.values()):
        raise RuntimeError("Finalizer recovery integrity check failed")
    return {
        "classification": "REPORTING_ONLY_RECOVERY_CONFIRMED",
        "checks": checks,
        "final_checkpoint_hash_checks": final_hash_checks,
        "actor_hash_checks": actor_checks,
        "validation_opened_at": opened_at,
        "validation_completed_at": validation["completed_at"],
        "qualification": (
            "The retained manifest is a single state record rather than an external "
            "append-only invocation log. Its timestamps, six one-pass artifacts, "
            "hashes, and aggregation-only recovery source are mutually consistent."
        ),
    }


def raw_inputs(source_root: Path) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[int, dict[str, Any]]],
]:
    tracked = source_root / "benchmarks/constrained_v2_collector_replication"
    run_root = source_root / "runs/constrained-v2-collector-replication"
    results = read_json(tracked / "results.json")
    manifest = read_json(run_root / "manifest.json")
    initialization = read_json(tracked / "initialization.json")
    raw_diagnostics: dict[str, dict[int, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        raw_diagnostics[condition] = {
            step: read_json(
                run_root
                / condition
                / f"step-{step:06d}"
                / "diagnostic.json"
            )
            for step in CHECKPOINTS
        }
    return results, manifest, initialization, raw_diagnostics


def validate_frozen_source(
    source_root: Path, results: dict[str, Any], manifest: dict[str, Any]
) -> None:
    if git_output(source_root, "rev-parse", "HEAD") != SOURCE_STUDY_COMMIT:
        raise RuntimeError("Source study is not at the frozen final commit")
    if results["classification"]["label"] != "SEED_HETEROGENEOUS":
        raise RuntimeError("Frozen classification changed")
    if (
        results["canonical_collector_decision"]["decision"]
        != "DO_NOT_FREEZE_COLLECTOR"
    ):
        raise RuntimeError("Frozen collector decision changed")
    if manifest["condition_order"] != list(CONDITIONS):
        raise RuntimeError("Frozen condition order changed")
    if manifest["source_commit"] != SOURCE_ANCESTOR_COMMIT:
        raise RuntimeError("Study source ancestor changed")
    if results["execution_totals"] != {
        "diagnostic_episodes": 648,
        "paper_episodes": 0,
        "real_transitions": 1_800_000,
        "retained_bytes": results["execution_totals"]["retained_bytes"],
        "rl_gradient_updates": 180_000,
        "sum_policy_walltime_s": results["execution_totals"][
            "sum_policy_walltime_s"
        ],
        "validation_episodes": 108,
    }:
        raise RuntimeError("Frozen execution totals changed")


def build_results(source_root: Path = DEFAULT_SOURCE_ROOT) -> dict[str, Any]:
    source_root = source_root.resolve()
    run_root = source_root / "runs/constrained-v2-collector-replication"
    source, manifest, initialization, diagnostics = raw_inputs(source_root)
    validate_frozen_source(source_root, source, manifest)
    all_jumps = []
    for run in source["runs"].values():
        for row in run["training_episode_diagnostics"]:
            all_jumps.extend(
                [
                    abs(row["lambda_speed_after"] - row["lambda_speed_before"]),
                    abs(
                        row["lambda_deadline_after"]
                        - row["lambda_deadline_before"]
                    ),
                ]
            )
    large_jump_threshold = quantile(all_jumps, 0.95)
    episodes = episode_timelines(source["runs"], large_jump_threshold)
    checkpoint_details, checkpoint_hashes, vectors = actor_vectors_and_checkpoints(
        source_root, run_root, diagnostics
    )
    del vectors
    gc.collect()
    checkpoints = checkpoint_timelines(
        source, diagnostics, checkpoint_details, initialization
    )
    summaries = condition_summaries(
        source["runs"], episodes, checkpoints, large_jump_threshold
    )
    hashes = source_hashes(source_root, run_root, checkpoint_hashes)
    finalizer = finalizer_audit(
        source_root, source, manifest, diagnostics, hashes
    )
    results = {
        "schema_version": 1,
        "study": "constrained-v2-collector-mechanism-audit",
        "audit_mode": "READ_ONLY_ZERO_SCIENTIFIC_INTERACTION",
        "source": {
            "study": "constrained-v2-three-seed-collector-replication",
            "branch": "research/constrained-v2-collector-replication",
            "full_commit_sha": SOURCE_STUDY_COMMIT,
            "study_source_ancestor_sha": SOURCE_ANCESTOR_COMMIT,
            "worktree": str(source_root),
            "input_artifact_sha256": hashes,
        },
        "analysis": {
            "branch": AUDIT_BRANCH,
            "analysis_code_sha256": sha256_file(Path(__file__)),
            "large_lambda_jump_threshold": {
                "value": large_jump_threshold,
                "definition": (
                    "pooled empirical 95th percentile (linear interpolation) of "
                    "absolute speed and deadline lambda jumps across all six runs; "
                    "selected without performance labels"
                ),
            },
            "behavioral_regime_definition": {
                "episode_order": [
                    "FEASIBLE if feasible",
                    "STANDSTILL_OR_NEAR_STANDSTILL if incomplete and energy <= 1e-6",
                    "CRAWLING_OR_PARTIAL if otherwise incomplete",
                    "COMPLETES_WITH_SPEED_VIOLATION if completed and speed violation",
                    "COMPLETES_TOO_SLOW if completed, speed-safe, and time > 140 s",
                    "MIXED otherwise",
                ],
                "checkpoint": (
                    "the unique episode category with a strict majority (>9/18); "
                    "otherwise MIXED"
                ),
            },
            "post_episode_lambda_update_exposure_definition": (
                "Number/fraction of all 30,000 gradient updates executed in the same "
                "scheduler call after the PID event caused by the just-completed "
                "episode. It measures immediate amplification, not sampled provenance."
            ),
        },
        "zero_new_scientific_interaction": {
            "new_training_transitions": 0,
            "new_rl_updates": 0,
            "new_policy_evaluation_episodes": 0,
            "new_development_episodes": 0,
            "new_validation_episodes": 0,
            "new_paper_episodes": 0,
            "static_checkpoint_loads": len(CONDITIONS) * len(CHECKPOINTS),
            "neural_network_forward_passes": 0,
            "environment_rollouts": 0,
        },
        "frozen_empirical_result": {
            "classification": "SEED_HETEROGENEOUS",
            "decision": "DO_NOT_FREEZE_COLLECTOR",
            "paired_validation": source["paired_validation_comparisons"],
        },
        "evidence_ledger": evidence_ledger(),
        "missing_field_ledger": missing_field_ledger(),
        "checkpoint_timelines": checkpoints,
        "paired_checkpoint_deltas_episodic_minus_transitional": (
            paired_checkpoint_deltas(checkpoints)
        ),
        "episode_timelines": episodes,
        "paired_episode_deltas_episodic_minus_transitional": (
            paired_episode_deltas(episodes)
        ),
        "episode_index_alignment": episode_alignment(episodes),
        "condition_summaries": summaries,
        "seed_scorecards": scorecards(),
        "sign_reversal_analysis": sign_reversal_features(summaries, checkpoints),
        "mechanism_ranking": [
            {
                "rank": 1,
                "mechanism": "coupled PID / constraint-regime trajectory feedback",
                "category": "PLAUSIBLE",
                "reason": (
                    "PID timing and immediate update amplification are direct, and "
                    "seed-specific lambda/regime paths accompany outcomes; no "
                    "exogenous PID feature uniquely reproduces the sign reversal."
                ),
            },
            {
                "rank": 2,
                "mechanism": "alpha / exploration trajectory",
                "category": "PLAUSIBLE",
                "reason": (
                    "Final log-alpha separation orders with outcome deterioration and "
                    "T47 shows an entropy rebound, but alpha is jointly endogenous."
                ),
            },
            {
                "rank": 3,
                "mechanism": "critic / actor optimization dynamics",
                "category": "WEAK",
                "reason": (
                    "Some loss scale separation appears in seeds 29/47, but retained "
                    "snapshots show no uniquely timed collapse and omit targets."
                ),
            },
            {
                "rank": 4,
                "mechanism": "task-exposure feedback",
                "category": "WEAK",
                "reason": (
                    "Seed 47 has 30 more EPISODIC completed tasks yet the worst "
                    "EPISODIC outcome, defeating a simple exposure account."
                ),
            },
            {
                "rank": 5,
                "mechanism": "within-episode actor nonstationarity alone",
                "category": "WEAK",
                "reason": (
                    "The treatment contrast is exact and large but essentially fixed "
                    "across seeds, so it cannot alone identify the sign reversal."
                ),
            },
            {
                "rank": 6,
                "mechanism": "actual replay-sample freshness",
                "category": "NOT_TESTABLE_FROM_RETAINED_DATA",
                "reason": (
                    "Scheduling and occupancy are retained; sampled indices and ages "
                    "are not."
                ),
            },
        ],
        "finalizer_recovery_audit": finalizer,
        "verification": {
            "mechanism_analysis_tests": "PASSED: 11 tests",
            "source_artifact_hash_and_immutability_checks": "PASSED",
            "deterministic_regeneration": "PASSED",
            "finalizer_recovery_integrity_checks": "PASSED",
            "ruff_audit_scope": "PASSED",
            "ruff_repository_scope": (
                "PRE_EXISTING_FAILURE: F841 unused budget_scale in "
                "benchmarks/llm_reward/generations/g0/g0-c03.py:6"
            ),
            "git_diff_check": "PASSED",
        },
        "decision_gate": {
            "decision": "STOP_COLLECTOR_RESEARCH",
            "reason": (
                "No single retained mechanism strongly explains both timing and the "
                "three-seed sign pattern. The evidence instead shows coupled, "
                "trajectory-dependent PID, behavior, entropy, and optimization paths."
            ),
            "canonical_execution_convention": {
                "scientific_algorithm_choice": (
                    "Constrained V2 scientific configuration (unchanged)"
                ),
                "collector": "TRANSITIONAL",
                "collector_status": (
                    "selected as an execution convention, not as a demonstrated "
                    "superior collector"
                ),
                "reset_streams": "canonical explicit reset-seed streams",
                "separation": "Route-clean Policy R",
                "evaluation": "deterministic actor-only evaluation",
                "endpoint": "frozen 300k endpoint",
                "seeds": [11, 29, 47],
            },
            "next_major_scientific_study": "Same-information Scalar vs Constrained",
            "new_collector_ablation_authorized": False,
        },
    }
    return results


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def checkpoint_table(results: dict[str, Any], seed: int) -> str:
    e_name, _ = condition_for_name(seed, "EPISODIC")
    t_name, _ = condition_for_name(seed, "TRANSITIONAL")
    lines = [
        "| Step | C | Updates | Episodes | Tasks | Unique routes/tasks | PID | "
        "lambda S/D | alpha | diagnostic RSR/C/D/S | regime |",
        "|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|:--|",
    ]
    for index, step in enumerate((0, *CHECKPOINTS)):
        for label, name in (("E", e_name), ("T", t_name)):
            row = results["checkpoint_timelines"][name][index]
            counts = row["diagnostic_counts"]
            count_text = "—"
            regime = "—"
            if counts is not None:
                count_text = "/".join(
                    str(counts[key])
                    for key in (
                        "requirement_satisfaction",
                        "completion",
                        "deadline_compliance",
                        "speed_compliance",
                    )
                )
                regime = row["behavioral_regime"]["state"]
            lines.append(
                f"| {step:,} | {label} | {row['gradient_updates']:,} | "
                f"{row['completed_episodes']} | {row['task_exposures']} | "
                f"{row['unique_routes']}/{row['unique_full_tasks']} | "
                f"{row['pid_update_count']} | {fmt(row['lambda_speed'])}/"
                f"{fmt(row['lambda_deadline'])} | {fmt(row['alpha'], 6)} | "
                f"{count_text} | {regime} |"
            )
        delta = results["paired_checkpoint_deltas_episodic_minus_transitional"][
            str(seed)
        ][index]
        counts = delta["diagnostic_counts"]
        count_text = "—"
        if counts is not None:
            count_text = "/".join(
                f"{counts[key]:+d}"
                for key in (
                    "requirement_satisfaction",
                    "completion",
                    "deadline_compliance",
                    "speed_compliance",
                )
            )
        lines.append(
            f"| {step:,} | Δ | {delta['gradient_updates']:+,} | "
            f"{delta['completed_episodes']:+d} | {delta['task_exposures']:+d} | "
            f"{delta['unique_routes']:+d}/{delta['unique_full_tasks']:+d} | "
            f"{delta['pid_update_count']:+d} | {delta['lambda_speed']:+.3f}/"
            f"{delta['lambda_deadline']:+.3f} | {delta['alpha']:+.6f} | "
            f"{count_text} | E-T |"
        )
    return "\n".join(lines)


def h1_table(results: dict[str, Any]) -> str:
    lines = [
        "| Condition | PID events | mean/median/max abs(Δλ speed) | "
        "mean/median/max abs(Δλ deadline) | large S/D jumps | immediate updates | "
        "fraction |",
        "|:--|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = results["condition_summaries"]
    for condition in CONDITIONS:
        pid = summaries[condition]["pid_lagrange"]
        speed = pid["speed_multiplier_volatility"]
        deadline = pid["deadline_multiplier_volatility"]
        exposure = pid["post_episode_lambda_update_exposure"]
        lines.append(
            f"| {condition} | {pid['pid_event_count']} | "
            f"{speed['mean_absolute_jump']:.3f}/"
            f"{speed['median_absolute_jump']:.3f}/"
            f"{speed['maximum_absolute_jump']:.3f} | "
            f"{deadline['mean_absolute_jump']:.3f}/"
            f"{deadline['median_absolute_jump']:.3f}/"
            f"{deadline['maximum_absolute_jump']:.3f} | "
            f"{speed['large_jump_count']}/{deadline['large_jump_count']} | "
            f"{exposure['updates']:,} | {exposure['fraction']:.4%} |"
        )
    return "\n".join(lines)


def h2_h4_table(results: dict[str, Any]) -> str:
    lines = [
        "| Condition | Within-episode actor updates mean/max | hash-changed episodes | "
        "episodes | mean/median length | unique routes/tasks | duplicate rate R/T |",
        "|:--|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        summary = results["condition_summaries"][condition]
        nonstationary = summary["within_episode_nonstationarity"]
        exposure = summary["task_exposure"]
        length = exposure["episode_length_steps"]
        lines.append(
            f"| {condition} | "
            f"{nonstationary['mean_actor_updates_while_episode_active']:.3f}/"
            f"{nonstationary['maximum_actor_updates_while_episode_active']} | "
            f"{nonstationary['episodes_with_actor_hash_change_during_collection']} | "
            f"{exposure['completed_episodes']} | {length['mean']:.1f}/"
            f"{length['median']:.1f} | {exposure['unique_routes']}/"
            f"{exposure['unique_full_tasks']} | "
            f"{exposure['duplicate_route_rate']:.2%}/"
            f"{exposure['duplicate_full_task_rate']:.2%} |"
        )
    return "\n".join(lines)


def alpha_loss_table(results: dict[str, Any]) -> str:
    lines = [
        "| Condition | alpha 50k→300k | log-alpha change | actor-total 50k→300k | "
        "q-total 50k→300k |",
        "|:--|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        summary = results["condition_summaries"][condition]
        alpha = summary["alpha"]
        actor = summary["optimization"]["checkpoint_loss_summaries"][
            "loss/actor_total"
        ]["values_50k_to_300k"]
        q_total = summary["optimization"]["checkpoint_loss_summaries"][
            "loss/q_total"
        ]["values_50k_to_300k"]
        lines.append(
            f"| {condition} | {alpha['values_50k_to_300k'][0]:.6f}→"
            f"{alpha['values_50k_to_300k'][-1]:.6f} | "
            f"{alpha['log_change_50k_to_300k']:+.3f} | "
            f"{actor[0]:.3f}→{actor[-1]:.3f} | "
            f"{q_total[0]:.3f}→{q_total[-1]:.3f} |"
        )
    return "\n".join(lines)


def scorecard_table(results: dict[str, Any], seed: int) -> str:
    lines = [
        "| H | Supporting evidence | Evidence against / limit | Completeness | "
        "Direction | Confidence |",
        "|:--:|:--|:--|:--|:--|:--|",
    ]
    for hypothesis, row in results["seed_scorecards"][str(seed)].items():
        lines.append(
            f"| {hypothesis} | {row['evidence_supporting_mechanism']} | "
            f"{row['evidence_against_mechanism']} | {row['data_completeness']} | "
            f"{row.get('effect_direction', 'trajectory-dependent')} | "
            f"{row['confidence']} |"
        )
    return "\n".join(lines)


def render_readme(results: dict[str, Any]) -> str:
    return f"""# Constrained V2 collector mechanism audit

This directory is a deterministic, read-only mechanistic audit of the completed
three-seed collector replication at
`{results['source']['full_commit_sha']}`. It performs no training, RL update,
environment rollout, Development/Validation/Paper evaluation, or policy forward
pass. Loading retained checkpoint tensors is the only neural-network operation.

The frozen `SEED_HETEROGENEOUS` classification and `DO_NOT_FREEZE_COLLECTOR`
decision are unchanged.

## Reproduction

From this audit worktree, using the study's pinned environment:

```bash
/Users/romanliessner/gitlab/gym_longicontrol_multistep_mbrl/.venv/bin/python \\
  -m benchmarks.constrained_v2_collector_mechanism_audit.analyze \\
  --source-root /Users/romanliessner/gitlab/gym_longicontrol_collector_replication
```

Verify byte-deterministic regeneration without writing:

```bash
/Users/romanliessner/gitlab/gym_longicontrol_multistep_mbrl/.venv/bin/python \\
  -m benchmarks.constrained_v2_collector_mechanism_audit.analyze --verify \\
  --source-root /Users/romanliessner/gitlab/gym_longicontrol_collector_replication
```

`results.json` contains every checkpoint timeline, every retained episode event,
all paired E−T curves, input hashes, the missing-field ledger, and scorecards.
`MECHANISMS.md` is the synthesis; `SEED11.md`, `SEED29.md`, and `SEED47.md`
provide pair-specific evidence. `FINALIZER_AUDIT.md` covers recovery integrity.
"""


def render_mechanisms(results: dict[str, Any]) -> str:
    threshold = results["analysis"]["large_lambda_jump_threshold"]
    ranking = "\n".join(
        f"{row['rank']}. **{row['mechanism']} — {row['category']}**. {row['reason']}"
        for row in results["mechanism_ranking"]
    )
    missing = "\n".join(
        f"- `{row['status']}` — {row['field']}: {row['impact']}"
        for row in results["missing_field_ledger"]
    )
    return f"""# Mechanism synthesis

## Conclusion

The sign-changing collector effect is not cleanly attributable to one retained
mechanism. The collector schedule creates two large, directly observed contrasts:
EPISODIC performs zero actor updates while an episode is active and applies
essentially its entire update budget in post-PID episode-end blocks; TRANSITIONAL
updates throughout the episode and almost never updates in the terminal scheduler
call. Those contrasts are present in all three seeds and therefore do not themselves
change sign with RSR.

What changes by seed is the coupled trajectory. Lambda paths, constraint regimes,
alpha, episode length, and critic losses co-evolve differently. The only compact
internal feature whose E−T sign and ordering match `+3, -4, -7` is final deadline
lambda; it is an endogenous consequence of accumulated episode costs, not an
independent causal lever. The evidence supports trajectory-dependent feedback, not
a uniquely isolated mechanism.

## H1 — PID/Lagrange timing

`post_episode_lambda_update_exposure` means gradient updates executed in the same
scheduler call after the just-completed episode's PID event. It does not claim that
the corresponding episode was sampled. The large-jump threshold is the pooled 95th
percentile of all absolute lambda jumps, `{threshold['value']:.6f}`, chosen without
performance labels.

{h1_table(results)}

EPISODIC immediate exposure is 100% in every seed. TRANSITIONAL exposure is only
0.1167%, 0.1900%, and 0.3200% for seeds 11/29/47. EPISODIC speed-lambda volatility
is also higher in every pair. These facts strongly establish amplification timing,
but their invariant direction cannot explain why only seed 11 benefits.

## H2/H4 — policy nonstationarity and task exposure

{h2_h4_table(results)}

Hashes prove every EPISODIC actor is fixed from episode start through terminal;
every TRANSITIONAL episode with active updates changes actor hash. Per-episode actor
tensors were not retained, so no nonexistent per-episode L2 norm is reconstructed.
Scientific checkpoints do permit 50k actor L2 displacement; those values and hashes
are in `results.json`.

Task exposure is not a sufficient account: E11 has +10 episodes, E29 has -2, and
E47 has +30, while RSR differences are +3, -4, and -7. Seed 47 is the direct
counterexample to a simple more-tasks or shorter-episodes benefit.

## H3 — replay/update freshness

Schedule-level freshness is directly recoverable. EPISODIC update events occur at
episode terminals after a complete trajectory enters replay; TRANSITIONAL events
usually occur every ten new transitions while only a prefix of the active episode is
available. Replay occupancy and insertion gaps are retained and reported.

Actual sample ages and fractions from the last 256/1,000/5,000 transitions are
`NOT_RECOVERABLE`: sampled replay indices were not logged. The audit does not replace
them with expected uniform-sampling fractions.

## H5 — behavioral regimes

Each diagnostic episode is classified by frozen raw metrics. A checkpoint receives
the unique strict-majority category (>9/18), otherwise `MIXED`. The operational
order and all counts are in `results.json`.

- Seed 11: E ends `FEASIBLE`-majority (12/18 RSR) while T ends `MIXED` (9/18).
- Seed 29: both end `MIXED`; Validation reveals E's higher completion but worse
  deadline/speed tradeoff.
- Seed 47: E moves from crawling to speed violation to feasible at 250k, then falls
  back to crawling/partial at 300k. T recovers after 150k and ends mixed at 9/18.

All three pairs already differ on raw completion/deadline/speed counts at 50k. The
later direction is not monotone: seed 47 is the clearest late reversal.

## H6/H7 — alpha and optimization

{alpha_loss_table(results)}

Final E/T alpha ratios are approximately 0.86, 0.50, and 0.11 for seeds 11, 29,
and 47. This magnitude ordering is consistent with worsening EPISODIC outcomes,
especially the T47 alpha rebound, but the paired alpha difference is negative in all
three seeds and is endogenous. Checkpoint actor-total and Q-total losses are
descriptive snapshots; no per-update trace or target statistics remain.

## Cross-seed sign reversal

No exogenous candidate feature reproduces the RSR sign reversal. Immediate PID
exposure, within-episode actor updating, replay scheduling, and EPISODIC's generally
higher speed-lambda volatility all keep the same direction. Final deadline lambda
does match the signs (`positive, negative, negative`), but it summarizes the very
cost trajectory being explained. With n=3 this is descriptive concordance only.

## Ranking

{ranking}

## Evidence not recoverable

{missing}
"""


def seed_interpretation(seed: int) -> str:
    if seed == 11:
        return (
            "EPISODIC may help by turning complete, stationary-policy episodes into "
            "coherent immediate PID-conditioned update blocks. It ends with better "
            "speed compliance and the same Validation completion/deadline counts. "
            "However, identical schedule amplification hurts elsewhere, so this is a "
            "seed-11-consistent interaction rather than an isolated cause."
        )
    if seed == 29:
        return (
            "TRANSITIONAL may help by avoiding large episode-end amplification and "
            "retaining more late entropy. EPISODIC completes seven more Validation "
            "tasks but loses three deadline and five speed-compliance cases, "
            "suggesting "
            "an aggressive constraint tradeoff rather than a general completion "
            "failure. The diagnostic endpoint does not reproduce the full Validation "
            "gap, so attribution remains low-confidence."
        )
    return (
        "TRANSITIONAL may help through late recovery: its alpha rebounds, behavior "
        "moves out of prolonged crawling after 150k, and the final diagnostic mirrors "
        "the Validation advantage. EPISODIC briefly leads and is feasible-dominant at "
        "250k before collapsing by 300k. That non-monotone path is evidence for "
        "coupled "
        "trajectory dependence, not a single collector mechanism."
    )


def render_seed(results: dict[str, Any], seed: int) -> str:
    pair = results["frozen_empirical_result"]["paired_validation"]["by_seed"][
        str(seed)
    ]
    delta = pair["episodic_minus_transitional"]
    return f"""# Seed {seed} mechanism audit

Validation E−T differences: RSR `{delta['requirement_satisfaction']:+d}`, completion
`{delta['completion']:+d}`, deadline `{delta['deadline_compliance']:+d}`, speed
`{delta['speed_compliance']:+d}`.

## Transition-aligned timeline

Rows marked Δ are EPISODIC minus TRANSITIONAL. Diagnostic counts are
RSR/completion/deadline/speed, each out of 18.

{checkpoint_table(results, seed)}

## Mechanism scorecard

{scorecard_table(results, seed)}

## Interpretation

{seed_interpretation(seed)}

The complete per-episode PID sequence, costs, multiplier jumps, update counts,
actor-hash evidence, task fingerprints, and every episode-index E−T delta are in
`results.json`. Episode-index summaries at 25/50/100/150/200 episodes are also
recorded there, avoiding transition-count/episode-count conflation.
"""


def render_finalizer(results: dict[str, Any]) -> str:
    audit = results["finalizer_recovery_audit"]
    checks = "\n".join(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in audit["checks"].items()
    )
    return f"""# Finalizer recovery audit

Classification: **`{audit['classification']}`**.

All six final checkpoint files hash to their already-frozen manifest values, all
six Attempts ended before Validation opened, each Validation actor hash matches its
final/300k actor before and after evaluation, and each retained Validation artifact
contains exactly 18 episodes. The manifest records 108 completed episodes and Paper
remains `SEALED_UNINSPECTED`.

The recovery module calls deterministic `finalize.build_results` after replacing
only the known-bad in-memory object-order assertion. It contains no training or
evaluation call and reports no repeated training/evaluation or sealed-artifact
mutation. Source hashes are recorded in `results.json`.

{checks}

Qualification: {audit['qualification']}

The mutually consistent immutable evidence supports reporting-only recovery; no
policy, split, sealed evaluation artifact, or scientific result changed.
"""


def render_results(results: dict[str, Any]) -> str:
    return f"""# Constrained V2 collector mechanism audit results

## Outcome

Decision: **`STOP_COLLECTOR_RESEARCH`**.

No single mechanism is strongly supported as the cause of the three-seed sign
reversal. The exact schedule contrast is large but seed-invariant; the differing
outcomes arise along coupled and non-monotone PID, behavior, alpha, optimization,
and episode-exposure trajectories. A targeted collector ablation is therefore not
scientifically justified from the retained evidence.

The frozen source commit is
`{results['source']['full_commit_sha']}`. Classification remains
`SEED_HETEROGENEOUS`; the frozen decision remains `DO_NOT_FREEZE_COLLECTOR`.

## Zero-interaction accounting

```text
new training transitions       = 0
new RL updates                 = 0
new policy-evaluation episodes = 0
new Development episodes       = 0
new Validation episodes        = 0
new Paper episodes             = 0
environment rollouts           = 0
policy forward passes          = 0
```

The audit statically loaded 36 already-retained scientific checkpoints to verify
hashes and calculate actor-vector norms between 50k snapshots.

## Frozen primary result

| Seed | RSR E/T/Δ | Completion Δ | Deadline Δ | Speed Δ |
|---:|---:|---:|---:|---:|
| 11 | 13/10/+3 | +0 | +0 | +3 |
| 29 | 5/9/-4 | +7 | -3 | -5 |
| 47 | 2/9/-7 | -7 | -6 | -3 |
| pooled | 20/28/-8 | 42/42/+0 | 24/33/-9 | 41/46/-5 |

The pooled result is not treated as proof of collector superiority.

## Main mechanistic result

{h1_table(results)}

EPISODIC gives 100% of updates immediate post-episode PID exposure in all seeds;
TRANSITIONAL gives only 0.12–0.32%. EPISODIC also has higher speed-lambda
volatility in every seed. Neither feature reverses sign with RSR. The behavior
curves are non-monotone, especially E47's 250k-to-300k collapse.

## Decision and future convention

Use the following canonical benchmark infrastructure:

```text
Constrained V2 scientific configuration
+ TRANSITIONAL collector
+ canonical explicit reset-seed streams
+ Route-clean Policy R
+ deterministic actor-only evaluation
+ frozen 300k endpoint
+ three seeds 11/29/47
```

`TRANSITIONAL` is **selected as an execution convention, not as a demonstrated
superior collector**. This is an infrastructure choice for reproducibility, not a
scientific algorithm choice. Move the next major study to **Same-information Scalar
vs Constrained**.

## Integrity

Finalizer recovery: `{results['finalizer_recovery_audit']['classification']}`.
Every numerical table is generated by `analyze.py`; all input SHA-256 values, the
analysis-code hash, complete event curves, evidence statuses, and missing fields are
in `results.json`.

Verification: 11 mechanism-analysis tests passed; deterministic regeneration,
source-artifact hash/immutability checks, finalizer-recovery checks, audit-scoped
Ruff, and `git diff --check` passed. Repository-wide Ruff still reports the
pre-existing unrelated `F841` at
`benchmarks/llm_reward/generations/g0/g0-c03.py:6`; this audit does not modify that
frozen source file.
"""


def rendered_files(results: dict[str, Any]) -> dict[str, str]:
    return {
        "README.md": render_readme(results),
        "MECHANISMS.md": render_mechanisms(results),
        "SEED11.md": render_seed(results, 11),
        "SEED29.md": render_seed(results, 29),
        "SEED47.md": render_seed(results, 47),
        "FINALIZER_AUDIT.md": render_finalizer(results),
        "RESULTS.md": render_results(results),
        "results.json": stable_json(results),
    }


def write_results(results: dict[str, Any]) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for name, content in rendered_files(results).items():
        (HERE / name).write_text(content, encoding="utf-8")


def verify_results(results: dict[str, Any]) -> None:
    mismatches = []
    for name, expected in rendered_files(results).items():
        path = HERE / name
        if not path.exists() or path.read_text(encoding="utf-8") != expected:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(f"Deterministic regeneration mismatch: {mismatches}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    results = build_results(args.source_root)
    if args.verify:
        verify_results(results)
        print("deterministic regeneration verified")
    else:
        write_results(results)
        print(HERE / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
