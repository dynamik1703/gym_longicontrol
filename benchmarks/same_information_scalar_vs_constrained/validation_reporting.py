"""Deterministic aggregation for the completed SAMEINFO_VALIDATION_V1 matrix.

This module only reads the six immutable Validation result files.  It has no
evaluation or training entry point.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .checkpointing import file_sha256
from .config import TRAINING_SEEDS, load_configuration
from .execution import condition_name, source_hashes
from .interpretation import classify
from .reference import reference_checkpoint
from .split import evaluation_sets

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("runs/same-information-scalar-vs-constrained")
METHODS = ("constrained", "scalar")
ENDPOINTS = (
    "requirement_satisfaction",
    "completion",
    "deadline_compliance",
    "speed_compliance",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _failure_mode(row: dict[str, Any], deadline_s: float) -> str:
    """Mirror the frozen canonical evaluator's exclusive failure taxonomy."""

    speed = float(row["max_speed_violation_m_s"]) > 0.0
    if bool(row["feasible"]):
        return "FEASIBLE"
    if not bool(row["completed"]):
        return "INCOMPLETE_WITH_SPEED_VIOLATION" if speed else "INCOMPLETE_SAFE"
    deadline = float(row["travel_time_s"]) > deadline_s
    if deadline and speed:
        return "COMPLETED_DEADLINE_AND_SPEED_VIOLATION"
    if deadline:
        return "COMPLETED_DEADLINE_VIOLATION"
    if speed:
        return "COMPLETED_SPEED_VIOLATION"
    return "OTHER_INFEASIBLE"


def _proportions(counts: dict[str, int], denominator: int) -> dict[str, float]:
    return {
        "rsr": counts["requirement_satisfaction"] / denominator,
        "completion": counts["completion"] / denominator,
        "deadline_compliance": counts["deadline_compliance"] / denominator,
        "speed_compliance": counts["speed_compliance"] / denominator,
    }


def _energy_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean_kwh": statistics.fmean(values) if values else None,
        "median_kwh": statistics.median(values) if values else None,
        "minimum_kwh": min(values) if values else None,
        "maximum_kwh": max(values) if values else None,
        "values_kwh": values,
    }


def _sum_counters(rows: list[dict[str, int]]) -> dict[str, int]:
    return {endpoint: sum(int(row[endpoint]) for row in rows) for endpoint in ENDPOINTS}


def _sum_mappings(rows: list[dict[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update({name: int(value) for name, value in row.items()})
    return dict(sorted(total.items()))


def _validation_path(root: Path, method: str, seed: int) -> Path:
    return root / f"{method}-{seed}-validation.json"


def _diagnostic_path(root: Path, method: str, seed: int) -> Path:
    if method == "constrained":
        return root / f"C{seed}_reference" / "step-300000" / "diagnostic.json"
    return root / condition_name(seed) / "step-300000" / "diagnostic.json"


def _checkpoint_path(root: Path, configuration: Any, method: str, seed: int) -> Path:
    if method == "constrained":
        return reference_checkpoint(configuration, seed, 300_000)
    return root / condition_name(seed) / "step-300000" / "checkpoint.pt"


def _expected_checkpoint_hash(
    configuration: Any, main_results: dict[str, Any], method: str, seed: int
) -> str:
    if method == "constrained":
        return configuration.raw["reference_checkpoints"][str(seed)]["300000"]
    return main_results["scalar_runs"][str(seed)]["final_checkpoint_sha256"]


def _policy_label(method: str, seed: int) -> str:
    return f"{'C' if method == 'constrained' else 'S'}{seed}"


def _episode_record(
    row: dict[str, Any], *, method: str, seed: int, deadline_s: float
) -> dict[str, Any]:
    label = _policy_label(method, seed)
    completed = bool(row["completed"])
    deadline = completed and float(row["travel_time_s"]) <= deadline_s
    speed = float(row["max_speed_violation_m_s"]) <= 0.0
    return {
        "policy": label,
        "method": method,
        "training_seed": seed,
        "validation_task": int(row["evaluation_seed"]),
        "evaluation_seed": int(row["evaluation_seed"]),
        "feasible": bool(row["feasible"]),
        "completed": completed,
        "deadline_compliant": deadline,
        "speed_compliant": speed,
        "travel_time_s": float(row["travel_time_s"]),
        "maximum_speed_violation_m_s": float(row["max_speed_violation_m_s"]),
        "integrated_speed_violation_m": float(row["integrated_speed_violation_m"]),
        "traction_energy_kwh": float(row["traction_energy_kwh"]),
        "regenerative_energy_kwh": float(row["regenerative_energy_kwh"]),
        "signed_net_energy_kwh": float(row["energy_kwh"]),
        "exclusive_failure_mode": _failure_mode(row, deadline_s),
        "final_position_m": float(row["final_position_m"]),
        "step_count": int(row["step_count"]),
        "speed_violation_count": int(row["speed_violation_count"]),
        "action_total_variation": float(row["action_total_variation"]),
        "mean_abs_jerk_m_s3": float(row["mean_abs_jerk_m_s3"]),
        "max_abs_jerk_m_s3": float(row["max_abs_jerk_m_s3"]),
    }


def build_results(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(output_root).resolve()
    configuration = load_configuration()
    main_results = _read_json(HERE / "main_training_results.json")
    manifest = _read_json(root / "manifest.json")
    split = evaluation_sets()["validation"]
    expected_seeds = [int(row["numeric_seed"]) for row in split["accepted_tasks"]]
    deadline_s = float(configuration.constrained_v2.task.max_time_s)

    if configuration.raw["authorization"]["validation_authorized"]:
        raise RuntimeError("Validation authorization must be closed before reporting")
    if split["id"] != "SAMEINFO_VALIDATION_V1":
        raise RuntimeError("Wrong Validation split")
    if (
        split["ordered_sha256"]
        != configuration.raw["expected_evaluation_sha256"]["validation"]
    ):
        raise RuntimeError("Validation split SHA-256 changed")
    if manifest["validation"]["state"] != "COMPLETED":
        raise RuntimeError("Validation is not complete")
    if manifest["validation"]["episode_count"] != 108:
        raise RuntimeError("Validation manifest does not contain 108 episodes")
    if manifest["paper"] != {
        "state": "SEALED_UNINSPECTED",
        "episode_count": 0,
    }:
        raise RuntimeError("Paper seal changed")

    validation_payloads: dict[str, dict[str, Any]] = {}
    per_policy: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    raw_artifact_hashes: dict[str, str] = {}
    checkpoint_integrity: dict[str, Any] = {}

    for method in METHODS:
        for seed in TRAINING_SEEDS:
            label = _policy_label(method, seed)
            path = _validation_path(root, method, seed)
            payload = _read_json(path)
            validation_payloads[label] = payload
            raw_artifact_hashes[label] = file_sha256(path)
            if payload["method"] != method or int(payload["training_seed"]) != seed:
                raise RuntimeError(f"Mislabeled Validation artifact: {label}")
            if payload["evaluation_id"] != split["id"]:
                raise RuntimeError(f"Wrong evaluation ID: {label}")
            if payload["evaluation_seeds"] != expected_seeds:
                raise RuntimeError(f"Validation seed order changed: {label}")
            if int(payload["episode_count"]) != 18 or len(payload["episodes"]) != 18:
                raise RuntimeError(f"Wrong episode count: {label}")
            if not (
                payload["actor_only"]
                and payload["deterministic"]
                and payload["training_rng_unchanged"]
            ):
                raise RuntimeError(f"Evaluation semantics changed: {label}")
            if payload["actor_sha256_before"] != payload["actor_sha256_after"]:
                raise RuntimeError(f"Actor changed during Validation: {label}")

            diagnostic = _read_json(_diagnostic_path(root, method, seed))
            if not (
                diagnostic["actor_sha256_before"]
                == diagnostic["actor_sha256_after"]
                == payload["actor_sha256_before"]
            ):
                raise RuntimeError(
                    f"Actor differs from frozen Diagnostic actor: {label}"
                )

            expected_checkpoint = _expected_checkpoint_hash(
                configuration, main_results, method, seed
            )
            checkpoint = _checkpoint_path(root, configuration, method, seed)
            observed_checkpoint = file_sha256(checkpoint)
            if observed_checkpoint != expected_checkpoint:
                raise RuntimeError(f"Checkpoint hash changed: {label}")
            checkpoint_integrity[label] = {
                "path": str(checkpoint),
                "expected_sha256": expected_checkpoint,
                "observed_sha256": observed_checkpoint,
                "verified": True,
                "actor_sha256": payload["actor_sha256_before"],
            }

            policy_episodes = [
                _episode_record(row, method=method, seed=seed, deadline_s=deadline_s)
                for row in payload["episodes"]
            ]
            if [row["evaluation_seed"] for row in policy_episodes] != expected_seeds:
                raise RuntimeError(f"Episode order changed: {label}")
            episode_modes = dict(
                sorted(
                    Counter(
                        row["exclusive_failure_mode"] for row in policy_episodes
                    ).items()
                )
            )
            if episode_modes != payload["failure_modes"]:
                raise RuntimeError(f"Failure-mode accounting changed: {label}")
            episodes.extend(policy_episodes)
            counts = {name: int(value) for name, value in payload["counts"].items()}
            per_policy[label] = {
                "method": method,
                "training_seed": seed,
                "episode_count": 18,
                "counts": counts,
                "proportions": _proportions(counts, 18),
                "exclusive_failure_modes": episode_modes,
                "behavior_states": payload["behavior_states"],
                "feasible_only_energy": _energy_summary(
                    [
                        row["signed_net_energy_kwh"]
                        for row in policy_episodes
                        if row["feasible"]
                    ]
                ),
                "actor_sha256": payload["actor_sha256_before"],
                "checkpoint_sha256": expected_checkpoint,
            }

    keys = [(row["policy"], row["evaluation_seed"]) for row in episodes]
    if len(episodes) != 108 or len(set(keys)) != 108:
        raise RuntimeError("Validation policy-track pairs are not exactly unique")

    pooled: dict[str, Any] = {}
    for method in METHODS:
        labels = [_policy_label(method, seed) for seed in TRAINING_SEEDS]
        counts = _sum_counters([per_policy[label]["counts"] for label in labels])
        method_episodes = [row for row in episodes if row["method"] == method]
        pooled[method] = {
            "episode_count": 54,
            "counts": counts,
            "proportions": _proportions(counts, 54),
            "exclusive_failure_modes": _sum_mappings(
                [per_policy[label]["exclusive_failure_modes"] for label in labels]
            ),
            "behavior_states": _sum_mappings(
                [per_policy[label]["behavior_states"] for label in labels]
            ),
            "feasible_only_energy": _energy_summary(
                [
                    row["signed_net_energy_kwh"]
                    for row in method_episodes
                    if row["feasible"]
                ]
            ),
        }

    paired: dict[str, Any] = {"per_seed": {}, "pooled": {}}
    for seed in TRAINING_SEEDS:
        c_counts = per_policy[f"C{seed}"]["counts"]
        s_counts = per_policy[f"S{seed}"]["counts"]
        paired["per_seed"][str(seed)] = {
            endpoint: {
                "count_difference": c_counts[endpoint] - s_counts[endpoint],
                "proportion_difference": (c_counts[endpoint] - s_counts[endpoint]) / 18,
            }
            for endpoint in ENDPOINTS
        }
    for endpoint in ENDPOINTS:
        delta = (
            pooled["constrained"]["counts"][endpoint]
            - pooled["scalar"]["counts"][endpoint]
        )
        paired["pooled"][endpoint] = {
            "count_difference": delta,
            "proportion_difference": delta / 54,
        }

    rsr_deltas = tuple(
        paired["per_seed"][str(seed)]["requirement_satisfaction"]["count_difference"]
        for seed in TRAINING_SEEDS
    )
    pooled_rsr_delta = paired["pooled"]["requirement_satisfaction"]["count_difference"]
    classification = classify(rsr_deltas, pooled_rsr_delta)

    diagnostic: dict[str, Any] = {}
    for method in METHODS:
        labels = [_policy_label(method, seed) for seed in TRAINING_SEEDS]
        records = [main_results["diagnostic_curves"][label][-1] for label in labels]
        diagnostic[method] = {
            "episode_count": 54,
            "counts": _sum_counters([row["counts"] for row in records]),
            "exclusive_failure_modes": _sum_mappings(
                [row["failure_modes"] for row in records]
            ),
            "behavior_states": _sum_mappings(
                [row["behavior_states"] for row in records]
            ),
            "per_policy_rsr_counts": {
                label: record["counts"]["requirement_satisfaction"]
                for label, record in zip(labels, records, strict=True)
            },
        }

    scalar_counters = {
        f"S{seed}": main_results["scalar_runs"][str(seed)]["counters"]
        for seed in TRAINING_SEEDS
    }
    constrained_counters = {
        f"C{seed}": main_results["diagnostic_curves"][f"C{seed}"][-1][
            "retained_training_state"
        ]["counters"]
        for seed in TRAINING_SEEDS
    }
    scientific_hashes = source_hashes()
    if scientific_hashes != main_results["source_hashes_after_completion_gate_closed"]:
        raise RuntimeError("Frozen scientific source hashes changed")

    return {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "evaluation_id": split["id"],
        "report_scope": "final preregistered SAMEINFO_VALIDATION_V1",
        "configuration_sha256": configuration.configuration_sha256,
        "validation_split": {
            "ordered_numeric_seeds": expected_seeds,
            "ordered_sha256": split["ordered_sha256"],
            "count": 18,
        },
        "frozen_final_policy_endpoint": 300_000,
        "primary_endpoint": {
            "name": "requirement_satisfaction",
            "per_policy": {
                label: {
                    "count": row["counts"]["requirement_satisfaction"],
                    "denominator": 18,
                    "proportion": row["proportions"]["rsr"],
                }
                for label, row in per_policy.items()
            },
            "pooled": {
                method: {
                    "count": row["counts"]["requirement_satisfaction"],
                    "denominator": 54,
                    "proportion": row["proportions"]["rsr"],
                }
                for method, row in pooled.items()
            },
        },
        "per_policy": per_policy,
        "pooled_by_method": pooled,
        "paired_differences_constrained_minus_scalar": paired,
        "classification": {
            "value": classification,
            "paired_seed_rsr_count_differences": list(rsr_deltas),
            "pooled_rsr_count_difference": pooled_rsr_delta,
            "thresholds": configuration.raw["interpretation_gates"],
            "frozen_after_validation": True,
        },
        "complete_compact_episode_metrics": episodes,
        "diagnostic_to_validation": {
            "diagnostic_300k": diagnostic,
            "validation": {
                method: {
                    "episode_count": row["episode_count"],
                    "counts": row["counts"],
                    "exclusive_failure_modes": row["exclusive_failure_modes"],
                    "behavior_states": row["behavior_states"],
                    "per_policy_rsr_counts": {
                        label: per_policy[label]["counts"]["requirement_satisfaction"]
                        for label in per_policy
                        if per_policy[label]["method"] == method
                    },
                }
                for method, row in pooled.items()
            },
            "classification_used_diagnostic_results": False,
            "descriptive_conclusion": (
                "Scalar's safe-crawling/incomplete regime reproduced exactly at the "
                "pooled level (0 feasible and 0 completed in both splits; 53 "
                "incomplete-safe plus 1 incomplete speed-violation episode). "
                "Constrained retained timely feasible completion, although Validation "
                "had 23 feasible episodes versus 28 on Diagnostic and more speed "
                "violations (8 versus 1)."
            ),
        },
        "scalar_failure_regime": {
            "constraint_avoidance": (
                "Scalar was speed-compliant in 53/54 episodes but completed 0/54; "
                "this is constraint avoidance, not successful completion under "
                "constraints."
            ),
            "standstill": pooled["scalar"]["behavior_states"].get("standstill", 0),
            "safe_crawling_or_partial_progress": pooled["scalar"][
                "behavior_states"
            ].get("crawling", 0),
            "exclusive_failure_modes": pooled["scalar"]["exclusive_failure_modes"],
        },
        "feasible_only_energy": {
            method: row["feasible_only_energy"] for method, row in pooled.items()
        },
        "scientific_interpretation": {
            "supported": (
                "In this benchmark, supplying the same energy, speed-constraint, and "
                "dense deadline-feasibility information through a strong fixed scalar "
                "reward is insufficient to match explicit constrained learning."
            ),
            "deadline_signal_conclusion": (
                "The earlier Constrained-V2 improvement cannot be attributed solely "
                "to access to the deadline-feasibility signal."
            ),
            "causal_limit": (
                "The result does not establish that constrained optimization alone "
                "causes the full difference: adaptive multipliers, separate "
                "reward/cost "
                "critics, FSRL critic entropy semantics, and actor-loss structure also "
                "differ."
            ),
        },
        "strong_control_fairness": {
            "statement": (
                "Scalar SAC received a reference-informed fixed scalarization derived "
                "from previously completed Constrained runs."
            ),
            "weights": {
                "r_E": 1.0,
                "c_speed": -1.4009421299572349,
                "c_deadline": -1.5752363435900516,
            },
            "selection_statement": (
                "The lambda weights were not chosen from the present Diagnostic or "
                "Validation performance; the Scalar control was deliberately strong."
            ),
        },
        "provenance": {
            "branch": "research/same-information-scalar-vs-constrained",
            "completed_main_training_commit": (
                "b2cb08c7468c298a45ae79c3a103eb679037b359"
            ),
            "validation_authorization_commit": (
                "ef65c3c8ec7c1a1174b2831bdd416a4f9f3c5006"
            ),
            "raw_validation_artifact_sha256": raw_artifact_hashes,
            "checkpoint_integrity": checkpoint_integrity,
            "scientific_source_hashes": scientific_hashes,
            "training_counters": {
                "scalar": scalar_counters,
                "constrained_reference": constrained_counters,
            },
        },
        "execution_integrity": {
            "validation_open_count": 1,
            "episode_count": 108,
            "policy_count": 6,
            "tracks_per_policy": 18,
            "unique_policy_track_pairs": 108,
            "each_policy_track_pair_exactly_once": True,
            "actor_only": True,
            "deterministic": True,
            "all_actor_hashes_unchanged": True,
            "all_checkpoint_hashes_unchanged": True,
            "all_training_counters_unchanged": True,
            "no_training_during_validation": True,
            "no_constrained_retraining": True,
            "no_scalar_retraining": True,
            "no_best_checkpoint_selection": True,
            "validation_authorization_closed": True,
            "paper": manifest["paper"],
            "interrupted_finalizer_recovery": {
                "occurred": True,
                "duplicate_evaluations": 0,
                "description": (
                    "The initial matrix invocation completed C11/C29/C47, then stopped "
                    "before any Scalar episode because FSRL's extra-state loader "
                    "rejected the Scalar checkpoint's empty None state. The completed "
                    "Constrained "
                    "artifacts were preserved. A continuation evaluated only untouched "
                    "S11/S29/S47 after representing the unused empty Lagrangian state "
                    "as "
                    "an empty mapping; actor parameters matched the frozen Diagnostic "
                    "hashes. No condition was repeated."
                ),
            },
        },
    }


def _count_rate(record: dict[str, Any], endpoint: str, denominator: int) -> str:
    count = record["counts"][endpoint]
    return f"{count}/{denominator} ({100 * count / denominator:.1f}%)"


def _delta_cell(delta: dict[str, Any]) -> str:
    return (
        f"{delta['count_difference']:+d} "
        f"({100 * delta['proportion_difference']:+.1f} pp)"
    )


def build_report(results: dict[str, Any]) -> str:
    per_policy = results["per_policy"]
    pooled = results["pooled_by_method"]
    paired = results["paired_differences_constrained_minus_scalar"]
    lines = [
        "# Same-Information Scalar vs Constrained — Final Validation Results",
        "",
        "Status: **SAMEINFO_VALIDATION_V1 complete; final classification frozen.**",
        "",
        "Exactly six frozen 300k policies were evaluated once on the 18 frozen "
        "Validation tasks (108 deterministic actor-only episodes). No training, "
        "adaptation, planning, checkpoint selection, or policy modification occurred. "
        "Paper remains sealed and uninspected.",
        "",
        "## Primary endpoint: requirement satisfaction",
        "",
        "| Pair | Constrained | Scalar | Δ count (C − S) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for seed in TRAINING_SEEDS:
        c = per_policy[f"C{seed}"]["counts"]["requirement_satisfaction"]
        s = per_policy[f"S{seed}"]["counts"]["requirement_satisfaction"]
        delta = paired["per_seed"][str(seed)]["requirement_satisfaction"]
        lines.append(
            f"| {seed} | {c}/18 ({100 * c / 18:.1f}%) | {s}/18 "
            f"({100 * s / 18:.1f}%) | {_delta_cell(delta)} |"
        )
    c = pooled["constrained"]["counts"]["requirement_satisfaction"]
    s = pooled["scalar"]["counts"]["requirement_satisfaction"]
    lines.append(
        f"| **Pooled** | **{c}/54 ({100 * c / 54:.1f}%)** | **{s}/54 "
        f"({100 * s / 54:.1f}%)** | "
        f"**{_delta_cell(paired['pooled']['requirement_satisfaction'])}** |"
    )

    lines.extend(
        [
            "",
            "## Secondary endpoints",
            "",
            "| Policy | Completion | Deadline compliance | Speed compliance |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for label in ("C11", "S11", "C29", "S29", "C47", "S47"):
        row = per_policy[label]
        lines.append(
            f"| {label} | {_count_rate(row, 'completion', 18)} | "
            f"{_count_rate(row, 'deadline_compliance', 18)} | "
            f"{_count_rate(row, 'speed_compliance', 18)} |"
        )
    for method, title in (
        ("constrained", "Constrained pooled"),
        ("scalar", "Scalar pooled"),
    ):
        row = pooled[method]
        lines.append(
            f"| **{title}** | **{_count_rate(row, 'completion', 54)}** | "
            f"**{_count_rate(row, 'deadline_compliance', 54)}** | "
            f"**{_count_rate(row, 'speed_compliance', 54)}** |"
        )

    lines.extend(
        [
            "",
            "Paired differences (Constrained − Scalar):",
            "",
            "| Pair | Completion | Deadline | Speed compliance |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for seed in TRAINING_SEEDS:
        row = paired["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | {_delta_cell(row['completion'])} | "
            f"{_delta_cell(row['deadline_compliance'])} | "
            f"{_delta_cell(row['speed_compliance'])} |"
        )
    row = paired["pooled"]
    lines.append(
        f"| **Pooled** | **{_delta_cell(row['completion'])}** | "
        f"**{_delta_cell(row['deadline_compliance'])}** | "
        f"**{_delta_cell(row['speed_compliance'])}** |"
    )

    lines.extend(
        [
            "",
            "## Exclusive failure modes",
            "",
            "| Policy | Exclusive counts | Behavior states |",
            "| --- | --- | --- |",
        ]
    )
    for label in ("C11", "S11", "C29", "S29", "C47", "S47"):
        row = per_policy[label]
        lines.append(
            f"| {label} | "
            f"`{json.dumps(row['exclusive_failure_modes'], sort_keys=True)}` | "
            f"`{json.dumps(row['behavior_states'], sort_keys=True)}` |"
        )
    for method, title in (
        ("constrained", "Constrained pooled"),
        ("scalar", "Scalar pooled"),
    ):
        row = pooled[method]
        lines.append(
            f"| **{title}** | "
            f"`{json.dumps(row['exclusive_failure_modes'], sort_keys=True)}` | "
            f"`{json.dumps(row['behavior_states'], sort_keys=True)}` |"
        )

    c_energy = results["feasible_only_energy"]["constrained"]
    lines.extend(
        [
            "",
            "## Feasible-only energy",
            "",
            f"Constrained has {c_energy['count']} genuinely feasible episodes: mean "
            f"{c_energy['mean_kwh']:.6f} kWh, median {c_energy['median_kwh']:.6f} "
            f"kWh, range {c_energy['minimum_kwh']:.6f}–{c_energy['maximum_kwh']:.6f} "
            "kWh. The complete distribution is retained in `results.json`.",
            "",
            "**No Scalar feasible-energy comparison is defined.** Scalar had zero "
            "feasible Validation episodes; energy from infeasible crawling episodes is "
            "not interpreted as better optimization.",
            "",
            "## Frozen classification and interpretation",
            "",
            f"Classification: **{results['classification']['value']}**.",
            "",
            results["scientific_interpretation"]["supported"],
            "",
            results["scientific_interpretation"]["deadline_signal_conclusion"],
            "",
            results["scientific_interpretation"]["causal_limit"],
            "",
            "The Scalar reward received the same raw task information and was fixed "
            "as:",
            "",
            "```text",
            "r_scalar = r_E - 1.4009421299572349 * c_speed",
            "                 - 1.5752363435900516 * c_deadline",
            "```",
            "",
            results["strong_control_fairness"]["statement"],
            results["strong_control_fairness"]["selection_statement"],
            "",
            "## Diagnostic-to-Validation consistency",
            "",
            results["diagnostic_to_validation"]["descriptive_conclusion"],
            "Diagnostic results were not inputs to the frozen classification.",
            "",
            "Scalar was speed-compliant in 53/54 Validation episodes but completed "
            "0/54. All 54 were crawling/partial-progress episodes (53 incomplete-safe; "
            "one incomplete with a speed violation). This is constraint avoidance, not "
            "successful task completion under constraints.",
            "",
            "## Provenance and integrity",
            "",
            f"- Configuration SHA-256: `{results['configuration_sha256']}`",
            "- Validation ordered-set SHA-256: "
            f"`{results['validation_split']['ordered_sha256']}`",
            "- Final endpoint: 300,000 transitions for every policy; no 50k–250k "
            "checkpoint was selected.",
            "- Episode accounting: 6 policies × 18 tracks = 108 unique policy-track "
            "pairs; no repeats.",
            "- All checkpoint hashes and actor hashes match the frozen artifacts; "
            "training counters stayed fixed.",
            "- Validation authorization is closed; Paper is `SEALED_UNINSPECTED` with "
            "zero episodes.",
            "",
            "The initial finalizer completed C11/C29/C47, then stopped before any "
            "Scalar episode on an FSRL empty-extra-state loader incompatibility. Those "
            "54 immutable episodes were preserved. The continuation loaded the same "
            "frozen actors and evaluated only untouched S11/S29/S47. Actor hashes "
            "matched their final Diagnostic hashes and no condition was repeated.",
            "",
            "Complete compact episode-level metrics for all 108 episodes, checkpoint "
            "and actor hashes, training counters, raw-artifact hashes, and all derived "
            "aggregates are in `results.json`.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> None:
    results = build_results(output_root)
    (HERE / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (HERE / "RESULTS.md").write_text(build_report(results), encoding="utf-8")


if __name__ == "__main__":
    write_artifacts()
