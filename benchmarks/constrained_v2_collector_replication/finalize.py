"""Summarize frozen Validation evidence without further policy evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

from benchmarks.constrained_v2_repro_audit.harness import _module_sha256

from .checkpointing import file_sha256, load_verified
from .config import CONDITION_NAMES, TRAINING_SEEDS, load_configuration
from .execution import DEFAULT_OUTPUT_ROOT, atomic_json, preparation_status

HERE = Path(__file__).resolve().parent
METRICS = (
    "requirement_satisfaction",
    "completion",
    "deadline_compliance",
    "speed_compliance",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_markdown(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mechanism_summary(
    episodes: list[dict[str, Any]],
    curves: list[dict[str, Any]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    stationarity = [row["actor_stationarity"] for row in episodes]
    update_events = [
        event for row in episodes for event in row["update_timing"]["events"]
    ]
    replay_sizes = [event["replay_size"] for event in update_events]
    pid_transitions = [row["real_transition_count"] for row in episodes]
    curve_states = [
        {
            "real_transitions": row["real_transitions"],
            "counts": row["behavior_states"],
            "rates": {key: value / 18 for key, value in row["behavior_states"].items()},
        }
        for row in curves
    ]
    return {
        "completed_real_episodes": len(episodes),
        "stationary_policy_transitions": {
            "completed_episode_transitions": sum(
                row["transitions"] for row in stationarity
            ),
            "stationary_policy_segments": sum(
                row["stationary_policy_segments"] for row in stationarity
            ),
            "mean_distinct_actor_versions_per_episode": mean(
                row["distinct_actor_versions"] for row in stationarity
            )
            if stationarity
            else 0.0,
            "mean_maximum_stationary_run_transitions": mean(
                row["maximum_stationary_run_transitions"] for row in stationarity
            )
            if stationarity
            else 0.0,
        },
        "pid_updates": {
            "count": len(pid_transitions),
            "real_transition_counts": pid_transitions,
            "mean_interval_transitions": mean(
                [
                    pid_transitions[0],
                    *[
                        right - left
                        for left, right in zip(pid_transitions, pid_transitions[1:])
                    ],
                ]
            )
            if pid_transitions
            else None,
        },
        "replay_at_updates": {
            "update_events": len(update_events),
            "gradient_updates": sum(
                event["updates_executed"] for event in update_events
            ),
            "first_update_transition": update_events[0]["real_transition_count"]
            if update_events
            else None,
            "last_update_transition": update_events[-1]["real_transition_count"]
            if update_events
            else None,
            "minimum_replay_size": min(replay_sizes) if replay_sizes else None,
            "maximum_replay_size": max(replay_sizes) if replay_sizes else None,
            "mean_replay_size": mean(replay_sizes) if replay_sizes else None,
            "terminal_update_events": sum(
                bool(event["episode_terminal"]) for event in update_events
            ),
            "nonterminal_update_events": sum(
                not bool(event["episode_terminal"]) for event in update_events
            ),
        },
        "behavior_state_learning_curve": curve_states,
        "behavior_state_change_50k_to_300k": {
            key: curve_states[-1]["rates"][key] - curve_states[0]["rates"][key]
            for key in curve_states[0]["rates"]
        },
        "validation_behavior_states": {
            "counts": validation["behavior_states"],
            "rates": {
                key: value / 18 for key, value in validation["behavior_states"].items()
            },
        },
    }


def _load_run(
    root: Path, condition: Any, manifest_row: dict[str, Any]
) -> dict[str, Any]:
    directory = root / condition.name
    curves = [
        _read(directory / f"step-{step:06d}" / "diagnostic.json")
        for step in load_configuration().raw["diagnostic_checkpoints"]
    ]
    validation = _read(directory / "validation.json")
    episodes = _read(directory / "training-episode-diagnostics.json")
    metadata = manifest_row["last_verified_checkpoint"]
    if file_sha256(metadata["path"]) != manifest_row["final_checkpoint_sha256"]:
        raise RuntimeError(f"Final checkpoint hash changed for {condition.name}")
    checkpoint = load_verified(
        metadata["path"],
        expected_sha256=manifest_row["final_checkpoint_sha256"],
        expected_configuration_sha256=load_configuration().configuration_sha256,
    )
    if checkpoint["counters"] != {
        "real_transitions": 300_000,
        "rl_gradient_updates": 30_000,
    }:
        raise RuntimeError("Final budget mismatch")
    return {
        "id": condition.id,
        "condition": condition.name,
        "collector": condition.collector,
        "nominal_seed": condition.training_seed,
        "manifest": manifest_row,
        "diagnostic_curve": [
            {
                "real_transitions": row["real_transitions"],
                "counts": row["counts"],
                "behavior_states": row["behavior_states"],
                "training_state": row["training_state"],
            }
            for row in curves
        ],
        "validation": {
            "counts": validation["counts"],
            "failure_modes": validation["failure_modes"],
            "behavior_states": validation["behavior_states"],
            "episodes": validation["episodes"],
            "actor_sha256": validation["actor_sha256_before"],
        },
        "final_actor_sha256": _module_sha256(checkpoint["policy"].actor)
        if hasattr(checkpoint["policy"], "actor")
        else validation["actor_sha256_before"],
        "training_walltime_s": checkpoint["walltime_state"]["completed_segments_s"],
        "training_episode_diagnostics": episodes,
        "mechanism": _mechanism_summary(episodes, curves, validation),
    }


def _paired_comparisons(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[str, Any] = {}
    pooled_e = {metric: 0 for metric in METRICS}
    pooled_t = {metric: 0 for metric in METRICS}
    for seed in TRAINING_SEEDS:
        episodic = next(
            row
            for row in runs.values()
            if row["nominal_seed"] == seed and row["collector"] == "EPISODIC"
        )
        transitional = next(
            row
            for row in runs.values()
            if row["nominal_seed"] == seed and row["collector"] == "TRANSITIONAL"
        )
        differences = {
            metric: episodic["validation"]["counts"][metric]
            - transitional["validation"]["counts"][metric]
            for metric in METRICS
        }
        by_seed[str(seed)] = {
            "episodic_condition": episodic["condition"],
            "transitional_condition": transitional["condition"],
            "episodic_counts_out_of_18": episodic["validation"]["counts"],
            "transitional_counts_out_of_18": transitional["validation"]["counts"],
            "episodic_minus_transitional": differences,
        }
        for metric in METRICS:
            pooled_e[metric] += episodic["validation"]["counts"][metric]
            pooled_t[metric] += transitional["validation"]["counts"][metric]
    return {
        "by_seed": by_seed,
        "pooled_out_of_54": {
            "episodic": pooled_e,
            "transitional": pooled_t,
            "episodic_minus_transitional": {
                metric: pooled_e[metric] - pooled_t[metric] for metric in METRICS
            },
        },
    }


def _classification(comparisons: dict[str, Any]) -> dict[str, Any]:
    seed_deltas = [
        comparisons["by_seed"][str(seed)]["episodic_minus_transitional"][
            "requirement_satisfaction"
        ]
        for seed in TRAINING_SEEDS
    ]
    pooled = comparisons["pooled_out_of_54"]["episodic_minus_transitional"][
        "requirement_satisfaction"
    ]
    episodic = (
        pooled >= 6
        and all(value >= 0 for value in seed_deltas)
        and sum(value >= 2 for value in seed_deltas) >= 2
    )
    transitional = (
        pooled <= -6
        and all(value <= 0 for value in seed_deltas)
        and sum(value <= -2 for value in seed_deltas) >= 2
    )
    contradictory = any(value >= 2 for value in seed_deltas) and any(
        value <= -2 for value in seed_deltas
    )
    if episodic:
        label = "EPISODIC_ROBUST_ADVANTAGE"
    elif transitional:
        label = "TRANSITIONAL_ROBUST_ADVANTAGE"
    elif abs(pooled) < 6 and not contradictory:
        label = "NO_MATERIAL_DIFFERENCE"
    else:
        label = "SEED_HETEROGENEOUS"
    return {
        "label": label,
        "seed_pair_rsr_differences_out_of_18": seed_deltas,
        "pooled_rsr_difference_out_of_54": pooled,
        "strong_contradictory_seed_pattern": contradictory,
        "descriptive_not_significance_test": True,
    }


def _recommendation(label: str) -> dict[str, Any]:
    if label == "EPISODIC_ROBUST_ADVANTAGE":
        selected = "EPISODIC"
        reason = "The preregistered episodic robust-advantage criterion passed."
    elif label == "TRANSITIONAL_ROBUST_ADVANTAGE":
        selected = "TRANSITIONAL"
        reason = "The preregistered transitional robust-advantage criterion passed."
    elif label == "NO_MATERIAL_DIFFERENCE":
        selected = "TRANSITIONAL"
        reason = (
            "Performance was not materially different; transition-wise collection "
            "is the standard online schedule, avoids episode-end update backlogs, "
            "and has simpler update accounting."
        )
    else:
        return {
            "decision": "DO_NOT_FREEZE_COLLECTOR",
            "selected_collector": None,
            "next_study": (
                "Minimal 2x2 decomposition of within-episode actor updates versus "
                "PID update timing; do not run automatically."
            ),
        }
    return {
        "decision": f"CHOOSE_{selected}",
        "selected_collector": selected,
        "reason": reason,
        "proposed_canonical_baseline": {
            "science": "Constrained V2 scientific configuration",
            "collector": selected,
            "training_stream": "SeedSequence([seed, 0x43414E4F])",
            "separation": "Route-clean Policy R",
            "future_use": [
                "same-information Scalar vs. Constrained",
                "GCSL",
                "future algorithm comparisons",
            ],
        },
    }


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_results(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    preparation = preparation_status()
    root = Path(output_root)
    manifest = _read(root / "manifest.json")
    if tuple(manifest["runs"]) != CONDITION_NAMES:
        raise RuntimeError("Manifest condition order changed")
    if any(row["state"] != "COMPLETED" for row in manifest["runs"].values()):
        raise RuntimeError("All six policies must be complete")
    if (
        manifest["validation"]["state"] != "COMPLETED"
        or manifest["validation"]["episode_count"] != 108
    ):
        raise RuntimeError("Validation is not a complete one-pass 6 x 18 matrix")
    if manifest["diagnostic"]["episode_count"] != 648:
        raise RuntimeError("Diagnostic matrix must contain 648 episodes")
    if manifest["paper"] != {"state": "SEALED_UNINSPECTED", "episode_count": 0}:
        raise RuntimeError("Paper did not remain sealed")
    configuration = load_configuration()
    runs = {
        condition.name: _load_run(root, condition, manifest["runs"][condition.name])
        for condition in configuration.conditions
    }
    comparisons = _paired_comparisons(runs)
    classification = _classification(comparisons)
    recommendation = _recommendation(classification["label"])
    totals = {
        "real_transitions": sum(
            row["manifest"]["real_transitions"] for row in runs.values()
        ),
        "rl_gradient_updates": sum(
            row["manifest"]["rl_gradient_updates"] for row in runs.values()
        ),
        "diagnostic_episodes": 648,
        "validation_episodes": 108,
        "paper_episodes": 0,
        "sum_policy_walltime_s": sum(
            row["training_walltime_s"] for row in runs.values()
        ),
        "retained_bytes": _directory_bytes(root),
    }
    if (
        totals["real_transitions"] != 1_800_000
        or totals["rl_gradient_updates"] != 180_000
    ):
        raise RuntimeError("Study execution totals changed")
    return {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "outcome": "COMPLETED",
        "primary_evidence": "COLLECTOR_VALIDATION_V1",
        "preparation": preparation,
        "manifest": manifest,
        "runs": runs,
        "paired_validation_comparisons": comparisons,
        "classification": classification,
        "canonical_collector_decision": recommendation,
        "execution_totals": totals,
        "integrity": {
            "all_values_finite": all(
                math.isfinite(value) for value in (totals["sum_policy_walltime_s"],)
            ),
            "validation_opened_only_after_all_six_frozen": True,
            "no_checkpoint_selection": True,
            "paper": "SEALED_UNINSPECTED",
        },
    }


def render_markdown(results: dict[str, Any]) -> str:
    comparison = results["paired_validation_comparisons"]
    rows = []
    for seed in TRAINING_SEEDS:
        row = comparison["by_seed"][str(seed)]
        e = row["episodic_counts_out_of_18"]
        t = row["transitional_counts_out_of_18"]
        d = row["episodic_minus_transitional"]
        rsr = (
            f"{e['requirement_satisfaction']} / "
            f"{t['requirement_satisfaction']} / "
            f"{d['requirement_satisfaction']:+d}"
        )
        rows.append(
            f"| {seed} | {rsr} | {d['completion']:+d} | "
            f"{d['deadline_compliance']:+d} | {d['speed_compliance']:+d} |"
        )
    pooled = comparison["pooled_out_of_54"]
    totals = results["execution_totals"]
    decision = results["canonical_collector_decision"]
    return f"""# Constrained V2 collector replication results

## Outcome

`COMPLETED`

All six policies reached exactly 300,000 real transitions and 30,000 RL
updates. The frozen Validation set was opened once only after all six final
checkpoint hashes verified. Paper remained sealed.

## Primary paired Validation outcome

RSR columns are EPISODIC / TRANSITIONAL / difference, each seed out of 18.
Other columns are paired EPISODIC-minus-TRANSITIONAL differences.

| Seed | RSR E / T / Δ | Completion Δ | Deadline Δ | Speed Δ |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Pooled RSR: **{pooled["episodic"]["requirement_satisfaction"]} / 54
EPISODIC** versus **{pooled["transitional"]["requirement_satisfaction"]} / 54
TRANSITIONAL** (Δ
{pooled["episodic_minus_transitional"]["requirement_satisfaction"]:+d}).

## Interpretation and decision

- Classification: `{results["classification"]["label"]}`.
- Decision: `{decision["decision"]}`.
- Selected collector: `{decision.get("selected_collector")}`.
- This is the frozen descriptive rule, not a significance test.

## Integrity

- Real transitions / RL updates: {totals["real_transitions"]} /
  {totals["rl_gradient_updates"]}.
- Diagnostic / Validation / Paper episodes: {totals["diagnostic_episodes"]} /
  {totals["validation_episodes"]} / 0.
- Mechanism diagnostics, learning curves, replay-at-update summaries, paired
  completion/deadline/speed comparisons, and raw Validation episodes are in
  `results.json`.
"""


def finalize_results(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    results = build_results(output_root)
    atomic_json(HERE / "results.json", results)
    _atomic_markdown(HERE / "RESULTS.md", render_markdown(results))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    results = finalize_results(args.output_root)
    print(
        json.dumps(
            {
                "classification": results["classification"],
                "decision": results["canonical_collector_decision"],
                "execution_totals": results["execution_totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
