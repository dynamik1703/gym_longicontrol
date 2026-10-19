"""Validate and summarize a completed V3 matrix without further evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any

from benchmarks.constrained_v2_seed11_factorial.stream_preflight import (
    stable_state_sha256,
)

from .checkpointing import load_verified
from .config import CHECKPOINTS, CONDITION_NAMES, load_configuration
from .execution import DEFAULT_OUTPUT_ROOT, exposure_summary, preparation_status
from .split import build_split_ledger, stream_rows

HERE = Path(__file__).resolve().parent
METRICS = (
    "requirement_satisfaction",
    "completion",
    "deadline_compliance",
    "speed_compliance",
)
ARCHIVED_CURRENT_SEED11 = {
    "source": "completed multistep-physics-mbrl-v1 matched_real_only-seed-11",
    "completed_training_episodes": 222,
    "policy_state_sha256": (
        "9c658cd73d2ac64106a3ee0eeeccd7944ac79f3ce35fe340c0ae446541347dd5"
    ),
    "optimizer_states_sha256": (
        "7b67f6d476eada5585fffd3f55f2451cc44c96c160e3720d7b9ce4847e839813"
    ),
    "lagrangian_states_sha256": (
        "513f8c7d92b597ce2e4b32b0b982237988e703949ef80fcf265fc1f8095c8d3f"
    ),
    "lambda_speed": [
        0.0,
        2.8285030680004364,
        1.105057932559636,
        1.621955983122045,
        2.332436893290417,
        2.141368117462012,
    ],
    "lambda_deadline": [
        9.169187127659349,
        5.64014094516136,
        1.1324206239917256,
        1.3848445840902686,
        8.95064679411747,
        1.6620454648850034,
    ],
    "alpha": [
        0.2232547253370285,
        0.05597502365708351,
        0.015163981355726719,
        0.012467099353671074,
        0.008809450082480907,
        0.0045278193429112434,
    ],
}


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


def _curve_row(diagnostic: dict[str, Any]) -> dict[str, Any]:
    counts = diagnostic["counts"]
    if diagnostic["episode_count"] != 18:
        raise RuntimeError("Every diagnostic checkpoint must contain 18 episodes")
    return {
        "real_transitions": diagnostic["real_transitions"],
        "counts": {name: int(counts[name]) for name in METRICS},
        "rates": {name: counts[name] / 18.0 for name in METRICS},
        "failure_modes": diagnostic["failure_modes"],
        "summary": diagnostic["summary"],
        "feasible_only_energy_kwh": diagnostic["feasible_only_energy_kwh"],
        "training_state": diagnostic["training_state"],
    }


def _load_run(root: Path, name: str, manifest_run: dict[str, Any]) -> dict[str, Any]:
    if manifest_run["state"] != "COMPLETED":
        raise RuntimeError(f"{name} is not complete")
    curves = []
    diagnostic_episode_count = 0
    for checkpoint in CHECKPOINTS:
        diagnostic = _read(
            root / name / f"step-{checkpoint:06d}" / "diagnostic.json"
        )
        curves.append(_curve_row(diagnostic))
        diagnostic_episode_count += int(diagnostic["episode_count"])
    training_rows = _read(root / name / "training-episode-diagnostics.json")
    metadata = manifest_run["last_verified_checkpoint"]
    payload = load_verified(
        metadata["path"],
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256=load_configuration().configuration_sha256,
    )
    recorder = payload["environment_state"]
    if payload["counters"] != {
        "real_transitions": 300_000,
        "rl_gradient_updates": 30_000,
    }:
        raise RuntimeError(f"{name} has incorrect final budgets")
    if len(training_rows) != recorder.completed_episodes:
        raise RuntimeError(f"{name} episode diagnostics are incomplete")
    attempts = manifest_run["attempts"]
    resumes = sum(len(row.get("resume_events", [])) for row in attempts)
    dynamics = [
        {
            key: row[key]
            for key in (
                "episode_index",
                "real_transition_count",
                "episode_duration_steps",
                "speed_episodic_cost",
                "deadline_episodic_cost",
                "lambda_speed_before",
                "lambda_speed_after",
                "lambda_deadline_before",
                "lambda_deadline_after",
                "rl_updates_before_pid",
                "rl_updates_after_pid",
                "rl_updates_after_episode_processing",
                "alpha",
            )
        }
        for row in training_rows
    ]
    return {
        "manifest": manifest_run,
        "attempt_count": len(attempts),
        "resume_count": resumes,
        "diagnostic_episode_count": diagnostic_episode_count,
        "curves": curves,
        "endpoint": curves[-1],
        "training": {
            "completed_episodes": recorder.completed_episodes,
            "pid_update_count": payload["pid_update_count"],
            "task_exposure": exposure_summary(recorder),
            "used_exposures": recorder.used_exposures,
            "episode_dynamics": dynamics,
            "checkpoint_lambda_speed": [
                row["training_state"]["lambda_speed"] for row in curves
            ],
            "checkpoint_lambda_deadline": [
                row["training_state"]["lambda_deadline"] for row in curves
            ],
            "checkpoint_alpha": [
                row["training_state"]["alpha"] for row in curves
            ],
            "walltime_s": payload["walltime_state"]["completed_segments_s"],
        },
        "learned_state_hashes": {
            "policy": stable_state_sha256(payload["policy"]),
            "optimizers": stable_state_sha256(payload["policy_optimizers"]),
            "lagrangian": stable_state_sha256(payload["lagrangian_states"]),
        },
    }


def _contrasts(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    endpoints = {
        key: runs[name]["endpoint"]["counts"]
        for key, name in zip("ABCD", CONDITION_NAMES, strict=True)
    }
    formulas = {
        "track_effect_historical_collector_C_minus_A": ("C", "A"),
        "track_effect_current_collector_D_minus_B": ("D", "B"),
        "collector_effect_historical_stream_B_minus_A": ("B", "A"),
        "collector_effect_current_stream_D_minus_C": ("D", "C"),
    }
    counts = {
        label: {
            metric: endpoints[left][metric] - endpoints[right][metric]
            for metric in METRICS
        }
        for label, (left, right) in formulas.items()
    }
    counts["interaction_D_minus_C_minus_B_plus_A"] = {
        metric: (
            endpoints["D"][metric]
            - endpoints["C"][metric]
            - endpoints["B"][metric]
            + endpoints["A"][metric]
        )
        for metric in METRICS
    }
    return {
        "endpoint_counts": endpoints,
        "counts": counts,
        "normalized": {
            label: {metric: value / 18.0 for metric, value in row.items()}
            for label, row in counts.items()
        },
    }


def _classification(contrasts: dict[str, Any]) -> dict[str, Any]:
    rows = contrasts["counts"]
    track = fmean(
        abs(rows[label][metric])
        for label in (
            "track_effect_historical_collector_C_minus_A",
            "track_effect_current_collector_D_minus_B",
        )
        for metric in METRICS
    )
    collector = fmean(
        abs(rows[label][metric])
        for label in (
            "collector_effect_historical_stream_B_minus_A",
            "collector_effect_current_stream_D_minus_C",
        )
        for metric in METRICS
    )
    interaction = fmean(
        abs(rows["interaction_D_minus_C_minus_B_plus_A"][metric])
        for metric in METRICS
    )
    if max(track, collector, interaction) == 0:
        label = "INCONCLUSIVE"
    elif interaction > 1.1 * max(track, collector):
        label = "INTERACTION_DOMINANT"
    elif track > 1.5 * collector and track > interaction:
        label = "TRACK_DOMINANT"
    elif collector > 1.5 * track and collector > interaction:
        label = "COLLECTOR_DOMINANT"
    elif track > 0 and collector > 0 and interaction <= 0.5 * max(track, collector):
        label = "ADDITIVE"
    else:
        label = "INCONCLUSIVE"
    return {
        "label": label,
        "descriptive_only_single_seed": True,
        "mean_absolute_count_effect": {
            "track": track,
            "collector": collector,
            "interaction": interaction,
        },
        "rule": (
            "interaction > 1.1*max(main effects): INTERACTION_DOMINANT; a main "
            "effect > 1.5*the other and > interaction: corresponding DOMINANT; "
            "interaction <= 0.5*max(main effects) with both nonzero: ADDITIVE; "
            "otherwise INCONCLUSIVE"
        ),
    }


def _recommendation(classification: str) -> dict[str, Any]:
    mapping = {
        "TRACK_DOMINANT": "FIXED_STREAM_POLICY_IS_DOMINANT_CONCERN",
        "COLLECTOR_DOMINANT": "CURRENT_STYLE_COLLECTOR",
        "ADDITIVE": "BOTH_STREAM_GENERATION_AND_COLLECTOR_MUST_BE_CANONICALIZED",
        "INTERACTION_DOMINANT": (
            "BOTH_STREAM_GENERATION_AND_COLLECTOR_MUST_BE_CANONICALIZED"
        ),
        "INCONCLUSIVE": "UNRESOLVED",
    }
    return {
        "regime": mapping[classification],
        "selection_objective": "reproducibility, not retrospective RSR maximization",
        "future_split_rule": "Route-clean Policy R",
    }


def _d_reproduction(d: dict[str, Any]) -> dict[str, Any]:
    archived = ARCHIVED_CURRENT_SEED11
    actual_stream = d["training"]["used_exposures"]
    expected_stream = stream_rows()["T_CURRENT"][: len(actual_stream)]
    stream_match = all(
        actual["reset_seed"] == expected["reset_seed"]
        and actual["route_sha256"] == expected["route_sha256"]
        and actual["full_task_sha256"] == expected["full_task_sha256"]
        for actual, expected in zip(actual_stream, expected_stream, strict=True)
    )
    checks = {
        "initial_hashes_expected": _read(HERE / "initialization.json")["passed"],
        "T_CURRENT_exposure_prefix_exact": stream_match,
        "transition_count_exact": (
            d["manifest"]["real_transitions"] == 300_000
        ),
        "update_count_exact": (
            d["manifest"]["rl_gradient_updates"] == 30_000
        ),
        "completed_episode_count_exact": (
            d["training"]["completed_episodes"]
            == archived["completed_training_episodes"]
        ),
        "pid_update_count_exact": (
            d["training"]["pid_update_count"]
            == archived["completed_training_episodes"]
        ),
        "lambda_speed_trajectory_exact": (
            d["training"]["checkpoint_lambda_speed"] == archived["lambda_speed"]
        ),
        "lambda_deadline_trajectory_exact": (
            d["training"]["checkpoint_lambda_deadline"]
            == archived["lambda_deadline"]
        ),
        "alpha_trajectory_exact": (
            d["training"]["checkpoint_alpha"] == archived["alpha"]
        ),
        "final_policy_state_exact": (
            d["learned_state_hashes"]["policy"]
            == archived["policy_state_sha256"]
        ),
        "final_optimizer_states_exact": (
            d["learned_state_hashes"]["optimizers"]
            == archived["optimizer_states_sha256"]
        ),
        "final_lagrangian_states_exact": (
            d["learned_state_hashes"]["lagrangian"]
            == archived["lagrangian_states_sha256"]
        ),
    }
    return {
        "archive": archived,
        "checks": checks,
        "status": (
            "EXACT_CURRENT_REGIME_REPRODUCTION"
            if all(checks.values())
            else "DIVERGED"
        ),
        "diagnostic_endpoint_not_compared": (
            "V3 uses a new route-clean set; archived Validation performance is not "
            "an equality target"
        ),
    }


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_results(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    preparation = preparation_status()
    root = Path(output_root)
    manifest = _read(root / "manifest.json")
    if tuple(manifest["runs"]) != CONDITION_NAMES:
        raise RuntimeError("Manifest condition order changed")
    if manifest["paper"] != {"state": "SEALED_UNINSPECTED", "episode_count": 0}:
        raise RuntimeError("Paper was not kept sealed")
    if manifest["legacy_development"]["episode_count"] != 0 or manifest[
        "legacy_validation"
    ]["episode_count"] != 0:
        raise RuntimeError("A forbidden legacy split was evaluated")
    runs = {
        name: _load_run(root, name, manifest["runs"][name])
        for name in CONDITION_NAMES
    }
    diagnostic_episode_count = sum(
        row["diagnostic_episode_count"] for row in runs.values()
    )
    if diagnostic_episode_count != 432:
        raise RuntimeError("The matrix must contain exactly 432 diagnostic episodes")
    contrasts = _contrasts(runs)
    classification = _classification(contrasts)
    d_name = CONDITION_NAMES[3]
    a_name = CONDITION_NAMES[0]
    return {
        "schema_version": 1,
        "study": load_configuration().raw["name"],
        "outcome": "COMPLETED",
        "preparation": preparation,
        "split": build_split_ledger(),
        "manifest": manifest,
        "runs": runs,
        "execution_totals": {
            "attempts": sum(row["attempt_count"] for row in runs.values()),
            "resumes": sum(row["resume_count"] for row in runs.values()),
            "real_transitions": sum(
                row["manifest"]["real_transitions"] for row in runs.values()
            ),
            "rl_gradient_updates": sum(
                row["manifest"]["rl_gradient_updates"] for row in runs.values()
            ),
            "diagnostic_episodes": diagnostic_episode_count,
            "legacy_development_episodes": 0,
            "legacy_validation_episodes": 0,
            "paper_episodes": 0,
            "sum_policy_walltime_s": sum(
                row["training"]["walltime_s"] for row in runs.values()
            ),
            "retained_bytes": _directory_bytes(root),
        },
        "factorial_contrasts": contrasts,
        "qualitative_classification": classification,
        "canonical_baseline_recommendation": _recommendation(
            classification["label"]
        ),
        "D_current_regime_reproduction": _d_reproduction(runs[d_name]),
        "A_historical_direction": {
            "status": (
                "RECONSTRUCTED_DIRECTIONAL_REGIME_"
                "NOT_EXACT_HISTORICAL_REPRODUCTION"
            ),
            "endpoint": runs[a_name]["endpoint"],
            "reason": (
                "Historical runtime and complete initial/RNG provenance were never "
                "retained; A is a current-runtime reconstruction."
            ),
        },
        "integrity": {
            "all_final_values_finite": all(
                math.isfinite(value)
                for row in runs.values()
                for curve in row["curves"]
                for value in (
                    curve["training_state"]["lambda_speed"],
                    curve["training_state"]["lambda_deadline"],
                    curve["training_state"]["alpha"],
                )
            ),
            "paper": "SEALED_UNINSPECTED",
            "only_route_clean_diagnostic_evaluated": True,
        },
    }


def _counts_text(row: dict[str, int]) -> str:
    return " / ".join(str(row[key]) for key in METRICS)


def render_markdown(results: dict[str, Any]) -> str:
    names = list(CONDITION_NAMES)
    endpoint_rows = "\n".join(
        f"| {name[0]} | {_counts_text(results['runs'][name]['endpoint']['counts'])} |"
        for name in names
    )
    curve_rows = []
    for name in names:
        curve_rows.append(
            f"| {name[0]} | "
            + " | ".join(
                str(row["counts"]["requirement_satisfaction"])
                for row in results["runs"][name]["curves"]
            )
            + " |"
        )
    attempts = results["execution_totals"]
    return f"""# V3 results

## Outcome

`COMPLETED`

All four seed-11 policies reached exactly 300,000 real transitions and 30,000 RL
updates. Only the frozen 18-task route-clean Diagnostic set was evaluated. Paper
remained sealed; legacy Development and Validation received zero policy episodes.

## Primary 300k endpoint

Counts are RSR / completion / deadline / speed, each out of 18.

| Condition | Counts |
|---|---:|
{endpoint_rows}

## Diagnostic RSR curves

| Condition | 50k | 100k | 150k | 200k | 250k | 300k |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(curve_rows)}

## Interpretation

- Factorial classification: `{results['qualitative_classification']['label']}`.
- Canonical baseline recommendation:
  `{results['canonical_baseline_recommendation']['regime']}`.
- D reproduction: `{results['D_current_regime_reproduction']['status']}`.
- A is a reconstructed historical-direction regime, not an exact historical run.
- These are descriptive single-seed contrasts; no significance claim is made.
- Future evaluation must follow `Route-clean Policy R`.

## Execution integrity

- Attempts: {attempts['attempts']}.
- Exact resumes: {attempts['resumes']}.
- Real transitions: {attempts['real_transitions']}.
- RL gradient updates: {attempts['rl_gradient_updates']}.
- Diagnostic episodes: {attempts['diagnostic_episodes']}.
- Legacy Development / Validation / Paper episodes: 0 / 0 / 0.

Machine-readable curves, endpoint episodes and failure modes, feasible-only energy,
episode-level PID/lambda/alpha trajectories, exposure statistics, normalized factorial
contrasts, reproduction anchors, and integrity evidence are in `results.json`.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)
    results = build_results(args.output_root)
    from .execution import atomic_json

    atomic_json(HERE / "results.json", results)
    _atomic_markdown(HERE / "RESULTS.md", render_markdown(results))
    print(json.dumps(results["execution_totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
