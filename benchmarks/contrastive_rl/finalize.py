"""Build compact final artifacts from the completed projected-goal CRL run."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import PLANNED_DEPTHS, PLANNED_SEEDS
from .execution import DEFAULT_OUTPUT_ROOT, EXPECTED_CONFIGURATION_SHA256, policy_key

HISTORICAL = {
    "Scalar SB3 SAC": {"successes": 5, "episodes": 27},
    "Action-repeat scalar": {"successes": 9, "episodes": 27},
    "Constrained V2": {"successes": 21, "episodes": 27},
    "Requirement-conditioned, canonical 140 s": {"successes": 12, "episodes": 27},
    "Binary Success": {"successes": 0, "episodes": 27},
    "LLM reward": {"successes": 1, "episodes": 27},
    "Goal-conditioned SAC, no HER": {"successes": 0, "episodes": 27},
    "Goal-conditioned SAC + HER": {"successes": 0, "episodes": 27},
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _mean(rows, getter):
    values = [getter(row) for row in rows]
    return float(np.mean(values)) if values else None


def _diagnostic_summary(raw):
    rows = raw["rows"]
    batches = [row["batch"] for row in rows]
    support = [row["exact_projected_goal_support"] for row in batches]
    detailed = [row for row in rows if row["learning"]["canonical_query"] is not None]
    requirement_keys = ("star-1-1", "star-0-1", "star-1-0", "star-0-0")
    return {
        **raw["summary"],
        "sampled_source_future_pairs": sum(
            row["sampled_pair_count"] for row in batches
        ),
        "mean_future_lag_decisions": _mean(
            batches, lambda x: x["future_lag_decisions"]["mean"]
        ),
        "mean_future_lag_seconds": _mean(
            batches, lambda x: x["future_lag_seconds"]["mean"]
        ),
        "mean_target_progress_distance": _mean(
            batches, lambda x: x["target_progress_distance"]["mean"]
        ),
        "mean_terminal_future_fraction": _mean(
            batches, lambda x: x["terminal_future_fraction"]
        ),
        "mean_late_future_fraction": _mean(
            batches, lambda x: x["late_future_fraction"]
        ),
        "mean_unsafe_future_fraction": _mean(
            batches, lambda x: x["unsafe_future_fraction"]
        ),
        "mean_timely_safe_future_fraction": _mean(
            batches, lambda x: x["timely_safe_future_fraction"]
        ),
        "mean_exact_unique_projected_goals": _mean(
            support, lambda x: x["unique_goal_count"]
        ),
        "mean_exact_duplicate_fraction": _mean(
            support, lambda x: x["duplicate_column_fraction"]
        ),
        "mean_pair_collision_rate": _mean(support, lambda x: x["pair_collision_rate"]),
        "mean_distinct_progress_values": _mean(
            support, lambda x: x["distinct_progress_count"]
        ),
        "mean_requirement_bit_frequencies": {
            key: _mean(
                support, lambda x, key=key: x["requirement_bit_frequencies"][key]
            )
            for key in requirement_keys
        },
        "minimum_goal_distance_to_canonical": min(
            row["minimum_goal_distance_to_canonical"] for row in batches
        ),
        "final_learning": rows[-1]["learning"],
        "final_detailed_learning": detailed[-1]["learning"],
        "detailed_diagnostic_count": len(detailed),
    }


def _aggregate_validation(episodes):
    feasible = [row for row in episodes if row["feasible"]]
    failures = Counter(row["failure_mode"] for row in episodes)
    incomplete = [
        row["final_position_m"] / 1000.0 for row in episodes if not row["completed"]
    ]
    return {
        "episode_count": len(episodes),
        "success_count": len(feasible),
        "requirement_satisfaction_rate": len(feasible) / len(episodes),
        "completion_count": sum(row["completed"] for row in episodes),
        "completed_by_deadline_count": sum(
            row["completed"] and row["travel_time_s"] <= 140.0 for row in episodes
        ),
        "speed_compliant_count": sum(
            row["max_speed_violation_m_s"] <= 0.0 for row in episodes
        ),
        "failure_mode_counts": dict(sorted(failures.items())),
        "incomplete_terminal_progress": incomplete,
        "mean_incomplete_terminal_progress": (
            float(np.mean(incomplete)) if incomplete else None
        ),
        "feasible_energy_count": len(feasible),
        "mean_feasible_energy_kwh": (
            float(np.mean([row["energy_kwh"] for row in feasible]))
            if feasible
            else None
        ),
    }


def build(repo_root: Path) -> dict:
    output = repo_root / "benchmarks/contrastive_rl"
    run_root = repo_root / DEFAULT_OUTPUT_ROOT
    manifest = _read(run_root / "manifest.json")
    measurements = _read(output / "resource_measurements_projected.json")
    parameter_counts = {
        row["depth_inside_residual_blocks"]: row["parameters"]["total_trainable"]
        for row in measurements["depths"]
    }
    if (
        manifest["status"] != "COMPLETED"
        or manifest["validation"]["status"] != "COMPLETED"
    ):
        raise RuntimeError("training and the single Validation pass must be complete")
    if manifest["validation"]["episode_count"] != 54 or manifest["paper_tracks_opened"]:
        raise RuntimeError("final split accounting is invalid")

    policies = []
    validation_episodes = _read(run_root / manifest["validation"]["episodes_path"])
    for depth in PLANNED_DEPTHS:
        for seed in PLANNED_SEEDS:
            item = manifest["policies"][policy_key(depth, seed)]
            directory = run_root / f"depth-{depth}" / f"seed-{seed}"
            training = _read(directory / "training-outcomes.json")
            diagnostics = _read(directory / "crl-diagnostics.json")
            validation = _read(directory / "validation-result.json")
            development = []
            for checkpoint in item["development_checkpoints"]:
                result = _read(run_root / checkpoint["development_result_path"])
                development.append(
                    {
                        "transition_count": checkpoint["transition_count"],
                        **result["summary"],
                    }
                )
            training.pop("episodes", None)
            policies.append(
                {
                    "depth": depth,
                    "training_seed": seed,
                    "parameter_count": parameter_counts[depth],
                    "native_transitions": item["native_transitions"],
                    "complete_update_cycles": item["complete_update_cycles"],
                    "attempt_count": len(item["attempts"]),
                    "interruption_count": sum(
                        attempt["status"] != "COMPLETED" for attempt in item["attempts"]
                    ),
                    "started_at": item["started_at"],
                    "completed_at": item["completed_at"],
                    "wall_clock_s": (
                        datetime.fromisoformat(item["completed_at"])
                        - datetime.fromisoformat(item["started_at"])
                    ).total_seconds(),
                    "configuration_sha256": item["configuration_sha256"],
                    "scientific_source_sha256": item["scientific_source_sha256"],
                    "execution_source_sha256": item["execution_source_sha256"],
                    "final_checkpoint_sha256": item["final_checkpoint_sha256"],
                    "training": training,
                    "development": development,
                    "diagnostics": _diagnostic_summary(diagnostics),
                    "validation": validation["summary"],
                }
            )

    depth_results = {}
    for depth in PLANNED_DEPTHS:
        rows = [row for row in validation_episodes if row["depth"] == depth]
        aggregate = _aggregate_validation(rows)
        aggregate["per_seed_success"] = {
            str(seed): sum(
                row["feasible"] for row in rows if row["training_seed"] == seed
            )
            for seed in PLANNED_SEEDS
        }
        depth_results[str(depth)] = aggregate

    created = datetime.fromisoformat(manifest["created_at"])
    completed = datetime.fromisoformat(manifest["completed_at"])
    validation_completed = datetime.fromisoformat(
        manifest["validation"]["completed_at"]
    )
    result = {
        "status": "COMPLETED",
        "canonical_configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
        "study_id": manifest["study_id"],
        "training_wall_clock_s": (completed - created).total_seconds(),
        "total_wall_clock_through_validation_s": (
            validation_completed - created
        ).total_seconds(),
        "accounting": manifest["actual_resources_across_attempts"],
        "validation_episode_count": 54,
        "paper_tracks_used": False,
        "depth_results": depth_results,
        "policies": policies,
        "historical_context": HISTORICAL,
        "interpretation_scope": (
            "Low-budget 300k-per-policy projected-goal CRL adaptation; depth 4 "
            "versus depth 16 is the strongest matched comparison."
        ),
    }
    _write(output / "results.json", result)
    _write(output / "validation_episodes.json", validation_episodes)
    _write(output / "execution_manifest.json", manifest)
    return result


def plots(repo_root: Path, result: dict) -> None:
    output = repo_root / "benchmarks/contrastive_rl/plots"
    output.mkdir(exist_ok=True)
    colors = {4: "#2878B5", 16: "#D95319"}
    line_styles = {11: "-", 29: "--", 47: ":"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy in result["policies"]:
        steps = [row["transition_count"] for row in policy["development"]]
        rates = [row["requirement_satisfaction_rate"] for row in policy["development"]]
        ax.plot(
            steps,
            rates,
            marker="o",
            color=colors[policy["depth"]],
            linestyle=line_styles[policy["training_seed"]],
            alpha=0.75,
            label=f"d{policy['depth']} s{policy['training_seed']}",
        )
    ax.set(
        xlabel="Native training transitions",
        ylabel="Development RSR",
        ylim=(-0.02, 1.02),
    )
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "development-rsr.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [f"d{p['depth']} s{p['training_seed']}" for p in result["policies"]]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(
        x - width,
        [p["validation"]["success_count"] for p in result["policies"]],
        width,
        label="feasible",
    )
    ax.bar(
        x,
        [p["validation"]["completion_count"] for p in result["policies"]],
        width,
        label="completed",
    )
    ax.bar(
        x + width,
        [p["validation"]["speed_compliant_count"] for p in result["policies"]],
        width,
        label="speed compliant",
    )
    ax.set(ylabel="Episodes out of 9", xticks=x, xticklabels=labels, ylim=(0, 9.5))
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "validation-requirements.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for policy in result["policies"]:
        d = policy["diagnostics"]
        ax.scatter(
            d["mean_exact_duplicate_fraction"],
            d["final_detailed_learning"]["contrastive"][
                "positive_minus_reference_mean"
            ],
            s=65,
            color=colors[policy["depth"]],
            label=f"d{policy['depth']} s{policy['training_seed']}",
        )
    ax.set(
        xlabel="Mean exact duplicate-goal fraction",
        ylabel="Final detailed positive-reference score gap",
    )
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "collision-vs-score-gap.png", dpi=160)
    plt.close(fig)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    result = build(root)
    plots(root, result)
    print(root / "benchmarks/contrastive_rl/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
