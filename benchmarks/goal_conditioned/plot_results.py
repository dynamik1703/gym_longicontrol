"""Create reproducible plots for the Goal/HER comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(figure, output: Path, filename: str) -> Path:
    path = output / filename
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return path


def generate_plots(results_path: str | Path, output_directory: str | Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = _read(results_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    seeds = (11, 29, 47)
    conditions = ("sac-no-her", "sac-her")
    labels = {"sac-no-her": "SAC without HER", "sac-her": "SAC + HER"}
    colors = {11: "tab:blue", 29: "tab:orange", 47: "tab:green"}
    paths = []

    figure, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row_index, condition in enumerate(conditions):
        for column_index, seed in enumerate(seeds):
            axis = axes[row_index, column_index]
            rows = [
                row
                for row in results["development_curves"]
                if row["condition_id"] == condition
                and row["training_seed"] == seed
            ]
            axis.plot(
                [row["training_transitions"] for row in rows],
                [row["requirement_satisfaction_rate"] for row in rows],
                color=colors[seed],
                marker="o",
            )
            axis.set_title(f"{labels[condition]} / seed {seed}")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Development RSR")
            if row_index == 1:
                axis.set_xlabel("simulator transitions")
    paths.append(_save(figure, output, "development-rsr.png"))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_values = np.arange(len(conditions))
    metric_names = (
        ("requirement_satisfaction_rate", "RSR"),
        ("completion_rate", "completion"),
        ("completed_by_deadline_rate", "by deadline"),
        ("speed_compliance_rate", "speed compliant"),
    )
    width = 0.18
    for index, (key, label) in enumerate(metric_names):
        values = [
            results["final_validation"][condition]["overall"][key]
            for condition in conditions
        ]
        axes[0].bar(
            x_values + (index - 1.5) * width,
            values,
            width,
            label=label,
        )
    axes[0].set_xticks(x_values, [labels[item] for item in conditions])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Validation rate")
    axes[0].set_title("Final canonical requirements")
    axes[0].legend(fontsize="small")
    axes[0].grid(axis="y", alpha=0.2)

    modes = sorted(
        {
            mode
            for condition in conditions
            for mode in results["final_validation"][condition]["overall"][
                "failure_mode_counts"
            ]
        }
    )
    bottom = np.zeros(len(conditions))
    for mode in modes:
        values = np.asarray(
            [
                results["final_validation"][condition]["overall"][
                    "failure_mode_counts"
                ].get(mode, 0)
                / 27
                for condition in conditions
            ]
        )
        axes[1].bar(x_values, values, bottom=bottom, label=mode.replace("+", " + "))
        bottom += values
    axes[1].set_xticks(x_values, [labels[item] for item in conditions])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("Validation proportion")
    axes[1].set_title("Mutually exclusive outcomes")
    axes[1].legend(fontsize="small")
    paths.append(_save(figure, output, "final-validation-outcomes.png"))
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_values = np.arange(len(seeds))
    width = 0.36
    for index, condition in enumerate(conditions):
        offset = (index - 0.5) * width
        real_successes = [
            results["training_outcomes"][condition]["by_training_seed"][str(seed)][
                "canonical_successful_episode_count"
            ]
            for seed in seeds
        ]
        bars = axes[0].bar(
            x_values + offset,
            real_successes,
            width,
            label=labels[condition],
        )
        axes[0].bar_label(bars)
        positive_virtual_rates = [
            results["replay_diagnostics"][condition]["by_training_seed"][str(seed)][
                "rates"
            ]["positive_virtual_reward_rate"]
            or 0.0
            for seed in seeds
        ]
        bars = axes[1].bar(
            x_values + offset,
            np.asarray(positive_virtual_rates) * 100,
            width,
            label=labels[condition],
        )
        axes[1].bar_label(bars, fmt="%.3f%%", fontsize="small")
    axes[0].set_xticks(x_values, [f"seed {seed}" for seed in seeds])
    axes[0].set_ylabel("real canonical training successes")
    axes[0].set_title("Real rollout success")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize="small")
    axes[1].set_xticks(x_values, [f"seed {seed}" for seed in seeds])
    axes[1].set_ylabel("positive virtual rows [% of virtual rows]")
    axes[1].set_title("HER replay signal")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize="small")
    paths.append(_save(figure, output, "training-and-replay-signal.png"))
    plt.close(figure)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/goal_conditioned/results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/goal_conditioned/plots"),
    )
    args = parser.parse_args()
    for path in generate_plots(args.results, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
