"""Generate headless Scalar V2 plots from a persisted analysis JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_summary_key(key: str):
    split, step, configuration_id = key.split("/")
    return split, int(step.removeprefix("step-")), configuration_id


def generate_v2_plots(
    analysis_path: str | Path, output_directory: str | Path
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    summaries = {
        _parse_summary_key(key): value
        for key, value in analysis["summaries"].items()
    }
    validation_id = "validation-3000-3008-v1"
    exploratory_id = "v1-exploratory-1000-1008-v1"
    identifiers = sorted(
        identifier
        for split, _step, identifier in summaries
        if split == validation_id
    )
    identifiers = sorted(set(identifiers))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(identifiers), 1, figsize=(8, 3 * len(identifiers)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, identifier in zip(axes, identifiers):
        rows = sorted(
            (step, value)
            for (split, step, item), value in summaries.items()
            if split == validation_id and item == identifier
        )
        steps = [step for step, _ in rows]
        seed_ids = sorted(rows[0][1]["by_training_seed"], key=int)
        for seed in seed_ids:
            axis.plot(
                steps,
                [
                    row["by_training_seed"][seed][
                        "requirement_satisfaction_rate"
                    ]
                    for _, row in rows
                ],
                marker="o",
                alpha=0.55,
                label=f"seed {seed}",
            )
        axis.plot(
            steps,
            [row["mean_rsr"] for _, row in rows],
            color="black",
            linewidth=2.5,
            marker="s",
            label="seed mean",
        )
        axis.set_title(identifier)
        axis.set_ylabel("validation RSR")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small", ncols=4)
    axes[-1].set_xlabel("online training steps")
    figure.suptitle("Validation learning curves (training seeds remain separate)")
    figure.tight_layout()
    curve_path = output / "validation-learning-curves.png"
    figure.savefig(curve_path, dpi=140)
    plt.close(figure)

    comparison_steps = sorted(
        {
            step
            for split, step, _identifier in summaries
            if split == exploratory_id
        }
    )
    x_values = np.arange(len(identifiers))
    width = 0.35
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for index, step in enumerate(comparison_steps):
        means = [
            summaries[(exploratory_id, step, identifier)]["mean_rsr"]
            for identifier in identifiers
        ]
        positions = x_values + (index - (len(comparison_steps) - 1) / 2) * width
        axis.bar(positions, means, width=width, alpha=0.7, label=f"{step // 1000}k")
        for position, identifier in zip(positions, identifiers):
            row = summaries[(exploratory_id, step, identifier)]
            axis.scatter(
                [position] * len(row["by_training_seed"]),
                [
                    seed["requirement_satisfaction_rate"]
                    for seed in row["by_training_seed"].values()
                ],
                color="black",
                s=22,
                zorder=3,
            )
    axis.set_xticks(x_values, identifiers, rotation=25, ha="right")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("exploratory RSR")
    axis.set_title("100k versus 300k with individual training seeds")
    axis.legend()
    figure.tight_layout()
    budget_path = output / "exploratory-budget-comparison.png"
    figure.savefig(budget_path, dpi=140)
    plt.close(figure)

    final_step = max(comparison_steps)
    mode_counts = {
        identifier: summaries[(exploratory_id, final_step, identifier)][
            "failure_mode_counts"
        ]
        for identifier in identifiers
    }
    modes = sorted(
        {mode for counts in mode_counts.values() for mode in counts},
        key=lambda mode: (mode != "feasible", mode),
    )
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bottoms = np.zeros(len(identifiers))
    for mode in modes:
        rates = np.array(
            [
                mode_counts[identifier].get(mode, 0)
                / sum(mode_counts[identifier].values())
                for identifier in identifiers
            ]
        )
        axis.bar(x_values, rates, bottom=bottoms, label=mode)
        bottoms += rates
    axis.set_xticks(x_values, identifiers, rotation=25, ha="right")
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("episode proportion")
    axis.set_title(f"Exclusive failure modes at {final_step // 1000}k")
    axis.legend(fontsize="small")
    figure.tight_layout()
    failure_path = output / "failure-modes-300k.png"
    figure.savefig(failure_path, dpi=140)
    plt.close(figure)
    return curve_path, budget_path, failure_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in generate_v2_plots(args.analysis, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
