"""Create deterministic plots from credit-assignment analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate_plots(
    analysis_path: str | Path, output_directory: str | Path
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    curves = analysis["learning_curves"]
    condition_ids = sorted({row["condition_id"] for row in curves})
    labels = {
        row["condition_id"]: row["condition_label"] for row in curves
    }

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, condition_id in zip(axes.flat, condition_ids):
        rows = [item for item in curves if item["condition_id"] == condition_id]
        for seed in sorted({item["training_seed"] for item in rows}):
            seed_rows = [item for item in rows if item["training_seed"] == seed]
            axis.plot(
                [item["simulator_steps"] for item in seed_rows],
                [item["rsr"] for item in seed_rows],
                marker="o",
                label=f"seed {seed}",
            )
        axis.set_title(labels[condition_id])
        axis.set_ylim(-0.02, 1.02)
        axis.set_ylabel("validation RSR")
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    for axis in axes[-1]:
        axis.set_xlabel("simulator transitions")
    figure.suptitle("Credit-assignment validation learning curves")
    figure.tight_layout()
    destination = output / "validation-rsr-vs-simulator-steps.png"
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    paths.append(destination)

    final_target = max(item["simulator_step_target"] for item in curves)
    summaries = {
        condition_id: analysis["summaries"][
            f"{condition_id}/validation-3000-3008-v1/"
            f"simulator-target-{final_target}"
        ]
        for condition_id in condition_ids
    }
    modes = sorted(
        {
            mode
            for summary in summaries.values()
            for mode in summary["failure_mode_counts"]
        },
        key=lambda item: (item != "feasible", item),
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    x_values = np.arange(len(condition_ids))
    bottoms = np.zeros(len(condition_ids))
    for mode in modes:
        values = np.array(
            [
                summaries[item]["failure_mode_counts"].get(mode, 0)
                / summaries[item]["episode_count"]
                for item in condition_ids
            ]
        )
        axis.bar(x_values, values, bottom=bottoms, label=mode)
        bottoms += values
    axis.set_xticks(x_values, [labels[item] for item in condition_ids])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("validation episode proportion")
    axis.set_title("Exclusive failure modes at 300k simulator transitions")
    axis.legend(fontsize="small")
    figure.tight_layout()
    destination = output / "failure-modes-300k.png"
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    paths.append(destination)

    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.22
    seeds = sorted(
        {
            int(seed)
            for summary in summaries.values()
            for seed in summary["by_training_seed"]
        }
    )
    for index, seed in enumerate(seeds):
        values = [
            summaries[item]["by_training_seed"][str(seed)][
                "requirement_satisfaction_rate"
            ]
            for item in condition_ids
        ]
        axis.bar(
            x_values + (index - 1) * width,
            values,
            width=width,
            label=f"seed {seed}",
        )
    axis.set_xticks(x_values, [labels[item] for item in condition_ids])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("validation RSR")
    axis.set_title("Training-seed reliability at 300k")
    axis.legend(fontsize="small")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    destination = output / "seed-reliability-300k.png"
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    paths.append(destination)

    discount = analysis["effective_discount"]
    figure, axis = plt.subplots(figsize=(8, 5))
    for row in discount:
        delays = [int(value) for value in row["weights"]]
        weights = [row["weights"][str(value)] for value in delays]
        axis.semilogy(delays, weights, marker="o", label=labels[row["condition_id"]])
    axis.set_xlabel("physical delay [s]")
    axis.set_ylabel("discount weight")
    axis.set_title("Effective real-time discounting")
    axis.grid(alpha=0.2, which="both")
    axis.legend(fontsize="small")
    figure.tight_layout()
    destination = output / "effective-real-time-discount.png"
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    paths.append(destination)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in generate_plots(args.analysis, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
