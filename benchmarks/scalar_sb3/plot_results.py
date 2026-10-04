"""Create deterministic plots from the persisted SB3 analysis summary."""

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
    algorithms = ("custom-sac-v2b", "sac", "ppo")
    titles = {
        "custom-sac-v2b": "Historical custom SAC (V2-B)",
        "sac": "SB3 SAC",
        "ppo": "SB3 PPO",
    }
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, sharey=True)
    for axis, algorithm in zip(axes, algorithms):
        rows = [row for row in curves if row["algorithm"] == algorithm]
        seeds = sorted({row["training_seed"] for row in rows})
        for seed in seeds:
            seed_rows = sorted(
                (row for row in rows if row["training_seed"] == seed),
                key=lambda row: row["training_steps"],
            )
            axis.plot(
                [row["training_steps"] for row in seed_rows],
                [row["rsr"] for row in seed_rows],
                marker="o",
                label=f"seed {seed}",
            )
        axis.set_title(titles[algorithm])
        axis.set_ylabel("validation RSR")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small", ncols=3)
    axes[-1].set_xlabel("environment interactions")
    figure.suptitle("External validation learning stability")
    figure.tight_layout()
    curve_path = output / "validation-rsr-learning-curves.png"
    figure.savefig(curve_path, dpi=140)
    plt.close(figure)
    paths.append(curve_path)

    final_step = max(
        row["training_steps"]
        for row in curves
        if row["algorithm"] in {"sac", "ppo"}
    )
    final_summaries = {
        algorithm: analysis["summaries"][
            f"{algorithm}/validation-3000-3008-v1/step-{final_step}"
        ]
        for algorithm in ("sac", "ppo")
    }
    modes = sorted(
        {
            mode
            for summary in final_summaries.values()
            for mode in summary["failure_mode_counts"]
        },
        key=lambda item: (item != "feasible", item),
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    x_values = np.arange(2)
    bottoms = np.zeros(2)
    for mode in modes:
        values = np.array(
            [
                final_summaries[algorithm]["failure_mode_counts"].get(mode, 0)
                / final_summaries[algorithm]["episode_count"]
                for algorithm in ("sac", "ppo")
            ]
        )
        axis.bar(x_values, values, bottom=bottoms, label=mode)
        bottoms += values
    axis.set_xticks(x_values, ("SB3 SAC", "SB3 PPO"))
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("validation episode proportion")
    axis.set_title("Exclusive failure modes at 300k")
    axis.legend(fontsize="small")
    figure.tight_layout()
    failure_path = output / "failure-modes-300k.png"
    figure.savefig(failure_path, dpi=140)
    plt.close(figure)
    paths.append(failure_path)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, algorithm in enumerate(("sac", "ppo")):
        key = f"{algorithm}/validation-3000-3008-v1/step-{final_step}"
        by_seed = analysis["summaries"][key]["by_training_seed"]
        values = [
            summary["requirement_satisfaction_rate"]
            for summary in by_seed.values()
        ]
        axis.scatter(
            [index] * len(values),
            values,
            s=55,
            label=algorithm.upper(),
        )
    axis.set_xticks((0, 1), ("SB3 SAC", "SB3 PPO"))
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("RSR by training seed")
    axis.set_title("Final seed reliability")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    reliability_path = output / "seed-reliability-300k.png"
    figure.savefig(reliability_path, dpi=140)
    plt.close(figure)
    paths.append(reliability_path)
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
