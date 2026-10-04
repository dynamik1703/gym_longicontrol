"""Create reproducible Binary Success Reward V1 plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(figure, output, filename):
    path = Path(output) / filename
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return path


def generate_result_plots(analysis_path, results_directory, output_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = _read(analysis_path)
    root = Path(results_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    seeds = sorted(
        int(value) for value in analysis["final_validation"]["by_training_seed"]
    )

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for seed in seeds:
        rows = [
            row for row in analysis["learning_curves"] if row["training_seed"] == seed
        ]
        axes[0].plot(
            [row["training_steps"] for row in rows],
            [row["rsr"] for row in rows],
            marker="o",
            label=f"binary seed {seed}",
        )
    scalar = analysis["frozen_comparisons"]["scalar_sb3_sac"]["learning_curves"]
    for seed in seeds:
        rows = [row for row in scalar if row["training_seed"] == seed]
        axes[1].plot(
            [row["training_steps"] for row in rows],
            [row["rsr"] for row in rows],
            marker="o",
            label=f"scalar seed {seed}",
        )
    axes[0].set_title("Binary terminal reward: Validation learning curves")
    axes[1].set_title("Frozen shaped Scalar SB3 SAC")
    axes[1].set_xlabel("environment interactions")
    for axis in axes:
        axis.set_ylabel("Validation RSR")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small", ncols=3)
    paths.append(_save(figure, output, "validation-rsr-learning-curves.png"))
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    checkpoints = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
    width = 12_000
    offsets = (-width, 0, width)
    for offset, seed in zip(offsets, seeds):
        summary = analysis["training_outcomes"]["by_training_seed"][str(seed)]
        counts = summary["successes_by_50k_interval"]
        axes[0].bar(
            np.asarray(checkpoints) + offset,
            [counts[str(step)] for step in checkpoints],
            width,
            label=f"seed {seed}",
        )
        cumulative = summary["cumulative_successes_by_checkpoint"]
        axes[1].plot(
            checkpoints,
            [cumulative[str(step)] for step in checkpoints],
            marker="o",
            label=f"seed {seed}",
        )
    axes[0].set_title("Successful training episodes by 50k interval")
    axes[0].set_ylabel("success count")
    axes[1].set_ylabel("cumulative successes")
    axes[1].set_xlabel("environment interactions")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    paths.append(_save(figure, output, "training-success-onset.png"))
    plt.close(figure)

    metrics = (
        ("requirement_satisfaction_rate", "RSR"),
        ("completion_rate", "completion"),
        ("deadline_compliance_rate", "deadline"),
        ("speed_compliance_rate", "speed"),
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    x_values = np.arange(len(seeds))
    width = 0.2
    for index, (key, label) in enumerate(metrics):
        axis.bar(
            x_values + (index - 1.5) * width,
            [
                analysis["final_validation"]["by_training_seed"][str(seed)][key]
                for seed in seeds
            ],
            width,
            label=label,
        )
    axis.set_xticks(x_values, [f"seed {seed}" for seed in seeds])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("final Validation rate")
    axis.set_title("Physical requirement breakdown at 300k")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncols=4, fontsize="small")
    paths.append(_save(figure, output, "final-requirement-breakdown.png"))
    plt.close(figure)

    modes = sorted(
        {
            mode
            for row in analysis["final_validation"]["by_training_seed"].values()
            for mode in row["failure_mode_counts"]
        }
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(seeds))
    for mode in modes:
        values = np.asarray(
            [
                analysis["final_validation"]["by_training_seed"][str(seed)][
                    "failure_mode_counts"
                ].get(mode, 0)
                / 9
                for seed in seeds
            ]
        )
        axis.bar(seeds, values, bottom=bottoms, label=mode.replace("+", " + "))
        bottoms += values
    axis.set_xticks(seeds, [f"seed {seed}" for seed in seeds])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("final Validation proportion")
    axis.set_title("Final physical outcomes and dominant failure modes")
    axis.legend(fontsize="small", ncols=2)
    paths.append(_save(figure, output, "final-failure-modes.png"))
    plt.close(figure)

    comparisons = analysis["frozen_comparisons"]
    labels = ("Binary V1", "Scalar SB3", "Constrained V2")
    counts = (
        analysis["final_validation"]["overall"]["feasible_count"],
        comparisons["scalar_sb3_sac"]["feasible_count"],
        comparisons["constrained_v2"]["feasible_count"],
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, np.asarray(counts) / 27)
    axis.bar_label(bars, labels=[f"{value}/27" for value in counts])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("final Validation RSR")
    axis.set_title("Task-specification comparison at 300k")
    axis.grid(axis="y", alpha=0.2)
    paths.append(_save(figure, output, "frozen-method-comparison.png"))
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for seed in seeds:
        rows = _read(root / f"training-seed-{seed}" / "training-diagnostics.json")
        x_values = [row["training_steps"] for row in rows]
        for axis, key in zip(
            axes,
            ("train/actor_loss", "train/critic_loss", "train/ent_coef"),
        ):
            filtered = [(x, row[key]) for x, row in zip(x_values, rows) if key in row]
            axis.plot(
                [item[0] for item in filtered],
                [item[1] for item in filtered],
                label=f"seed {seed}",
            )
    for axis, label in zip(axes, ("actor loss", "critic loss", "entropy alpha")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    axes[0].set_title("SB3 SAC optimization diagnostics")
    axes[-1].set_xlabel("environment interactions")
    paths.append(_save(figure, output, "optimizer-diagnostics.png"))
    plt.close(figure)
    return tuple(paths)


def generate_trajectory_plots(trajectories_path, output_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _read(trajectories_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for track_text, role in payload["track_roles"].items():
        track = int(track_text)
        rows = [
            item for item in payload["trajectories"] if item["evaluation_seed"] == track
        ]
        figure, axes = plt.subplots(6, 1, figsize=(11, 15), sharex=True)
        for trajectory in rows:
            samples = trajectory["samples"]
            times = [item["time_s"] for item in samples]
            label = f"seed {trajectory['training_seed']} / {trajectory['failure_mode']}"
            line = axes[0].plot(
                times, [item["position_m"] for item in samples], label=label
            )[0]
            color = line.get_color()
            axes[1].plot(
                times,
                [item["velocity_m_s"] for item in samples],
                color=color,
            )
            axes[1].plot(
                times,
                [item["speed_limit_m_s"] for item in samples],
                color=color,
                linestyle="--",
                alpha=0.4,
            )
            for axis, field in zip(
                axes[2:],
                (
                    "action",
                    "acceleration_m_s2",
                    "jerk_m_s3",
                    "cumulative_energy_kwh",
                ),
            ):
                axis.plot(times, [item[field] for item in samples], color=color)
        for axis, label in zip(
            axes,
            (
                "position [m]",
                "velocity / limit [m/s]",
                "action",
                "acceleration [m/s²]",
                "jerk [m/s³]",
                "cumulative energy [kWh]",
            ),
        ):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize="small")
        axes[-1].set_xlabel("time [s]")
        figure.suptitle(f"Binary SAC final policies / track {track}: {role}")
        paths.append(
            _save(figure, output, f"representative-trajectories-track-{track}.png")
        )
        plt.close(figure)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = generate_result_plots(args.analysis, args.results_dir, args.output_dir)
    paths += generate_trajectory_plots(args.trajectories, args.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
