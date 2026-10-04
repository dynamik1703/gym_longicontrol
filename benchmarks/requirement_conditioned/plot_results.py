"""Create reproducible plots for the requirement-conditioned study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(figure, output: Path, filename: str):
    path = output / filename
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return path


def _final_episodes(results_directory: Path, analysis):
    target = max(row["simulator_step_target"] for row in analysis["learning_curves"])
    episodes = []
    for seed in sorted(
        int(value) for value in analysis["final_validation"]["by_training_seed"]
    ):
        payload = _read(
            results_directory
            / f"training-seed-{seed}"
            / f"target-{target:06d}"
            / "validation-result.json"
        )
        for item in payload["episodes"]:
            episodes.append({**item, "training_seed": seed})
    return episodes


def generate_result_plots(
    analysis_path: str | Path,
    results_directory: str | Path,
    output_directory: str | Path,
) -> tuple[Path, ...]:
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
    margins = sorted(int(value) for value in analysis["final_validation"]["by_margin"])

    figure, axis = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(margins))
    width = 0.24
    for index, seed in enumerate(seeds):
        values = analysis["final_validation"]["by_training_seed"][str(seed)][
            "by_margin"
        ]
        axis.bar(
            positions + (index - 1) * width,
            [
                values[str(margin)]["requirement_satisfaction_rate"]
                for margin in margins
            ],
            width,
            label=f"training seed {seed}",
        )
    axis.axhline(0.5, color="black", linestyle="--", label="preregistered minimum")
    axis.set_xticks(positions, [f"{value} s" for value in margins])
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("requirement margin beyond optimistic minimum")
    axis.set_ylabel("validation RSR")
    axis.set_title("Requirement satisfaction by policy seed and requirement")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize="small", ncols=2)
    paths.append(_save(figure, output, "validation-rsr-by-requirement.png"))
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for seed in seeds:
        rows = sorted(
            (
                row
                for row in analysis["learning_curves"]
                if row["training_seed"] == seed
            ),
            key=lambda row: row["simulator_step_target"],
        )
        axes[0].plot(
            [row["simulator_step_target"] for row in rows],
            [row["overall_rsr"] for row in rows],
            marker="o",
            label=f"seed {seed}",
        )
        for margin, style in zip((20, 40, 60), ("-", "--", ":")):
            axes[1].plot(
                [row["simulator_step_target"] for row in rows],
                [row["rsr_by_margin"][str(margin)] for row in rows],
                linestyle=style,
                marker=".",
                label=f"seed {seed} / {margin} s",
            )
    axes[0].set_title("Validation learning curves: no post-hoc budget extension")
    axes[0].set_ylabel("overall RSR")
    axes[1].set_ylabel("seen-requirement RSR")
    axes[1].set_xlabel("underlying simulator transitions")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize="x-small", ncols=3)
    paths.append(_save(figure, output, "validation-learning-curves.png"))
    plt.close(figure)

    episodes = _final_episodes(root, analysis)
    grouped = {}
    for item in episodes:
        grouped.setdefault((item["training_seed"], item["track_seed"]), []).append(item)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for values in grouped.values():
        rows = sorted(values, key=lambda item: item["requirement_margin_s"])
        axes[0].plot(
            [row["requirement_margin_s"] for row in rows],
            [row["travel_time_s"] for row in rows],
            color="0.72",
            alpha=0.7,
        )
        feasible = [row for row in rows if row["feasible"]]
        if feasible:
            axes[1].plot(
                [row["requirement_margin_s"] for row in feasible],
                [row["energy_kwh"] for row in feasible],
                marker="o",
                color="0.45",
                alpha=0.7,
            )
    mean_times = [
        analysis["final_validation"]["by_margin"][str(value)]["mean_travel_time_s"]
        for value in margins
    ]
    axes[0].plot(margins, mean_times, color="tab:blue", marker="o", linewidth=3)
    axes[0].set_title("Within-policy deadline response (27 paired groups)")
    axes[0].set_xlabel("requirement margin [s]")
    axes[0].set_ylabel("achieved travel time [s]")
    axes[1].set_title("Energy only for requirement-feasible episodes")
    axes[1].set_xlabel("requirement margin [s]")
    axes[1].set_ylabel("net energy [kWh]")
    for axis in axes:
        axis.grid(alpha=0.2)
    paths.append(_save(figure, output, "requirement-response.png"))
    plt.close(figure)

    groups = analysis["controllability"]["groups"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].hist(
        [row["travel_time_range_s"] for row in groups], bins=10, color="tab:blue"
    )
    axes[0].axvline(1.0, color="black", linestyle="--", label="sensitivity threshold")
    axes[0].set_xlabel("five-requirement travel-time range [s]")
    axes[0].set_ylabel("policy/track groups")
    axes[0].legend(fontsize="small")
    axes[1].scatter(
        [row["travel_time_range_s"] for row in groups],
        [
            row["tight_to_loose_action_profile_mean_absolute_difference"]
            for row in groups
        ],
        c=[row["training_seed"] for row in groups],
        cmap="viridis",
    )
    axes[1].axhline(0.05, color="black", linestyle="--")
    axes[1].set_xlabel("travel-time range [s]")
    axes[1].set_ylabel("tight/loose mean absolute action difference")
    figure.suptitle("Requirement sensitivity is measurable but not sufficient for RSR")
    for axis in axes:
        axis.grid(alpha=0.2)
    paths.append(_save(figure, output, "requirement-sensitivity.png"))
    plt.close(figure)

    diagnostics = analysis["training_diagnostics"]
    pooled = diagnostics["pooled_by_training_requirement_margin"]
    figure, axes = plt.subplots(2, 1, figsize=(9, 8))
    x_values = np.arange(3)
    width = 0.34
    training_margins = (20, 40, 60)
    axes[0].bar(
        x_values - width / 2,
        [pooled[str(value)]["mean_speed_cost_return_m"] for value in training_margins],
        width,
        label="speed cost",
    )
    axes[0].bar(
        x_values + width / 2,
        [
            pooled[str(value)]["mean_deadline_cost_return_s"]
            for value in training_margins
        ],
        width,
        label="deadline cost",
    )
    axes[0].set_xticks(x_values, [f"{value} s" for value in training_margins])
    axes[0].set_ylabel("mean episode cost return")
    axes[0].set_title("Tight requirements create the largest constraint returns")
    axes[0].legend()
    for seed in seeds:
        rows = _read(root / f"training-seed-{seed}" / "training-diagnostics.json")
        axes[1].plot(
            [row["simulator_steps"] for row in rows],
            [row["lagrange_speed"] for row in rows],
            label=f"speed / {seed}",
        )
        axes[1].plot(
            [row["simulator_steps"] for row in rows],
            [row["lagrange_deadline"] for row in rows],
            linestyle="--",
            label=f"deadline / {seed}",
        )
    axes[1].set_xlabel("underlying simulator transitions")
    axes[1].set_ylabel("PID multiplier")
    axes[1].legend(fontsize="x-small", ncols=3)
    for axis in axes:
        axis.grid(alpha=0.2)
    paths.append(_save(figure, output, "training-costs-and-multipliers.png"))
    plt.close(figure)

    canonical = analysis["canonical_140"]
    figure, axis = plt.subplots(figsize=(7, 4.8))
    values = (
        canonical["requirement_conditioned"]["requirement_satisfaction_rate"],
        canonical["frozen_constrained_v2"]["requirement_satisfaction_rate"],
    )
    bars = axis.bar(("Conditioned V1", "Frozen Constrained V2"), values)
    axis.bar_label(
        bars,
        labels=(
            f"{canonical['requirement_conditioned']['feasible_count']}/27",
            f"{canonical['frozen_constrained_v2']['feasible_count']}/27",
        ),
    )
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("canonical 140-s RSR")
    axis.set_title("Cost of one policy serving multiple deadlines")
    axis.grid(axis="y", alpha=0.2)
    paths.append(_save(figure, output, "canonical-140-comparison.png"))
    plt.close(figure)
    return tuple(paths)


def generate_trajectory_plots(
    trajectories_path: str | Path, output_directory: str | Path
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = _read(trajectories_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for track in payload["selected_tracks"]:
        trajectories = [
            item for item in payload["trajectories"] if item["track_seed"] == track
        ]
        figure, axes_grid = plt.subplots(4, 2, figsize=(13, 15))
        axes = axes_grid.ravel()
        colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(trajectories)))
        for trajectory, color in zip(trajectories, colors):
            samples = trajectory["samples"]
            times = [item["time_s"] for item in samples]
            positions = [item["position_m"] for item in samples]
            label = f"margin {trajectory['requirement_margin_s']:g} s"
            axes[0].plot(times, positions, color=color, label=label)
            axes[1].plot(
                positions,
                [item["velocity_m_s"] for item in samples],
                color=color,
                label=label,
            )
            axes[2].plot(positions, [item["action"] for item in samples], color=color)
            axes[3].plot(
                positions,
                [item["acceleration_m_s2"] for item in samples],
                color=color,
            )
            axes[4].plot(
                positions, [item["jerk_m_s3"] for item in samples], color=color
            )
            axes[5].plot(
                positions,
                [item["cumulative_energy_kwh"] for item in samples],
                color=color,
            )
            axes[6].plot(
                positions,
                [item["deadline_slack_s"] for item in samples],
                color=color,
            )
        first = trajectories[0]["samples"]
        axes[1].plot(
            [item["position_m"] for item in first],
            [item["speed_limit_m_s"] for item in first],
            color="black",
            linestyle="--",
            label="speed limit",
        )
        labels = (
            ("time [s]", "position [m]"),
            ("position [m]", "velocity [m/s]"),
            ("position [m]", "action"),
            ("position [m]", "acceleration [m/s²]"),
            ("position [m]", "jerk [m/s³]"),
            ("position [m]", "cumulative net energy [kWh]"),
            ("position [m]", "deadline slack [s]"),
        )
        for axis, (x_label, y_label) in zip(axes[:7], labels):
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize="small", ncols=2)
        axes[1].legend(fontsize="small", ncols=2)
        axes[7].axis("off")
        figure.suptitle(
            f"Requirement-conditioned policy seed {payload['training_seed']} / "
            f"validation track {track}"
        )
        paths.append(
            _save(figure, output, f"representative-trajectory-track-{track}.png")
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
