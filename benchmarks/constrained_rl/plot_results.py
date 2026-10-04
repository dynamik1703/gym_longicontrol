"""Create reproducible plots for the constrained-RL study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rolling(values, width=10):
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    for index in range(len(array)):
        result[index] = np.mean(array[max(0, index - width + 1) : index + 1])
    return result


def _save(figure, output: Path, filename: str):
    path = output / filename
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    return path


def generate_result_plots(
    analysis_path: str | Path,
    diagnostics_directory: str | Path,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = _read(analysis_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []

    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, sharey=True)
    comparisons = (
        (
            analysis["learning_curves"],
            "Explicit constraints: FSRL SAC-Lagrangian",
            "simulator_step_target",
        ),
        (
            analysis["frozen_comparisons"]["sb3_sac_v2b_learning_curves"],
            "Frozen scalar reward: SB3 SAC V2-B",
            "training_steps",
        ),
    )
    for axis, (rows, title, step_key) in zip(axes, comparisons):
        for seed in sorted({row["training_seed"] for row in rows}):
            seed_rows = sorted(
                (row for row in rows if row["training_seed"] == seed),
                key=lambda row: row[step_key],
            )
            axis.plot(
                [row[step_key] for row in seed_rows],
                [row["rsr"] for row in seed_rows],
                marker="o",
                label=f"seed {seed}",
            )
        axis.set_title(title)
        axis.set_ylabel("validation RSR")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(ncols=3, fontsize="small")
    axes[-1].set_xlabel("underlying simulator transitions")
    paths.append(
        _save(figure, output, "validation-rsr-constrained-vs-scalar.png")
    )
    plt.close(figure)

    final_key = "validation-3000-3008-v1/simulator-target-300000"
    summaries = (
        analysis["summaries"][final_key],
        analysis["frozen_comparisons"]["sb3_sac_v2b_300k"],
        analysis["frozen_comparisons"]["credit_condition_c_300k"],
    )
    labels = ("Constrained", "SB3 SAC", "Credit C")
    modes = sorted(
        {
            mode
            for summary in summaries
            for mode in summary["failure_mode_counts"]
        },
        key=lambda item: (item != "feasible", item),
    )
    figure, axis = plt.subplots(figsize=(8, 4.8))
    x_values = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    for mode in modes:
        values = np.asarray(
            [
                summary["failure_mode_counts"].get(mode, 0)
                / summary["episode_count"]
                for summary in summaries
            ]
        )
        axis.bar(x_values, values, bottom=bottoms, label=mode)
        bottoms += values
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("validation episode proportion")
    axis.set_title("Final exclusive failure modes")
    axis.legend(fontsize="small", ncols=2)
    paths.append(_save(figure, output, "failure-modes-300k.png"))
    plt.close(figure)

    material = analysis["material_improvement"]
    seeds = sorted(int(seed) for seed in material["constrained_rsr_by_seed"])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(seeds))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [material["constrained_rsr_by_seed"][str(seed)] for seed in seeds],
        width,
        label="Constrained",
    )
    axis.bar(
        positions + width / 2,
        [material["frozen_scalar_sac_rsr_by_seed"][str(seed)] for seed in seeds],
        width,
        label="Frozen SB3 SAC",
    )
    axis.set_xticks(positions, [f"seed {seed}" for seed in seeds])
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("validation RSR")
    axis.set_title("Final training-seed reliability")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    paths.append(_save(figure, output, "seed-reliability-300k.png"))
    plt.close(figure)

    diagnostics_root = Path(diagnostics_directory)
    histories = {
        seed: _read(
            diagnostics_root
            / f"training-seed-{seed}"
            / "training-diagnostics.json"
        )
        for seed in seeds
    }
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for seed, rows in histories.items():
        steps = [row["simulator_steps"] for row in rows]
        axes[0].plot(
            steps,
            _rolling([row["objective_return"] for row in rows]),
            label=f"seed {seed}",
        )
        for key, style, name in (
            ("speed_integral_m", "-", "speed"),
            ("task_failure", "--", "task"),
        ):
            axes[1].plot(
                steps,
                _rolling([row[key] for row in rows]),
                linestyle=style,
                label=f"{name} / {seed}",
            )
        for key, style, name in (
            ("lagrange_speed", "-", "speed"),
            ("lagrange_task", "--", "task"),
        ):
            axes[2].plot(
                steps,
                [row[key] for row in rows],
                linestyle=style,
                label=f"{name} / {seed}",
            )
    axes[0].set_ylabel("objective return\n(10-episode mean)")
    axes[1].set_ylabel("episode cost\n(10-episode mean)")
    axes[2].set_ylabel("PID multiplier")
    axes[2].set_xlabel("underlying simulator transitions")
    axes[0].set_title("Objective, costs, and separate Lagrange multipliers")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize="x-small", ncols=3)
    paths.append(_save(figure, output, "objective-cost-multiplier-dynamics.png"))
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    for seed, rows in histories.items():
        steps = [row["simulator_steps"] for row in rows]
        for key, style, name in (
            ("loss/q0", "-", "objective"),
            ("loss/q1", "--", "speed"),
            ("loss/q2", ":", "task"),
        ):
            axes[0].plot(
                steps,
                _rolling([row[key] for row in rows]),
                linestyle=style,
                label=f"{name} / {seed}",
            )
        axes[1].plot(
            steps,
            _rolling([row["loss/actor_total"] for row in rows]),
            label=f"seed {seed}",
        )
        axes[2].plot(
            steps,
            [row["alpha"] for row in rows],
            label=f"seed {seed}",
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("critic loss\n(10-episode mean)")
    axes[1].set_ylabel("actor loss\n(10-episode mean)")
    axes[2].set_ylabel("entropy alpha")
    axes[2].set_xlabel("underlying simulator transitions")
    axes[0].set_title("SAC-Lagrangian optimization diagnostics")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize="x-small", ncols=3)
    paths.append(_save(figure, output, "optimizer-diagnostics.png"))
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
    for trajectory in payload["trajectories"]:
        samples = trajectory["samples"]
        times = [sample["time_s"] for sample in samples]
        figure, axes = plt.subplots(6, 1, figsize=(10, 13), sharex=True)
        axes[0].plot(times, [sample["position_m"] for sample in samples])
        axes[1].plot(
            times,
            [sample["velocity_m_s"] for sample in samples],
            label="velocity",
        )
        axes[1].plot(
            times,
            [sample["speed_limit_m_s"] for sample in samples],
            linestyle="--",
            label="speed limit",
        )
        axes[2].plot(times, [sample["acceleration_m_s2"] for sample in samples])
        axes[3].plot(times, [sample["jerk_m_s3"] for sample in samples])
        axes[4].plot(times, [sample["action"] for sample in samples])
        axes[5].plot(
            times, [sample["cumulative_energy_kwh"] for sample in samples]
        )
        for axis, label in zip(
            axes,
            (
                "position [m]",
                "speed [m/s]",
                "acceleration [m/s²]",
                "jerk [m/s³]",
                "action",
                "net energy [kWh]",
            ),
        ):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[1].legend(fontsize="small")
        axes[-1].set_xlabel("time [s]")
        figure.suptitle(
            f"FSRL SACLag seed {payload['training_seed']} / "
            f"track {trajectory['evaluation_seed']} / "
            f"{trajectory['failure_mode']}"
        )
        filename = (
            "representative-trajectory-track-"
            f"{trajectory['evaluation_seed']}.png"
        )
        paths.append(_save(figure, output, filename))
        plt.close(figure)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = generate_result_plots(
        args.analysis, args.diagnostics_dir, args.output_dir
    )
    paths += generate_trajectory_plots(args.trajectories, args.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
