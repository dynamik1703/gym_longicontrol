"""Create the frozen constrained-V2 speed-failure diagnosis figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _failed_trajectory(source, training_seed, track_seed):
    return next(
        item
        for item in source["trajectories"]
        if item["training_seed"] == training_seed and item["track_seed"] == track_seed
    )


def plot_failure_severity(diagnosis, output_directory: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = {
        "small_constant_section_boundary_overshoot": "#4477aa",
        "late_braking_after_speed_limit_reduction": "#cc3311",
    }
    labels_seen = set()
    annotation_counts = {key: 0 for key in colors}
    for episode in diagnosis["failed_episodes"]:
        mechanisms = {item["mechanism"] for item in episode["violation_events"]}
        mechanism = next(iter(mechanisms))
        label = mechanism.replace("_", " ") if mechanism not in labels_seen else None
        labels_seen.add(mechanism)
        axis.scatter(
            episode["max_speed_violation_m_s"],
            episode["integrated_speed_violation_m"],
            color=colors[mechanism],
            s=65,
            label=label,
        )
        axis.annotate(
            f"{episode['training_seed']}/{episode['track_seed']}",
            (
                episode["max_speed_violation_m_s"],
                episode["integrated_speed_violation_m"],
            ),
            xytext=(5, 4 + 9 * annotation_counts[mechanism]),
            textcoords="offset points",
            fontsize=8,
        )
        annotation_counts[mechanism] += 1
    axis.set_xlabel("Maximum speed excess [m/s]")
    axis.set_ylabel("Integrated speed excess [m]")
    axis.set_title("Six frozen V2 failures: severity and mechanism")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    path = output_directory / "failure-severity.png"
    _save(figure, path)
    return path


def plot_transition_alignment(diagnosis, output_directory: Path) -> Path:
    figure, (distance_axis, time_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    annotation_counts = {"reduction": 0, "increase": 0}
    for episode in diagnosis["failed_episodes"]:
        for event in episode["violation_events"]:
            transition = event["nearest_speed_limit_transition"]
            color = "#cc3311" if transition["kind"] == "reduction" else "#4477aa"
            marker = "o" if transition["kind"] == "reduction" else "s"
            label = f"{episode['training_seed']}/{episode['track_seed']}"
            distance_axis.scatter(
                transition["signed_distance_m"],
                event["peak_overspeed_m_s"],
                color=color,
                marker=marker,
            )
            time_axis.scatter(
                transition["signed_time_s"],
                event["peak_overspeed_m_s"],
                color=color,
                marker=marker,
            )
            distance_axis.annotate(
                label,
                (transition["signed_distance_m"], event["peak_overspeed_m_s"]),
                xytext=(3, 4 + 7 * annotation_counts[transition["kind"]]),
                textcoords="offset points",
                fontsize=7,
            )
            time_axis.annotate(
                label,
                (transition["signed_time_s"], event["peak_overspeed_m_s"]),
                xytext=(3, 4 + 7 * annotation_counts[transition["kind"]]),
                textcoords="offset points",
                fontsize=7,
            )
            annotation_counts[transition["kind"]] += 1
    for axis, label in (
        (distance_axis, "Signed distance from nearest limit transition [m]"),
        (time_axis, "Signed time from nearest limit transition [s]"),
    ):
        axis.axvline(0, color="black", linewidth=1, alpha=0.5)
        axis.set_xlabel(label)
        axis.set_ylabel("Peak speed excess [m/s]")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Violation alignment: reductions (red circles), increases (blue squares)"
    )
    path = output_directory / "speed-limit-transition-alignment.png"
    _save(figure, path)
    return path


def plot_training_dynamics(
    diagnosis, raw_results: Path, output_directory: Path
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for seed, summary in diagnosis["training_diagnostics"].items():
        rows = json.loads(
            (
                raw_results / f"training-seed-{seed}" / "training-diagnostics.json"
            ).read_text(encoding="utf-8")
        )
        steps = [row["simulator_steps"] for row in rows]
        axes[0, 0].plot(
            steps, [row["lagrange_speed"] for row in rows], label=f"seed {seed}"
        )
        axes[1, 0].plot(
            steps, [row["lagrange_deadline"] for row in rows], label=f"seed {seed}"
        )
        curve = summary["checkpoint_validation"]
        checkpoint_steps = [row["simulator_step_target"] for row in curve]
        axes[0, 1].plot(
            checkpoint_steps,
            [row["rsr"] for row in curve],
            marker="o",
            label=f"seed {seed}",
        )
        axes[1, 1].plot(
            checkpoint_steps,
            [row["speed_compliance_rate"] for row in curve],
            marker="o",
            label=f"seed {seed}",
        )
    axes[0, 0].set_ylabel("lambda_speed")
    axes[1, 0].set_ylabel("lambda_deadline")
    axes[0, 1].set_ylabel("Validation RSR")
    axes[1, 1].set_ylabel("Speed compliance")
    axes[1, 0].set_xlabel("Simulator transitions")
    axes[1, 1].set_xlabel("Simulator transitions")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[1, 1].set_ylim(-0.03, 1.03)
    figure.suptitle("Frozen V2 multipliers and Validation behavior")
    path = output_directory / "multiplier-and-validation-dynamics.png"
    _save(figure, path)
    return path


def plot_event_windows(source, diagnosis, output_directory: Path) -> list[Path]:
    paths = []
    for episode in diagnosis["failed_episodes"]:
        trajectory = _failed_trajectory(
            source, episode["training_seed"], episode["track_seed"]
        )
        samples = trajectory["samples"]
        event_count = len(episode["violation_events"])
        figure, axes = plt.subplots(
            event_count, 3, figsize=(13, 3.5 * event_count), squeeze=False
        )
        for row, event in enumerate(episode["violation_events"]):
            center = event["first_violation_time_s"]
            window = [item for item in samples if abs(item["time_s"] - center) <= 15.0]
            time = np.asarray([item["time_s"] - center for item in window])
            velocity_axis, pressure_axis, control_axis = axes[row]
            velocity_axis.plot(
                time, [item["velocity_m_s"] for item in window], label="vehicle"
            )
            velocity_axis.step(
                time,
                [item["speed_limit_m_s"] for item in window],
                where="post",
                label="limit",
            )
            pressure_axis.plot(
                time, [item["deadline_slack_s"] for item in window], label="slack"
            )
            pressure_axis.plot(
                time, [item["deadline_deficit_s"] for item in window], label="deficit"
            )
            control_axis.plot(time, [item["action"] for item in window], label="action")
            control_axis.plot(
                time,
                [item["acceleration_m_s2"] for item in window],
                label="acceleration",
            )
            transition = event["nearest_speed_limit_transition"]
            transition_relative = transition["crossing_time_s"] - center
            for axis in axes[row]:
                axis.axvspan(0, event["duration_s"], color="#cc3311", alpha=0.12)
                if abs(transition_relative) <= 15:
                    axis.axvline(
                        transition_relative, color="black", linestyle="--", linewidth=1
                    )
                axis.grid(alpha=0.25)
                axis.legend(fontsize=8)
                axis.set_xlabel("Time from first violating sample [s]")
            velocity_axis.set_ylabel("Speed [m/s]")
            pressure_axis.set_ylabel("Deadline state [s]")
            control_axis.set_ylabel("Action / acceleration [m/s²]")
        title = (
            f"V2 seed {episode['training_seed']}, track "
            f"{episode['track_seed']}: ±15 s event windows"
        )
        figure.suptitle(title)
        path = output_directory / (
            f"event-windows-seed-{episode['training_seed']}-track-{episode['track_seed']}.png"
        )
        _save(figure, path)
        paths.append(path)
    return paths


def create_plots(
    trajectory_path: str | Path,
    diagnosis_path: str | Path,
    raw_results: str | Path,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    source = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    diagnosis = json.loads(Path(diagnosis_path).read_text(encoding="utf-8"))
    output = Path(output_directory)
    paths = [
        plot_failure_severity(diagnosis, output),
        plot_transition_alignment(diagnosis, output),
        plot_training_dynamics(diagnosis, Path(raw_results), output),
    ]
    paths.extend(plot_event_windows(source, diagnosis, output))
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("diagnosis", type=Path)
    parser.add_argument("--v2-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in create_plots(
        args.trajectories, args.diagnosis, args.v2_results, args.output_dir
    ):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
