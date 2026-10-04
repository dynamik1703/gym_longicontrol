"""Generate deterministic, headless diagnostic plots from result JSON files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import numpy as np

from .analysis import failure_mode
from .evaluation import BenchmarkRunResult, load_run_result


def _discover(path: Path) -> tuple[BenchmarkRunResult, ...]:
    paths = sorted(path.rglob("result.json")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No result.json files found under {path}")
    return tuple(load_run_result(item) for item in paths)


def generate_plots(result_path: str | Path, output_directory: str | Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _discover(Path(result_path))
    grouped = defaultdict(list)
    for run in runs:
        grouped[run.reward_parameters.configuration_id].append(run)
    labels = sorted(grouped)
    satisfaction = [
        fmean(run.summary.requirement_satisfaction_rate for run in grouped[label])
        for label in labels
    ]
    satisfaction_min = [
        min(run.summary.requirement_satisfaction_rate for run in grouped[label])
        for label in labels
    ]
    satisfaction_max = [
        max(run.summary.requirement_satisfaction_rate for run in grouped[label])
        for label in labels
    ]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(max(7, len(labels)), 4))
    positions = range(len(labels))
    axis.bar(positions, satisfaction, label="mean RSR", color="#4c78a8")
    axis.errorbar(
        list(positions),
        satisfaction,
        yerr=[
            np.subtract(satisfaction, satisfaction_min),
            np.subtract(satisfaction_max, satisfaction),
        ],
        fmt="none",
        color="black",
        capsize=4,
        label="training-seed range",
    )
    for index, label in enumerate(labels):
        axis.scatter(
            [index] * len(grouped[label]),
            [run.summary.requirement_satisfaction_rate for run in grouped[label]],
            color="white",
            edgecolor="black",
            zorder=3,
        )
    axis.set_xticks(list(positions), labels, rotation=35, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("rate")
    axis.set_title("Scalar reward sensitivity")
    axis.legend()
    figure.tight_layout()
    sensitivity_path = output / "reward-sensitivity.png"
    figure.savefig(sensitivity_path, dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    for label in labels:
        for run in grouped[label]:
            energy = run.summary.mean_feasible_energy_kwh
            if energy is not None:
                axis.scatter(
                    run.summary.requirement_satisfaction_rate,
                    energy,
                    label=label,
                )
    axis.set_xlabel("requirement satisfaction rate")
    axis.set_ylabel("mean energy of feasible episodes [kWh]")
    axis.set_title("Energy versus requirement satisfaction")
    handles, legend_labels = axis.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    if unique:
        axis.legend(unique.values(), unique.keys(), fontsize="small")
    figure.tight_layout()
    energy_path = output / "energy-vs-satisfaction.png"
    figure.savefig(energy_path, dpi=120)
    plt.close(figure)

    failure_counts = {
        label: {
            mode: sum(
                failure_mode(episode, run.task) == mode
                for run in grouped[label]
                for episode in run.episodes
            )
            for mode in {
                failure_mode(episode, run.task)
                for run in grouped[label]
                for episode in run.episodes
            }
        }
        for label in labels
    }
    modes = sorted(
        {mode for values in failure_counts.values() for mode in values},
        key=lambda item: (item != "feasible", item),
    )
    figure, axis = plt.subplots(figsize=(max(7, len(labels)), 4))
    x_values = list(range(len(labels)))
    bottom = np.zeros(len(labels))
    for mode in modes:
        rates = np.array(
            [
                failure_counts[label].get(mode, 0)
                / sum(failure_counts[label].values())
                for label in labels
            ]
        )
        axis.bar(x_values, rates, bottom=bottom, label=mode)
        bottom += rates
    axis.set_xticks(x_values, labels, rotation=35, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("episode rate")
    axis.set_title("Exclusive feasibility and failure modes")
    axis.legend()
    figure.tight_layout()
    diagnostics_path = output / "failure-diagnostics.png"
    figure.savefig(diagnostics_path, dpi=120)
    plt.close(figure)
    return sensitivity_path, energy_path, diagnostics_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in generate_plots(args.results, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
