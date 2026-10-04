"""Record and plot deterministic representative SAC trajectories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.checkpoint import load_checkpoint
from training.cli import _build_agent

from .analysis import discover_results
from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .experiment import (
    _agent_arguments,
    _base_environment,
    _policy_adapter,
    _training_environment,
)

TRAJECTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RepresentativeSelection:
    configuration_id: str
    training_seed: int
    evaluation_seed: int
    label: str


def select_representatives(results_directory: str | Path):
    """Select successful regimes on a common track plus one stationary failure.

    For every reward configuration with at least one feasible episode, the seed
    with the highest RSR is selected (lowest seed breaks ties). The evaluation
    track feasible for the largest number of those policies is used (lowest
    track seed breaks ties). One high-energy, zero-RSR policy is added as a
    failure control, choosing the least-progress seed and then the lowest seed.
    """

    runs = discover_results(results_directory)
    by_configuration: dict[str, list[Any]] = {}
    for run in runs:
        by_configuration.setdefault(
            run.reward_parameters.configuration_id, []
        ).append(run)

    successful = []
    for _configuration_id, candidates in sorted(by_configuration.items()):
        if not any(
            item.summary.requirement_satisfaction_rate > 0 for item in candidates
        ):
            continue
        best = min(
            candidates,
            key=lambda item: (
                -item.summary.requirement_satisfaction_rate,
                item.training_seed,
            ),
        )
        successful.append(best)
    if not successful:
        raise ValueError("No policy has a feasible episode")

    evaluation_seeds = sorted(
        {episode.evaluation_seed for run in successful for episode in run.episodes}
    )
    common_seed = min(
        evaluation_seeds,
        key=lambda seed: (
            -sum(
                next(
                    episode.feasible
                    for episode in run.episodes
                    if episode.evaluation_seed == seed
                )
                for run in successful
            ),
            seed,
        ),
    )
    selections = [
        RepresentativeSelection(
            configuration_id=run.reward_parameters.configuration_id,
            training_seed=run.training_seed,
            evaluation_seed=common_seed,
            label=(
                f"{run.reward_parameters.configuration_id}, "
                f"seed {run.training_seed}"
            ),
        )
        for run in successful
    ]

    maximum_energy_weight = max(
        run.reward_parameters.energy_weight for run in runs
    )
    failed = [
        run
        for run in runs
        if run.reward_parameters.energy_weight == maximum_energy_weight
        and run.summary.requirement_satisfaction_rate == 0
    ]
    if failed:
        control = min(
            failed,
            key=lambda run: (
                sum(item.final_position_m for item in run.episodes)
                / len(run.episodes),
                run.reward_parameters.configuration_id,
                run.training_seed,
            ),
        )
        selections.append(
            RepresentativeSelection(
                configuration_id=control.reward_parameters.configuration_id,
                training_seed=control.training_seed,
                evaluation_seed=common_seed,
                label=(
                    f"{control.reward_parameters.configuration_id}, "
                    f"seed {control.training_seed} (failure control)"
                ),
            )
        )
    return tuple(selections)


def _record_one(results_directory, configuration, selection):
    run_directory = (
        Path(results_directory)
        / selection.configuration_id
        / f"training-seed-{selection.training_seed}"
    )
    checkpoint_path = run_directory / "checkpoint.tar"
    training_environment = _training_environment(
        configuration,
        next(
            item
            for item in configuration.reward_grid
            if item.configuration_id == selection.configuration_id
        ),
    )
    evaluation_environment = _base_environment(configuration)
    try:
        arguments = _agent_arguments(
            configuration, training_seed=selection.training_seed, device="cpu"
        )
        agent = _build_agent(arguments, training_environment, evaluation_environment)
        load_checkpoint(
            checkpoint_path,
            agent,
            map_location="cpu",
            load_optimizers=False,
        )
        policy = _policy_adapter(agent)
        observation, _ = evaluation_environment.reset(
            seed=selection.evaluation_seed
        )
        samples = []
        while True:
            action = policy(np.asarray(observation, dtype=np.float32))
            observation, _, terminated, truncated, info = evaluation_environment.step(
                action
            )
            samples.append(
                {
                    "time_s": float(info["elapsed_time_s"]),
                    "position_m": float(info["position_m"]),
                    "velocity_m_s": float(info["velocity_m_s"]),
                    "speed_limit_m_s": float(info["speed_limit_m_s"]),
                    "net_energy_kwh": float(info["total_energy_kwh"]),
                    "action": float(np.asarray(action)[0]),
                    "acceleration_m_s2": float(info["acceleration_m_s2"]),
                }
            )
            if terminated or truncated:
                episode = next(
                    item
                    for item in discover_results(run_directory)[0].episodes
                    if item.evaluation_seed == selection.evaluation_seed
                )
                observed = info["episode_metrics"]
                if (
                    bool(observed["completed"]) != episode.completed
                    or not np.isclose(observed["energy_kwh"], episode.energy_kwh)
                    or not np.isclose(
                        observed["travel_time_s"], episode.travel_time_s
                    )
                ):
                    raise RuntimeError(
                        "Replayed trajectory differs from stored evaluation result"
                    )
                return {
                    "selection": asdict(selection),
                    "feasible": episode.feasible,
                    "completed": episode.completed,
                    "travel_time_s": episode.travel_time_s,
                    "energy_kwh": episode.energy_kwh,
                    "max_speed_violation_m_s": (
                        episode.max_speed_violation_m_s
                    ),
                    "samples": samples,
                }
    finally:
        training_environment.close()
        evaluation_environment.close()


def record_trajectories(
    results_directory: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    selections: tuple[RepresentativeSelection, ...] | None = None,
) -> Path:
    configuration = load_configuration(config_path)
    chosen = selections or select_representatives(results_directory)
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "benchmark_name": configuration.name,
        "configuration_sha256": configuration_sha256(configuration),
        "selection_rule": (
            "Highest-RSR seed per nonzero-RSR reward configuration (lowest seed "
            "breaks ties), on the evaluation track feasible for most selected "
            "policies (lowest track seed breaks ties), plus the least-progress "
            "high-energy zero-RSR policy as a failure control."
        ),
        "trajectories": [
            _record_one(results_directory, configuration, selection)
            for selection in chosen
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def plot_trajectories(trajectory_path: str | Path, output_path: str | Path) -> Path:
    """Create a headless plot using only a persisted trajectory JSON file."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported trajectory schema version")
    figure, axes = plt.subplots(5, 1, figsize=(10, 13), sharex=True)
    for trajectory in payload["trajectories"]:
        samples = trajectory["samples"]
        times = [sample["time_s"] for sample in samples]
        label = trajectory["selection"]["label"]
        line = axes[0].plot(
            times, [sample["position_m"] for sample in samples], label=label
        )[0]
        color = line.get_color()
        axes[1].plot(
            times,
            [sample["velocity_m_s"] for sample in samples],
            color=color,
            label=f"{label}: velocity",
        )
        axes[1].plot(
            times,
            [sample["speed_limit_m_s"] for sample in samples],
            color=color,
            linestyle="--",
            alpha=0.55,
            label=f"{label}: limit",
        )
        axes[2].plot(
            times,
            [sample["net_energy_kwh"] for sample in samples],
            color=color,
            label=label,
        )
        axes[3].plot(
            times,
            [sample["action"] for sample in samples],
            color=color,
            label=label,
        )
        axes[4].plot(
            times,
            [sample["acceleration_m_s2"] for sample in samples],
            color=color,
            label=label,
        )
    labels = (
        "position [m]",
        "velocity / limit [m/s]",
        "net energy [kWh]",
        "action",
        "acceleration [m/s²]",
    )
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize="small", ncols=2)
    axes[1].legend(fontsize="x-small", ncols=2)
    axes[-1].set_xlabel("time [s]")
    figure.suptitle("Representative deterministic trajectories")
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    args = parser.parse_args()
    trajectory_path = record_trajectories(
        args.results, args.trajectory_json, config_path=args.config
    )
    print(trajectory_path)
    print(plot_trajectories(trajectory_path, args.plot))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
