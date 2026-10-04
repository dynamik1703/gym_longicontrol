"""Record representative Scalar V2 trajectories from persisted checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from training.checkpoint import load_checkpoint
from training.cli import _build_agent

from .analysis import failure_mode
from .experiment import _agent_arguments, _base_environment, _policy_adapter
from .v2_config import (
    DEFAULT_V2_CONFIG_PATH,
    load_v2_configuration,
    v2_configuration_sha256,
)
from .v2_evaluation import load_v2_result
from .v2_experiment import _training_environment

V2_TRAJECTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class V2TrajectorySelection:
    configuration_id: str
    training_seed: int
    training_steps: int
    role: str


def _exploratory_runs(results_directory: str | Path):
    paths = Path(results_directory).glob(
        "*/training-seed-*/step-*/exploratory-result.json"
    )
    return tuple(load_v2_result(path) for path in sorted(paths))


def select_v2_representatives(results_directory: str | Path):
    runs = _exploratory_runs(results_directory)

    def ranked(step):
        values = [run for run in runs if run.training_steps == step]
        return sorted(
            values,
            key=lambda run: (
                -run.summary.requirement_satisfaction_rate,
                -run.summary.completion_rate,
                run.reward_parameters.configuration_id,
                run.training_seed,
            ),
        )

    best_100 = ranked(100_000)[0]
    completion_100 = min(
        (run for run in ranked(100_000) if run is not best_100),
        key=lambda run: (
            -run.summary.completion_rate,
            run.summary.requirement_satisfaction_rate,
            run.reward_parameters.configuration_id,
            run.training_seed,
        ),
    )
    best_300 = ranked(300_000)[0]
    same_configuration = [
        run
        for run in ranked(300_000)
        if run.reward_parameters.configuration_id
        == best_300.reward_parameters.configuration_id
    ]
    failed_control = min(
        same_configuration,
        key=lambda run: (
            run.summary.requirement_satisfaction_rate,
            run.summary.completion_rate,
            run.training_seed,
        ),
    )
    compliance_candidates = [
        run
        for run in ranked(300_000)
        if "strong-compliance" in run.reward_parameters.configuration_id
    ]
    best_compliance = compliance_candidates[0]
    roles_and_runs = (
        ("best at 100k", best_100),
        ("highest-completion constraint-limited at 100k", completion_100),
        ("best at 300k", best_300),
        ("same-config failed seed at 300k", failed_control),
        ("best strong-compliance at 300k", best_compliance),
    )
    unique = []
    seen = set()
    for role, run in roles_and_runs:
        identity = (
            run.reward_parameters.configuration_id,
            run.training_seed,
            run.training_steps,
        )
        if identity not in seen:
            seen.add(identity)
            unique.append(V2TrajectorySelection(*identity, role))
    return tuple(unique)


def _track_choices(runs, selections):
    by_identity = {
        (
            run.reward_parameters.configuration_id,
            run.training_seed,
            run.training_steps,
        ): run
        for run in runs
    }
    selected_runs = [
        by_identity[
            (
                selection.configuration_id,
                selection.training_seed,
                selection.training_steps,
            )
        ]
        for selection in selections
    ]
    moving = [
        run for run in selected_runs if any(item.feasible for item in run.episodes)
    ]
    seeds = sorted(item.evaluation_seed for item in selected_runs[0].episodes)
    common_feasible = min(
        seeds,
        key=lambda seed: (
            -sum(
                next(
                    item.feasible
                    for item in run.episodes
                    if item.evaluation_seed == seed
                )
                for run in moving
            ),
            seed,
        ),
    )
    contrast = min(
        seeds,
        key=lambda seed: (
            -len(
                {
                    failure_mode(
                        next(
                            item
                            for item in run.episodes
                            if item.evaluation_seed == seed
                        ),
                        run.task,
                    )
                    for run in selected_runs
                }
            ),
            seed,
        ),
    )
    return common_feasible, contrast


def _record_one(results_directory, configuration, selection, evaluation_seed):
    step_directory = (
        Path(results_directory)
        / selection.configuration_id
        / f"training-seed-{selection.training_seed}"
        / f"step-{selection.training_steps:06d}"
    )
    result = load_v2_result(step_directory / "exploratory-result.json")
    parameters = next(
        item
        for item in configuration.reward_candidates
        if item.configuration_id == selection.configuration_id
    )
    training_environment = _training_environment(configuration, parameters)
    evaluation_environment = _base_environment(configuration)
    try:
        arguments = _agent_arguments(
            configuration, training_seed=selection.training_seed, device="cpu"
        )
        agent = _build_agent(arguments, training_environment, evaluation_environment)
        load_checkpoint(
            step_directory / "checkpoint.tar",
            agent,
            map_location="cpu",
            load_optimizers=False,
        )
        policy = _policy_adapter(agent)
        observation, _ = evaluation_environment.reset(seed=evaluation_seed)
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
                    "jerk_m_s3": float(info["jerk_m_s3"]),
                }
            )
            if terminated or truncated:
                expected = next(
                    item
                    for item in result.episodes
                    if item.evaluation_seed == evaluation_seed
                )
                if (
                    bool(info["episode_metrics"]["completed"]) != expected.completed
                    or not np.isclose(
                        info["episode_metrics"]["energy_kwh"], expected.energy_kwh
                    )
                ):
                    raise RuntimeError("V2 trajectory replay differs from result")
                return {
                    "selection": asdict(selection),
                    "evaluation_seed": evaluation_seed,
                    "feasible": expected.feasible,
                    "failure_mode": failure_mode(expected, result.task),
                    "travel_time_s": expected.travel_time_s,
                    "energy_kwh": expected.energy_kwh,
                    "max_speed_violation_m_s": expected.max_speed_violation_m_s,
                    "samples": samples,
                }
    finally:
        training_environment.close()
        evaluation_environment.close()


def record_v2_trajectories(
    results_directory: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_V2_CONFIG_PATH,
) -> Path:
    configuration = load_v2_configuration(config_path)
    runs = _exploratory_runs(results_directory)
    selections = select_v2_representatives(results_directory)
    common_track, contrast_track = _track_choices(runs, selections)
    payload = {
        "schema_version": V2_TRAJECTORY_SCHEMA_VERSION,
        "benchmark_name": configuration.name,
        "configuration_sha256": v2_configuration_sha256(configuration),
        "selection_rule": (
            "Best-RSR policy at 100k and 300k; highest-completion remaining "
            "policy at 100k; lowest-RSR seed of the best 300k configuration; "
            "and best strong-compliance policy at 300k. Tracks are the lowest "
            "seed maximizing common feasibility and the lowest seed maximizing "
            "distinct failure modes."
        ),
        "track_roles": {
            str(common_track): "maximum common feasibility",
            str(contrast_track): "maximum distinct failure modes",
        },
        "trajectories": [
            _record_one(results_directory, configuration, selection, track)
            for track in (common_track, contrast_track)
            for selection in selections
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def plot_v2_trajectories(
    trajectory_path: str | Path, output_directory: str | Path
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(Path(trajectory_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != V2_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported V2 trajectory schema version")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    track_seeds = sorted(
        {item["evaluation_seed"] for item in payload["trajectories"]}
    )
    for track_seed in track_seeds:
        figure, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=True)
        for trajectory in payload["trajectories"]:
            if trajectory["evaluation_seed"] != track_seed:
                continue
            samples = trajectory["samples"]
            times = [sample["time_s"] for sample in samples]
            selection = trajectory["selection"]
            label = (
                f"{selection['configuration_id']}/s{selection['training_seed']}"
                f"/{selection['training_steps'] // 1000}k ({selection['role']})"
            )
            line = axes[0].plot(
                times, [sample["position_m"] for sample in samples], label=label
            )[0]
            color = line.get_color()
            fields = (
                "velocity_m_s",
                "net_energy_kwh",
                "action",
                "acceleration_m_s2",
                "jerk_m_s3",
            )
            for axis, field in zip(axes[1:], fields):
                axis.plot(
                    times,
                    [sample[field] for sample in samples],
                    color=color,
                    label=label,
                )
            axes[1].plot(
                times,
                [sample["speed_limit_m_s"] for sample in samples],
                color=color,
                linestyle="--",
                alpha=0.45,
            )
        labels = (
            "position [m]",
            "velocity / limit [m/s]",
            "net energy [kWh]",
            "action",
            "acceleration [m/s²]",
            "jerk [m/s³]",
        )
        for axis, label in zip(axes, labels):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize="x-small", ncols=2)
        axes[-1].set_xlabel("time [s]")
        figure.suptitle(f"Scalar V2-B representative trajectories: track {track_seed}")
        figure.tight_layout()
        destination = output / f"representative-trajectories-track-{track_seed}.png"
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        paths.append(destination)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_V2_CONFIG_PATH)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path = record_v2_trajectories(
        args.results, args.trajectory_json, config_path=args.config
    )
    print(path)
    for plot_path in plot_v2_trajectories(path, args.output_dir):
        print(plot_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
