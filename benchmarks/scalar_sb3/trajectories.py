"""Select and replay representative final SB3 policies without cherry-picking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from benchmarks.scalar_sac.analysis import failure_mode
from benchmarks.scalar_sac.experiment import _base_environment

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import load_result

TRAJECTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Selection:
    algorithm: str
    training_seed: int
    role: str


def _validation_runs(results_directory):
    paths = Path(results_directory).glob(
        "*/training-seed-*/step-300000/validation-3000-3008-result.json"
    )
    return tuple(load_result(path) for path in sorted(paths))


def select_representatives(results_directory: str | Path):
    final_runs = _validation_runs(results_directory)
    if not final_runs:
        raise ValueError("No final SB3 validation results found")

    def best(algorithm):
        return min(
            (run for run in final_runs if run.algorithm == algorithm),
            key=lambda run: (
                -run.summary.requirement_satisfaction_rate,
                -run.summary.completion_rate,
                -run.summary.speed_compliance_rate,
                run.training_seed,
            ),
        )

    def worst(algorithm):
        return min(
            (run for run in final_runs if run.algorithm == algorithm),
            key=lambda run: (
                run.summary.requirement_satisfaction_rate,
                run.summary.completion_rate,
                run.summary.speed_compliance_rate,
                -run.training_seed,
            ),
        )

    roles = [
        ("best final SAC", best("sac")),
        ("lowest final SAC", worst("sac")),
        ("best final PPO", best("ppo")),
        ("lowest final PPO", worst("ppo")),
    ]
    all_paths = Path(results_directory).glob(
        "*/training-seed-*/step-*/validation-3000-3008-result.json"
    )
    by_policy = {}
    for path in sorted(all_paths):
        run = load_result(path)
        by_policy.setdefault((run.algorithm, run.training_seed), []).append(run)
    collapse_runs = min(
        by_policy.items(),
        key=lambda item: (
            item[1][-1].summary.requirement_satisfaction_rate
            - max(
                run.summary.requirement_satisfaction_rate for run in item[1]
            ),
            item[0],
        ),
    )[1]
    peak_rsr = max(
        run.summary.requirement_satisfaction_rate for run in collapse_runs
    )
    final_rsr = collapse_runs[-1].summary.requirement_satisfaction_rate
    if final_rsr < peak_rsr:
        roles.append(
            ("largest peak-to-final RSR decline", collapse_runs[-1])
        )
    selections = []
    seen = set()
    for role, run in roles:
        identity = (run.algorithm, run.training_seed)
        if identity not in seen:
            seen.add(identity)
            selections.append(Selection(*identity, role))
    return tuple(selections)


def _track_choices(runs, selections):
    by_identity = {(run.algorithm, run.training_seed): run for run in runs}
    selected = [
        by_identity[(item.algorithm, item.training_seed)] for item in selections
    ]
    seeds = [episode.evaluation_seed for episode in selected[0].episodes]
    common = min(
        seeds,
        key=lambda seed: (
            -sum(
                next(
                    episode.feasible
                    for episode in run.episodes
                    if episode.evaluation_seed == seed
                )
                for run in selected
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
                            episode
                            for episode in run.episodes
                            if episode.evaluation_seed == seed
                        ),
                        run.task,
                    )
                    for run in selected
                }
            ),
            seed,
        ),
    )
    stalled_progress = min(
        seeds,
        key=lambda seed: (
            -max(
                (
                    episode.final_position_m
                    for run in selected
                    for episode in run.episodes
                    if episode.evaluation_seed == seed and not episode.feasible
                ),
                default=0.0,
            ),
            seed,
        ),
    )
    choices = {}
    for seed, role in (
        (common, "maximum common feasibility"),
        (contrast, "maximum distinct failure modes"),
        (
            stalled_progress,
            "maximum progress among infeasible selected policies",
        ),
    ):
        choices.setdefault(seed, role)
    return tuple(choices.items())


def _load_model(results_directory, selection):
    from stable_baselines3 import PPO, SAC

    model_path = (
        Path(results_directory)
        / selection.algorithm
        / f"training-seed-{selection.training_seed}"
        / "step-300000"
        / "model.zip"
    )
    model_class = SAC if selection.algorithm == "sac" else PPO
    return model_class.load(model_path, device="cpu")


def _record_one(results_directory, configuration, selection, track_seed):
    result_path = (
        Path(results_directory)
        / selection.algorithm
        / f"training-seed-{selection.training_seed}"
        / "step-300000"
        / "validation-3000-3008-result.json"
    )
    result = load_result(result_path)
    expected = next(
        item for item in result.episodes if item.evaluation_seed == track_seed
    )
    model = _load_model(results_directory, selection)
    environment = _base_environment(configuration)
    try:
        observation, _ = environment.reset(seed=track_seed)
        samples = []
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            samples.append(
                {
                    "time_s": float(info["elapsed_time_s"]),
                    "position_m": float(info["position_m"]),
                    "velocity_m_s": float(info["velocity_m_s"]),
                    "speed_limit_m_s": float(info["speed_limit_m_s"]),
                    "acceleration_m_s2": float(info["acceleration_m_s2"]),
                    "action": float(np.asarray(action).reshape((1,))[0]),
                    "cumulative_energy_kwh": float(info["total_energy_kwh"]),
                }
            )
            if terminated or truncated:
                actual = info["episode_metrics"]
                if bool(actual["completed"]) != expected.completed or not np.isclose(
                    actual["energy_kwh"], expected.energy_kwh
                ):
                    raise RuntimeError("SB3 trajectory replay differs from result")
                return {
                    "selection": asdict(selection),
                    "evaluation_seed": track_seed,
                    "feasible": expected.feasible,
                    "failure_mode": failure_mode(expected, result.task),
                    "travel_time_s": expected.travel_time_s,
                    "energy_kwh": expected.energy_kwh,
                    "max_speed_violation_m_s": expected.max_speed_violation_m_s,
                    "samples": samples,
                }
    finally:
        environment.close()


def record(results_directory, output_path, *, config_path=DEFAULT_CONFIG_PATH):
    configuration = load_configuration(config_path)
    runs = _validation_runs(results_directory)
    selections = select_representatives(results_directory)
    track_choices = _track_choices(runs, selections)
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "configuration_sha256": configuration_sha256(configuration),
        "selection_rule": (
            "Best and lowest final-RSR seed per algorithm and (when present) "
            "largest peak-to-final RSR decline; ties use completion, compliance, "
            "then the lower best or higher worst seed. Tracks maximize common "
            "feasibility, distinct "
            "failure modes, and progress among infeasible selected policies, with "
            "the lowest seed breaking ties."
        ),
        "track_roles": {str(track): role for track, role in track_choices},
        "trajectories": [
            _record_one(results_directory, configuration, selection, track)
            for track, _role in track_choices
            for selection in selections
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def plot(path, output_directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported trajectory schema version")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for track_seed in sorted(
        {item["evaluation_seed"] for item in payload["trajectories"]}
    ):
        figure, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        for trajectory in payload["trajectories"]:
            if trajectory["evaluation_seed"] != track_seed:
                continue
            samples = trajectory["samples"]
            times = [sample["time_s"] for sample in samples]
            selection = trajectory["selection"]
            label = (
                f"{selection['algorithm'].upper()}/s{selection['training_seed']} "
                f"({selection['role']})"
            )
            line = axes[0].plot(
                times, [sample["position_m"] for sample in samples], label=label
            )[0]
            color = line.get_color()
            axes[1].plot(
                times,
                [sample["velocity_m_s"] for sample in samples],
                color=color,
            )
            axes[1].plot(
                times,
                [sample["speed_limit_m_s"] for sample in samples],
                color=color,
                linestyle="--",
                alpha=0.45,
            )
            for axis, field in zip(
                axes[2:],
                ("acceleration_m_s2", "action", "cumulative_energy_kwh"),
            ):
                axis.plot(times, [sample[field] for sample in samples], color=color)
        for axis, label in zip(
            axes,
            (
                "position [m]",
                "velocity / limit [m/s]",
                "acceleration [m/s²]",
                "action",
                "cumulative energy [kWh]",
            ),
        ):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize="x-small", ncols=2)
        axes[-1].set_xlabel("time [s]")
        figure.suptitle(f"SB3 representative trajectories: track {track_seed}")
        figure.tight_layout()
        destination = output / f"representative-trajectories-track-{track_seed}.png"
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        paths.append(destination)
    return tuple(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    path = record(args.results, args.trajectory_json, config_path=args.config)
    print(path)
    for plot_path in plot(path, args.output_dir):
        print(plot_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
