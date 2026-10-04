"""Replay deterministically selected final policies at native simulator resolution."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np

from benchmarks.scalar_sac.analysis import failure_mode
from benchmarks.scalar_sac.experiment import _base_environment

from .action_repeat import ActionRepeat
from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import load_result

TRAJECTORY_SCHEMA_VERSION = 1
TRACKS = (3000, 3005)


@dataclass(frozen=True)
class Selection:
    condition_id: str
    condition_label: str
    training_seed: int
    action_repeat: int
    gamma: float
    result_path: str
    role: str = "median final validation RSR"


def _final_result_paths(results_directory, final_target):
    paths = Path(results_directory).glob(
        "*/training-seed-*/simulator-step-*/validation-3000-3008-result.json"
    )
    selected = []
    for path in sorted(paths):
        run = load_result(path)
        if run.simulator_step_target == final_target:
            selected.append((path, run))
    return selected


def select_representatives(results_directory: str | Path, configuration):
    final = _final_result_paths(
        results_directory, configuration.simulator_step_budget
    )
    if not final:
        raise ValueError("No final credit-assignment validation results found")
    selections = []
    for condition in configuration.conditions:
        candidates = [
            (path, run)
            for path, run in final
            if run.condition_id == condition.condition_id
        ]
        values = sorted(
            run.summary.requirement_satisfaction_rate for _path, run in candidates
        )
        middle = float(median(values))
        path, run = min(
            (
                (path, run)
                for path, run in candidates
                if run.summary.requirement_satisfaction_rate == middle
            ),
            key=lambda item: item[1].training_seed,
        )
        selections.append(
            Selection(
                condition_id=run.condition_id,
                condition_label=run.condition_label,
                training_seed=run.training_seed,
                action_repeat=run.action_repeat,
                gamma=run.gamma,
                result_path=str(path),
            )
        )
    return tuple(selections)


def _behavior(samples, expected, task):
    velocities = [item["velocity_m_s"] for item in samples]
    actions = [item["action"] for item in samples]
    if expected.final_position_m < 50 or fmean(velocities) < 0.5:
        return "standstill"
    if not expected.completed:
        return "crawling"
    if expected.max_speed_violation_m_s > task.max_speed_violation_m_s:
        return "overspeeding"
    mean_action_change = (
        fmean(abs(right - left) for left, right in zip(actions, actions[1:]))
        if len(actions) > 1
        else 0.0
    )
    if mean_action_change > 0.15:
        return "oscillatory action"
    if max(abs(item["jerk_m_s3"]) for item in samples) > 20:
        return "aggressive completion"
    return "smooth completion" if expected.feasible else failure_mode(expected, task)


def fmean(values):
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0


def _record_one(configuration, selection, track_seed):
    from stable_baselines3 import SAC

    result_path = Path(selection.result_path)
    result = load_result(result_path)
    expected = next(
        item for item in result.episodes if item.evaluation_seed == track_seed
    )
    model = SAC.load(result_path.parent / "model.zip", device="cpu")
    environment = ActionRepeat(
        _base_environment(configuration),
        selection.action_repeat,
        capture_trace=True,
    )
    try:
        observation, _ = environment.reset(seed=track_seed)
        samples = []
        agent_decisions = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            agent_decisions += 1
            samples.extend(info["action_repeat_trace"])
            if terminated or truncated:
                actual = info["episode_metrics"]
                if (
                    bool(actual["completed"]) != expected.completed
                    or not np.isclose(actual["energy_kwh"], expected.energy_kwh)
                    or len(samples) != expected.step_count
                ):
                    raise RuntimeError(
                        "Credit-assignment trajectory replay differs from result"
                    )
                return {
                    "selection": asdict(selection),
                    "evaluation_seed": track_seed,
                    "feasible": expected.feasible,
                    "failure_mode": failure_mode(expected, result.task),
                    "behavior_classification": _behavior(
                        samples, expected, result.task
                    ),
                    "travel_time_s": expected.travel_time_s,
                    "energy_kwh": expected.energy_kwh,
                    "max_speed_violation_m_s": expected.max_speed_violation_m_s,
                    "simulator_steps": len(samples),
                    "agent_decisions": agent_decisions,
                    "samples": samples,
                }
    finally:
        environment.close()


def record(results_directory, output_path, *, config_path=DEFAULT_CONFIG_PATH):
    configuration = load_configuration(config_path)
    selections = select_representatives(results_directory, configuration)
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "configuration_sha256": configuration_sha256(configuration),
        "selection_rule": (
            "For each condition, select the seed at the median final validation "
            "RSR; the lowest seed breaks an RSR tie. Replay preregistered "
            "validation tracks 3000 and 3005."
        ),
        "track_seeds": list(TRACKS),
        "trajectories": [
            _record_one(configuration, selection, track_seed)
            for track_seed in TRACKS
            for selection in selections
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def plot(path: str | Path, output_directory: str | Path) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("Unsupported trajectory schema version")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for track_seed in payload["track_seeds"]:
        figure, axes = plt.subplots(6, 1, figsize=(10, 14), sharex=True)
        for trajectory in payload["trajectories"]:
            if trajectory["evaluation_seed"] != track_seed:
                continue
            samples = trajectory["samples"]
            times = [sample["time_s"] for sample in samples]
            selection = trajectory["selection"]
            label = (
                f"{selection['condition_label']}/s{selection['training_seed']} "
                f"({trajectory['behavior_classification']})"
            )
            line = axes[0].plot(
                times, [sample["position_m"] for sample in samples], label=label
            )[0]
            color = line.get_color()
            axes[1].plot(
                times, [sample["velocity_m_s"] for sample in samples], color=color
            )
            axes[1].plot(
                times,
                [sample["speed_limit_m_s"] for sample in samples],
                color=color,
                linestyle="--",
                alpha=0.4,
            )
            for axis, field in zip(
                axes[2:],
                (
                    "acceleration_m_s2",
                    "jerk_m_s3",
                    "action",
                    "cumulative_energy_kwh",
                ),
            ):
                axis.plot(times, [sample[field] for sample in samples], color=color)
        for axis, label in zip(
            axes,
            (
                "position [m]",
                "velocity / limit [m/s]",
                "acceleration [m/s²]",
                "jerk [m/s³]",
                "action",
                "cumulative energy [kWh]",
            ),
        ):
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize="x-small", ncols=2)
        axes[-1].set_xlabel("physical time [s]")
        figure.suptitle(f"Credit-assignment trajectories: track {track_seed}")
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
