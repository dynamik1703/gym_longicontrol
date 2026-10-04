"""Select and replay final binary-reward policies without cherry-picking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.scalar_sac.analysis import failure_mode
from benchmarks.scalar_sac.experiment import _base_environment

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .results import load_result


def _final_runs(results_directory, configuration):
    target = configuration.total_training_steps
    return tuple(
        load_result(
            Path(results_directory)
            / f"training-seed-{seed}"
            / f"step-{target:06d}"
            / "validation-result.json"
        )
        for seed in configuration.training_seeds
    )


def select_tracks(runs) -> tuple[tuple[int, str], ...]:
    """Select common-success and maximum-failure-diversity tracks."""

    seeds = [item.evaluation_seed for item in runs[0].episodes]
    common = min(
        seeds,
        key=lambda seed: (
            -sum(
                next(
                    item.feasible
                    for item in run.episodes
                    if item.evaluation_seed == seed
                )
                for run in runs
            ),
            seed,
        ),
    )
    diverse = min(
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
                    for run in runs
                }
            ),
            seed,
        ),
    )
    progress = min(
        seeds,
        key=lambda seed: (
            -max(
                (
                    item.final_position_m
                    for run in runs
                    for item in run.episodes
                    if item.evaluation_seed == seed and not item.feasible
                ),
                default=-1.0,
            ),
            seed,
        ),
    )
    selected = {}
    for seed, role in (
        (common, "maximum common feasibility"),
        (diverse, "maximum distinct final failure modes"),
        (progress, "maximum progress among final failures"),
    ):
        selected.setdefault(seed, role)
    return tuple(selected.items())


def _load_model(results_directory, configuration, seed):
    from stable_baselines3 import SAC

    path = (
        Path(results_directory)
        / f"training-seed-{seed}"
        / f"step-{configuration.total_training_steps:06d}"
        / "model.zip"
    )
    return SAC.load(path, device="cpu")


def _record_one(model, configuration, expected, seed, track):
    environment = _base_environment(configuration)
    try:
        observation, _ = environment.reset(seed=track)
        samples = []
        while True:
            action, _state = model.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, info = environment.step(action)
            samples.append(
                {
                    "time_s": float(info["elapsed_time_s"]),
                    "position_m": float(info["position_m"]),
                    "velocity_m_s": float(info["velocity_m_s"]),
                    "speed_limit_m_s": float(info["speed_limit_m_s"]),
                    "action": float(np.asarray(action).reshape((1,))[0]),
                    "acceleration_m_s2": float(info["acceleration_m_s2"]),
                    "jerk_m_s3": float(info["jerk_m_s3"]),
                    "cumulative_energy_kwh": float(info["total_energy_kwh"]),
                    "speed_excess_m_s": float(info["speed_excess_m_s"]),
                }
            )
            if terminated or truncated:
                metrics = info["episode_metrics"]
                if (
                    bool(metrics["completed"]) != expected.completed
                    or not np.isclose(metrics["energy_kwh"], expected.energy_kwh)
                    or not np.isclose(metrics["travel_time_s"], expected.travel_time_s)
                ):
                    raise RuntimeError("Binary trajectory replay differs from result")
                return {
                    "training_seed": seed,
                    "evaluation_seed": track,
                    "feasible": expected.feasible,
                    "failure_mode": failure_mode(expected, configuration.task),
                    "completed": expected.completed,
                    "travel_time_s": expected.travel_time_s,
                    "energy_kwh": expected.energy_kwh,
                    "max_speed_violation_m_s": expected.max_speed_violation_m_s,
                    "integrated_speed_violation_m": (
                        expected.integrated_speed_violation_m
                    ),
                    "samples": samples,
                }
    finally:
        environment.close()


def record(
    results_directory: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> Path:
    configuration = load_configuration(config_path)
    runs = _final_runs(results_directory, configuration)
    tracks = select_tracks(runs)
    trajectories = []
    for run in runs:
        model = _load_model(results_directory, configuration, run.training_seed)
        for track, _role in tracks:
            expected = next(
                item for item in run.episodes if item.evaluation_seed == track
            )
            trajectories.append(
                _record_one(
                    model,
                    configuration,
                    expected,
                    run.training_seed,
                    track,
                )
            )
    payload = {
        "schema_version": 1,
        "configuration_sha256": configuration_sha256(configuration),
        "selection_rule": (
            "All final training seeds on the lowest Validation tracks maximizing "
            "common feasibility, distinct failure modes, and progress among "
            "failures; duplicate track choices are collapsed"
        ),
        "track_roles": {str(track): role for track, role in tracks},
        "training_seeds": configuration.training_seeds,
        "trajectories": trajectories,
        "reserved_final_test_evaluated": False,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(record(args.results_directory, args.output, config_path=args.config))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
