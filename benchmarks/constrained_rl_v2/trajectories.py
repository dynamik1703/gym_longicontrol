"""Record deterministic native-step V2 trajectories with deadline diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks.scalar_sac.experiment import _base_environment

from .adapter import (
    build_agent,
    deterministic_policy_adapter,
    load_checkpoint,
    multiplier_values,
)
from .analysis import behavior_category
from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .costs import DenseDeadlineTaskWrapper, deadline_deficit_cost, deadline_state
from .results import load_result

TRAJECTORY_SCHEMA_VERSION = 1


def select_training_seed(results_directory, configuration) -> int:
    """Select median final Validation RSR, breaking ties by lower seed."""

    rows = []
    for seed in configuration.training_seeds:
        path = (
            Path(results_directory)
            / f"training-seed-{seed}"
            / f"target-{configuration.simulator_step_budget:06d}"
            / "validation-3000-3008-result.json"
        )
        run = load_result(path)
        rows.append((run.summary.requirement_satisfaction_rate, seed))
    values = sorted(value for value, _seed in rows)
    median_rsr = values[len(values) // 2]
    return min(seed for value, seed in rows if value == median_rsr)


def select_tracks(run) -> tuple[int, ...]:
    """Select lowest feasible and lowest completed speed-violating track."""

    feasible = sorted(
        episode.evaluation_seed for episode in run.episodes if episode.feasible
    )
    speed_violating = sorted(
        episode.evaluation_seed
        for episode in run.episodes
        if episode.completed and episode.max_speed_violation_m_s > 0
    )
    tracks = []
    if feasible:
        tracks.append(feasible[0])
    if speed_violating:
        tracks.append(speed_violating[0])
    if len(tracks) < 2:
        for episode in sorted(run.episodes, key=lambda item: item.evaluation_seed):
            if episode.evaluation_seed not in tracks:
                tracks.append(episode.evaluation_seed)
            if len(tracks) == 2:
                break
    return tuple(tracks)


def _record_one(policy, multipliers, configuration, expected_run, track_seed):
    expected = next(
        episode
        for episode in expected_run.episodes
        if episode.evaluation_seed == track_seed
    )
    environment = _base_environment(configuration)
    act = deterministic_policy_adapter(policy)
    try:
        observation, _ = environment.reset(seed=track_seed)
        samples = []
        cumulative_deadline_cost_s = 0.0
        previous_time_s = 0.0
        while True:
            action = act(np.asarray(observation, dtype=np.float32))
            observation, _reward, terminated, truncated, info = environment.step(action)
            elapsed_time_s = float(info["elapsed_time_s"])
            dt_s = elapsed_time_s - previous_time_s
            previous_time_s = elapsed_time_s
            base = environment.unwrapped
            remaining_s, slack_s, deficit_s = deadline_state(
                elapsed_time_s=elapsed_time_s,
                position_m=float(info["position_m"]),
                track=base.track,
                track_length_m=float(base.config.track_length_m),
                deadline_s=configuration.task.max_time_s,
            )
            step_deadline_cost_s = deadline_deficit_cost(
                deficit_s=deficit_s,
                dt_s=dt_s,
                normalization_s=configuration.deadline_cost.normalization_s,
            )
            cumulative_deadline_cost_s += step_deadline_cost_s
            samples.append(
                {
                    "time_s": elapsed_time_s,
                    "position_m": float(info["position_m"]),
                    "velocity_m_s": float(info["velocity_m_s"]),
                    "speed_limit_m_s": float(info["speed_limit_m_s"]),
                    "acceleration_m_s2": float(info["acceleration_m_s2"]),
                    "jerk_m_s3": float(info["jerk_m_s3"]),
                    "action": float(action[0]),
                    "cumulative_energy_kwh": float(info["total_energy_kwh"]),
                    "optimistic_remaining_time_s": remaining_s,
                    "deadline_slack_s": slack_s,
                    "deadline_deficit_s": deficit_s,
                    "deadline_cost_s": step_deadline_cost_s,
                    "cumulative_deadline_cost_s": cumulative_deadline_cost_s,
                    "lambda_speed": multipliers[0],
                    "lambda_deadline": multipliers[1],
                }
            )
            if terminated or truncated:
                metrics = info["episode_metrics"]
                if bool(metrics["completed"]) != expected.completed or not np.isclose(
                    metrics["energy_kwh"], expected.energy_kwh
                ):
                    raise RuntimeError(
                        "V2 trajectory replay differs from stored result"
                    )
                return {
                    "evaluation_seed": track_seed,
                    "feasible": expected.feasible,
                    "behavior_category": behavior_category(
                        expected, expected_run.task
                    ),
                    "travel_time_s": expected.travel_time_s,
                    "energy_kwh": expected.energy_kwh,
                    "max_speed_violation_m_s": expected.max_speed_violation_m_s,
                    "integrated_speed_violation_m": (
                        expected.integrated_speed_violation_m
                    ),
                    "deadline_deficit_integral_s": cumulative_deadline_cost_s,
                    "checkpoint_multipliers": {
                        "speed": multipliers[0],
                        "deadline": multipliers[1],
                    },
                    "samples": samples,
                }
    finally:
        environment.close()


def record(
    results_directory: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
) -> Path:
    configuration = load_configuration(config_path)
    seed = select_training_seed(results_directory, configuration)
    run_directory = Path(results_directory) / f"training-seed-{seed}"
    checkpoint_directory = (
        run_directory / f"target-{configuration.simulator_step_budget:06d}"
    )
    expected = load_result(
        checkpoint_directory / "validation-3000-3008-result.json"
    )
    tracks = select_tracks(expected)
    construction_environment = DenseDeadlineTaskWrapper(
        _base_environment(configuration),
        task=configuration.task,
        energy_scale_kwh=configuration.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.deadline_cost.normalization_s,
    )
    try:
        agent, _logger = build_agent(
            configuration,
            construction_environment,
            training_seed=seed,
            device=device,
            threads=1,
        )
        metadata = load_checkpoint(
            checkpoint_directory / "policy.pt", agent.policy, device=device
        )
        if metadata["configuration_sha256"] != configuration_sha256(configuration):
            raise ValueError("V2 trajectory checkpoint configuration differs")
        agent.policy.eval()
        multipliers = multiplier_values(agent.policy)
        payload = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "configuration_sha256": configuration_sha256(configuration),
            "selection_rule": (
                "median final validation RSR across training seeds; lowest seed "
                "breaks ties; then lowest feasible and lowest completed "
                "speed-violating validation tracks, with ascending-track fallback"
            ),
            "training_seed": seed,
            "selected_tracks": tracks,
            "checkpoint_metadata": metadata,
            "trajectories": [
                _record_one(
                    agent.policy,
                    multipliers,
                    configuration,
                    expected,
                    track,
                )
                for track in tracks
            ],
            "task": asdict(configuration.task),
        }
    finally:
        construction_environment.close()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    print(
        record(
            args.results,
            args.output,
            config_path=args.config,
            device=args.device,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
