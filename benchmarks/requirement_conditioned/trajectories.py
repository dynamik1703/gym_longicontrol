"""Record deterministic trajectories for multiple requirements on the same tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gym_longicontrol.domain.metrics import EpisodeMetrics

from .adapter import (
    build_agent,
    deterministic_policy_adapter,
    load_checkpoint,
    multiplier_values,
)
from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .experiment import _environment
from .results import RequirementRunResult, load_result

TRAJECTORY_SCHEMA_VERSION = 1


def _final_result_path(results_directory, configuration, seed):
    return (
        Path(results_directory)
        / f"training-seed-{seed}"
        / f"target-{configuration.simulator_step_budget:06d}"
        / "validation-result.json"
    )


def select_training_seed(results_directory, configuration) -> int:
    """Select median final Validation RSR, breaking ties by lower seed."""

    rows = []
    for seed in configuration.training_seeds:
        run = load_result(_final_result_path(results_directory, configuration, seed))
        rows.append((run.overall_summary["requirement_satisfaction_rate"], seed))
    median_rsr = sorted(value for value, _seed in rows)[len(rows) // 2]
    return min(seed for value, seed in rows if value == median_rsr)


def select_tracks(run: RequirementRunResult) -> tuple[int, int]:
    """Select median- and maximum-sensitivity tracks without manual cherry-picking."""

    grouped = {}
    for episode in run.episodes:
        grouped.setdefault(episode.track_seed, []).append(episode.travel_time_s)
    rows = sorted((max(values) - min(values), seed) for seed, values in grouped.items())
    median_track = rows[len(rows) // 2][1]
    maximum_track = max(rows, key=lambda item: (item[0], -item[1]))[1]
    if median_track == maximum_track:
        median_track = next(seed for _value, seed in rows if seed != maximum_track)
    return median_track, maximum_track


def _record_one(policy, multipliers, configuration, expected, track_seed, margin):
    environment = _environment(configuration, sampler_seed=0)
    act = deterministic_policy_adapter(policy)
    try:
        observation, info = environment.reset(
            seed=track_seed, options={"requirement_margin_s": margin}
        )
        samples = []
        while True:
            action = act(np.asarray(observation, dtype=np.float32))
            observation, _reward, terminated, truncated, info = environment.step(action)
            samples.append(
                {
                    "time_s": float(info["elapsed_time_s"]),
                    "position_m": float(info["position_m"]),
                    "velocity_m_s": float(info["velocity_m_s"]),
                    "speed_limit_m_s": float(info["speed_limit_m_s"]),
                    "acceleration_m_s2": float(info["acceleration_m_s2"]),
                    "jerk_m_s3": float(info["jerk_m_s3"]),
                    "action": float(action[0]),
                    "cumulative_energy_kwh": float(info["total_energy_kwh"]),
                    "optimistic_remaining_time_s": float(
                        info["deadline_optimistic_remaining_time_s"]
                    ),
                    "deadline_slack_s": float(info["deadline_slack_s"]),
                    "deadline_deficit_s": float(info["deadline_deficit_s"]),
                    "cumulative_deadline_cost_s": float(
                        info["deadline_deficit_integral_s"]
                    ),
                }
            )
            if terminated or truncated:
                metrics = EpisodeMetrics(**info["episode_metrics"])
                if (
                    metrics.completed != expected.completed
                    or not np.isclose(metrics.energy_kwh, expected.energy_kwh)
                    or not np.isclose(metrics.travel_time_s, expected.travel_time_s)
                ):
                    raise RuntimeError(
                        "Requirement-conditioned replay differs from stored result"
                    )
                return {
                    "track_seed": track_seed,
                    "requirement_kind": expected.requirement_kind,
                    "requirement_margin_s": margin,
                    "t_min_start_s": expected.t_min_start_s,
                    "max_time_s": expected.max_time_s,
                    "completed": expected.completed,
                    "deadline_met": expected.deadline_met,
                    "speed_compliant": expected.speed_compliant,
                    "feasible": expected.feasible,
                    "travel_time_s": expected.travel_time_s,
                    "energy_kwh": expected.energy_kwh,
                    "max_speed_violation_m_s": expected.max_speed_violation_m_s,
                    "integrated_speed_violation_m": (
                        expected.integrated_speed_violation_m
                    ),
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
    expected_run = load_result(
        _final_result_path(results_directory, configuration, seed)
    )
    tracks = select_tracks(expected_run)
    checkpoint_directory = (
        Path(results_directory)
        / f"training-seed-{seed}"
        / f"target-{configuration.simulator_step_budget:06d}"
    )
    construction_environment = _environment(configuration, sampler_seed=0)
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
        expected_hash = configuration_sha256(configuration)
        if metadata["configuration_sha256"] != expected_hash:
            raise ValueError("Trajectory checkpoint configuration differs")
        agent.policy.eval()
        multipliers = multiplier_values(agent.policy)
        trajectories = []
        for track_seed in tracks:
            for margin in configuration.requirements.evaluation_margins_s:
                expected = next(
                    item
                    for item in expected_run.episodes
                    if item.track_seed == track_seed
                    and item.requirement_margin_s == margin
                )
                trajectories.append(
                    _record_one(
                        agent.policy,
                        multipliers,
                        configuration,
                        expected,
                        track_seed,
                        margin,
                    )
                )
        payload = {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "configuration_sha256": expected_hash,
            "selection_rule": (
                "median final validation RSR across training seeds with lower-seed "
                "tie break; median and maximum five-requirement travel-time-range "
                "validation tracks"
            ),
            "training_seed": seed,
            "selected_tracks": tracks,
            "requirements_plotted_s": (configuration.requirements.evaluation_margins_s),
            "checkpoint_metadata": metadata,
            "trajectories": trajectories,
            "reserved_final_test_evaluated": False,
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
    parser.add_argument("results_directory", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    print(
        record(
            args.results_directory,
            args.output,
            config_path=args.config,
            device=args.device,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
