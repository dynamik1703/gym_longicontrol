"""Replay and diagnose the frozen constrained-RL V2 validation episodes.

This module never trains a policy. It replays the existing 300k checkpoints and
derives event-level physical diagnostics from native 10-Hz simulator samples.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_rl_v2.adapter import (
    build_agent,
    deterministic_policy_adapter,
    load_checkpoint,
    multiplier_values,
)
from benchmarks.constrained_rl_v2.config import (
    DEFAULT_CONFIG_PATH,
    configuration_sha256,
    load_configuration,
)
from benchmarks.constrained_rl_v2.costs import deadline_deficit_cost, deadline_state
from benchmarks.constrained_rl_v2.results import load_result
from benchmarks.scalar_sac.experiment import _base_environment

DIAGNOSIS_SCHEMA_VERSION = 1
DEFAULT_V2_RESULTS = Path("runs/constrained-rl-v2-20261002")


@dataclass(frozen=True)
class SpeedLimitTransition:
    position_m: float
    previous_limit_m_s: float
    next_limit_m_s: float

    @property
    def change_m_s(self) -> float:
        return self.next_limit_m_s - self.previous_limit_m_s

    @property
    def kind(self) -> str:
        return "reduction" if self.change_m_s < 0 else "increase"


def _step_width(samples: Sequence[Mapping[str, Any]], index: int) -> float:
    previous_time = 0.0 if index == 0 else float(samples[index - 1]["time_s"])
    width = float(samples[index]["time_s"]) - previous_time
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Trajectory sample times must be finite and increasing")
    return width


def speed_limit_transitions(
    positions_m: Sequence[float], limits_m_s: Sequence[float]
) -> tuple[SpeedLimitTransition, ...]:
    """Return all actual speed-limit changes after the route origin."""

    positions = np.asarray(positions_m, dtype=np.float64)
    limits = np.asarray(limits_m_s, dtype=np.float64)
    if (
        positions.ndim != 1
        or positions.size == 0
        or positions.shape != limits.shape
        or positions[0] != 0
        or not np.isfinite(positions).all()
        or not np.isfinite(limits).all()
        or not (np.diff(positions) > 0).all()
        or not (limits > 0).all()
    ):
        raise ValueError("Invalid piecewise-constant speed-limit profile")
    return tuple(
        SpeedLimitTransition(float(position), float(previous), float(current))
        for position, previous, current in zip(positions[1:], limits[:-1], limits[1:])
        if current != previous
    )


def transition_crossing_time_s(
    samples: Sequence[Mapping[str, Any]], position_m: float
) -> float:
    """Linearly interpolate the first time the trajectory crosses a position."""

    previous_position = 0.0
    previous_time = 0.0
    for sample in samples:
        current_position = float(sample["position_m"])
        current_time = float(sample["time_s"])
        if current_position >= position_m:
            distance = current_position - previous_position
            if distance <= 0:
                return current_time
            fraction = (position_m - previous_position) / distance
            return previous_time + fraction * (current_time - previous_time)
        previous_position = current_position
        previous_time = current_time
    raise ValueError("Trajectory does not cross the requested position")


def extract_violation_events(
    samples: Sequence[Mapping[str, Any]],
    transitions: Sequence[SpeedLimitTransition] = (),
) -> tuple[dict[str, Any], ...]:
    """Extract contiguous strictly-positive speed-excess events.

    Each post-step sample represents the preceding integration interval. Event
    duration and integrated overspeed therefore use the exact interval widths.
    Distance is the physical position increment over violating intervals.
    """

    events: list[dict[str, Any]] = []
    start: int | None = None
    for index, sample in enumerate(samples):
        excess = max(
            0.0,
            float(sample["velocity_m_s"]) - float(sample["speed_limit_m_s"]),
        )
        if excess > 0 and start is None:
            start = index
        if start is not None and (excess == 0 or index + 1 == len(samples)):
            stop = index if excess > 0 and index + 1 == len(samples) else index - 1
            event_samples = samples[start : stop + 1]
            widths = [_step_width(samples, item) for item in range(start, stop + 1)]
            excesses = [
                max(
                    0.0,
                    float(item["velocity_m_s"]) - float(item["speed_limit_m_s"]),
                )
                for item in event_samples
            ]
            previous_position = (
                0.0 if start == 0 else float(samples[start - 1]["position_m"])
            )
            distance = float(event_samples[-1]["position_m"]) - previous_position
            first = event_samples[0]
            event: dict[str, Any] = {
                "start_sample_index": start,
                "end_sample_index": stop,
                "first_violation_time_s": float(first["time_s"]),
                "first_violation_position_m": float(first["position_m"]),
                "duration_s": float(sum(widths)),
                "distance_while_violating_m": distance,
                "peak_overspeed_m_s": float(max(excesses)),
                "integrated_overspeed_m": float(
                    sum(value * width for value, width in zip(excesses, widths))
                ),
                "local_speed_limit_m_s": float(first["speed_limit_m_s"]),
                "vehicle_speed_m_s": float(first["velocity_m_s"]),
                "acceleration_m_s2": float(first["acceleration_m_s2"]),
                "action": float(first["action"]),
                "deadline_slack_s": float(first["deadline_slack_s"]),
                "deadline_deficit_s": float(first["deadline_deficit_s"]),
                "speed_cost_m": float(
                    sum(value * width for value, width in zip(excesses, widths))
                ),
                "deadline_cost_s": float(first["deadline_cost_s"]),
                "lambda_speed": float(first["lambda_speed"]),
                "lambda_deadline": float(first["lambda_deadline"]),
            }
            if transitions:
                nearest = min(
                    transitions,
                    key=lambda item: abs(float(first["position_m"]) - item.position_m),
                )
                crossing_time = transition_crossing_time_s(samples, nearest.position_m)
                event["nearest_speed_limit_transition"] = {
                    **asdict(nearest),
                    "change_m_s": nearest.change_m_s,
                    "kind": nearest.kind,
                    "signed_distance_m": (
                        float(first["position_m"]) - nearest.position_m
                    ),
                    "signed_time_s": float(first["time_s"]) - crossing_time,
                    "crossing_time_s": crossing_time,
                }
            events.append(event)
            start = None
    return tuple(events)


def trajectory_smoothness(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Return deterministic physical and action smoothness diagnostics."""

    if not samples:
        raise ValueError("A trajectory must contain at least one sample")
    acceleration = np.asarray([float(item["acceleration_m_s2"]) for item in samples])
    jerk = np.asarray([float(item["jerk_m_s3"]) for item in samples])
    actions = np.asarray([float(item["action"]) for item in samples])
    signs = np.sign(actions)
    sign_changes = int(np.count_nonzero(signs[1:] * signs[:-1] < 0))
    return {
        "max_acceleration_m_s2": float(np.max(acceleration)),
        "max_deceleration_m_s2": float(np.min(acceleration)),
        "max_abs_jerk_m_s3": float(np.max(np.abs(jerk))),
        "action_variance": float(np.var(actions)),
        "action_sign_change_count": sign_changes,
        "action_total_variation": float(np.abs(np.diff(actions)).sum()),
    }


def _track_features(trajectory: Mapping[str, Any]) -> np.ndarray:
    limits = np.asarray(trajectory["track"]["limits_m_s"], dtype=np.float64)
    transitions = speed_limit_transitions(trajectory["track"]["positions_m"], limits)
    reductions = [-item.change_m_s for item in transitions if item.change_m_s < 0]
    return np.asarray(
        [
            len(transitions),
            len(reductions),
            sum(reductions),
            max(reductions, default=0.0),
            limits[0],
            np.mean(limits),
        ],
        dtype=np.float64,
    )


def comparison_track(
    failed: Mapping[str, Any], successful: Sequence[Mapping[str, Any]]
) -> int:
    """Choose the closest successful same-policy track by physical layout features."""

    if not successful:
        raise ValueError("A same-seed successful comparison is required")
    target = _track_features(failed)
    scales = np.asarray([5.0, 3.0, 20.0, 10.0, 10.0, 10.0])
    return int(
        min(
            successful,
            key=lambda item: (
                float(np.linalg.norm((_track_features(item) - target) / scales)),
                int(item["track_seed"]),
            ),
        )["track_seed"]
    )


def _record_trajectory(policy, multipliers, configuration, expected, track_seed):
    environment = _base_environment(configuration)
    act = deterministic_policy_adapter(policy)
    try:
        observation, _ = environment.reset(seed=track_seed)
        base = environment.unwrapped
        track = {
            "positions_m": [float(value) for value in base.track.positions_m],
            "limits_m_s": [float(value) for value in base.track.limits_m_s],
        }
        samples = []
        previous_time = 0.0
        while True:
            action = act(np.asarray(observation, dtype=np.float32))
            observation, _reward, terminated, truncated, info = environment.step(action)
            elapsed = float(info["elapsed_time_s"])
            dt_s = elapsed - previous_time
            previous_time = elapsed
            remaining, slack, deficit = deadline_state(
                elapsed_time_s=elapsed,
                position_m=float(info["position_m"]),
                track=base.track,
                track_length_m=float(base.config.track_length_m),
                deadline_s=configuration.task.max_time_s,
            )
            samples.append(
                {
                    "time_s": elapsed,
                    "position_m": float(info["position_m"]),
                    "velocity_m_s": float(info["velocity_m_s"]),
                    "speed_limit_m_s": float(info["speed_limit_m_s"]),
                    "speed_excess_m_s": float(info["speed_excess_m_s"]),
                    "acceleration_m_s2": float(info["acceleration_m_s2"]),
                    "jerk_m_s3": float(info["jerk_m_s3"]),
                    "action": float(action[0]),
                    "deadline_optimistic_remaining_time_s": remaining,
                    "deadline_slack_s": slack,
                    "deadline_deficit_s": deficit,
                    "speed_cost_m": float(info["speed_excess_m_s"]) * dt_s,
                    "deadline_cost_s": deadline_deficit_cost(
                        deficit_s=deficit,
                        dt_s=dt_s,
                        normalization_s=configuration.deadline_cost.normalization_s,
                    ),
                    "lambda_speed": float(multipliers[0]),
                    "lambda_deadline": float(multipliers[1]),
                }
            )
            if terminated or truncated:
                metrics = info["episode_metrics"]
                checks = (
                    (metrics["travel_time_s"], expected.travel_time_s),
                    (metrics["energy_kwh"], expected.energy_kwh),
                    (
                        metrics["max_speed_violation_m_s"],
                        expected.max_speed_violation_m_s,
                    ),
                    (
                        metrics["integrated_speed_violation_m"],
                        expected.integrated_speed_violation_m,
                    ),
                )
                if bool(metrics["completed"]) != expected.completed or any(
                    not np.isclose(actual, stored, atol=1e-10)
                    for actual, stored in checks
                ):
                    raise RuntimeError("Frozen V2 replay differs from stored result")
                return {
                    "track_seed": int(track_seed),
                    "feasible": bool(expected.feasible),
                    "travel_time_s": float(expected.travel_time_s),
                    "energy_kwh": float(expected.energy_kwh),
                    "speed_violation_count": int(expected.speed_violation_count),
                    "max_speed_violation_m_s": float(expected.max_speed_violation_m_s),
                    "integrated_speed_violation_m": float(
                        expected.integrated_speed_violation_m
                    ),
                    "track": track,
                    "samples": samples,
                }
    finally:
        environment.close()


def replay_frozen_v2(
    results_directory: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
) -> Path:
    """Replay all 27 final V2 Validation episodes without training."""

    configuration = load_configuration(config_path)
    payload: dict[str, Any] = {
        "schema_version": DIAGNOSIS_SCHEMA_VERSION,
        "configuration_sha256": configuration_sha256(configuration),
        "source": "frozen constrained RL V2 300k checkpoints",
        "trajectories": [],
    }
    for training_seed in configuration.training_seeds:
        checkpoint = (
            Path(results_directory)
            / f"training-seed-{training_seed}"
            / f"target-{configuration.simulator_step_budget:06d}"
        )
        expected = load_result(checkpoint / "validation-3000-3008-result.json")
        construction_environment = _base_environment(configuration)
        try:
            agent, _logger = build_agent(
                configuration,
                construction_environment,
                training_seed=training_seed,
                device=device,
                threads=1,
            )
            metadata = load_checkpoint(
                checkpoint / "policy.pt", agent.policy, device=device
            )
            if metadata["configuration_sha256"] != configuration_sha256(configuration):
                raise ValueError("Frozen checkpoint configuration hash differs")
            agent.policy.eval()
            multipliers = multiplier_values(agent.policy)
            for episode in expected.episodes:
                trajectory = _record_trajectory(
                    agent.policy,
                    multipliers,
                    configuration,
                    episode,
                    episode.evaluation_seed,
                )
                trajectory["training_seed"] = training_seed
                payload["trajectories"].append(trajectory)
        finally:
            construction_environment.close()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_V2_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        replay_frozen_v2(
            args.results, args.output, config_path=args.config, device=args.device
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
