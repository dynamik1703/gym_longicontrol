"""Reproduce the separate-track pilot used to choose 140 seconds."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from gym_longicontrol.domain.metrics import EpisodeMetrics

from .config import DEFAULT_CONFIG_PATH, load_configuration
from .references import privileged_speed_action


def run_profile(configuration, *, margin_m_s: float, braking_m_s2: float):
    import gymnasium as gym

    import gym_longicontrol  # noqa: F401

    environment = gym.make(
        configuration.environment_id,
        max_episode_steps=configuration.max_episode_steps,
    )
    episodes = []
    try:
        for seed in configuration.calibration_seeds:
            _, info = environment.reset(seed=seed)
            while True:
                action = privileged_speed_action(
                    environment,
                    info,
                    margin_m_s=margin_m_s,
                    braking_m_s2=braking_m_s2,
                )
                _, _, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    episodes.append(
                        {"evaluation_seed": seed, **info["episode_metrics"]}
                    )
                    break
    finally:
        environment.close()
    return episodes


def _summary(episodes):
    metrics = [
        EpisodeMetrics(
            **{
                key: value
                for key, value in item.items()
                if key != "evaluation_seed"
            }
        )
        for item in episodes
    ]
    times = np.array([item.travel_time_s for item in metrics])
    return {
        "episode_count": len(metrics),
        "completion_count": sum(item.completed for item in metrics),
        "zero_violation_count": sum(
            item.max_speed_violation_m_s == 0.0 for item in metrics
        ),
        "min_travel_time_s": float(times.min()),
        "median_travel_time_s": float(np.median(times)),
        "max_travel_time_s": float(times.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    profiles = {
        "fast_safe": {"margin_m_s": 0.5, "braking_m_s2": 0.75},
        "conservative": {"margin_m_s": 3.0, "braking_m_s2": 0.75},
    }
    result = {
        "benchmark": configuration.name,
        "task": asdict(configuration.task),
        "profiles": {},
    }
    for name, profile in profiles.items():
        episodes = run_profile(configuration, **profile)
        result["profiles"][name] = {
            "controller": profile,
            "summary": _summary(episodes),
            "episodes": episodes,
        }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
