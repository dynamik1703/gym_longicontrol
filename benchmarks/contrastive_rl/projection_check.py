"""Bounded scripted check of projected-goal collisions; never trains a policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_longicontrol  # noqa: F401
from gym_longicontrol.domain.task import TaskSpecification

from .goal_adapter import physical_outcome, project_outcome, projected_goal_is_canonical
from .sampling import equivalent_goal_rate


def measure_projection_collisions(*, transitions: int) -> dict:
    if transitions <= 0 or transitions > 500:
        raise ValueError("revision check permits 1 to 500 simulator transitions")
    env = gym.make("StochasticTrack-v1")
    _, info = env.reset(seed=2001)
    raw_outcomes = []
    terminal_flags = []
    previous_position = float(info["position_m"])
    try:
        for _ in range(transitions):
            _, _, terminated, truncated, info = env.step(np.array([0.0]))
            if terminated or truncated:
                raise RuntimeError("stationary projection check ended unexpectedly")
            raw_outcomes.append(
                physical_outcome(
                    position_m=float(info["position_m"]),
                    previous_position_m=previous_position,
                    elapsed_time_s=float(info["elapsed_time_s"]),
                    max_speed_violation_m_s=float(
                        info["max_speed_violation_m_s"]
                    ),
                )
            )
            terminal_flags.append(bool(terminated))
            previous_position = float(info["position_m"])
    finally:
        env.close()

    raw = np.asarray(raw_outcomes)
    projected = project_outcome(
        raw,
        terminated=np.asarray(terminal_flags),
        task=TaskSpecification(140.0, 0.0),
    )
    return {
        "kind": "preparation_only_projection_collision_check",
        "development_seed": 2001,
        "scripted_action": 0.0,
        "native_simulator_transitions": transitions,
        "used_for_learning": False,
        "raw_unique_rows": int(len(np.unique(raw, axis=0))),
        "projected_unique_rows": int(len(np.unique(projected, axis=0))),
        "raw_equivalent_pair_rate": equivalent_goal_rate(raw),
        "projected_equivalent_pair_rate": equivalent_goal_rate(projected),
        "canonical_projected_outcomes": int(
            np.sum(projected_goal_is_canonical(projected))
        ),
        "training_performed": False,
        "validation_tracks_used": False,
        "paper_tracks_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transitions", type=int, default=500)
    args = parser.parse_args()
    result = measure_projection_collisions(transitions=args.transitions)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
