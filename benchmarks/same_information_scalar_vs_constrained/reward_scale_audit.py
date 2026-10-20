"""Reconstruct immutable replay signals for the scalarization scale audit.

This utility only reads retained reference checkpoints.  It never constructs an
environment, policy, collector, or evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import CHECKPOINTS, TRAINING_SEEDS, load_configuration

HERE = Path(__file__).resolve().parent


def _statistics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError("Scale-audit values must be a finite nonempty vector")
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "max_abs": float(np.max(np.abs(values))),
        "nonzero_fraction": float(np.count_nonzero(values) / len(values)),
    }


def _checkpoint_path(root: Path, condition: str, step: int) -> Path:
    return root / condition / f"step-{step:06d}" / "checkpoint.pt"


def _seed_signals(
    root: Path, condition: str, *, budget: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    objective = np.full(budget, np.nan, dtype=np.float64)
    speed = np.full(budget, np.nan, dtype=np.float64)
    deadline = np.full(budget, np.nan, dtype=np.float64)
    seen = np.zeros(budget, dtype=np.bool_)
    for step in CHECKPOINTS:
        payload = torch.load(
            _checkpoint_path(root, condition, step),
            map_location="cpu",
            weights_only=False,
        )
        replay = payload["replay_buffer"]
        indices = replay.sample_indices(0)
        batch = replay[indices]
        transition_ids = np.asarray(
            batch.info.real_transition_id, dtype=np.int64
        ).reshape(-1)
        actual_objective = np.asarray(
            batch.info.objective_reward, dtype=np.float64
        ).reshape(-1)
        costs = np.asarray(batch.info.cost, dtype=np.float64)
        if costs.shape != (len(transition_ids), 2):
            raise RuntimeError("Retained replay cost shape changed")
        if (
            np.any(transition_ids < 0)
            or np.any(transition_ids >= budget)
            or len(np.unique(transition_ids)) != len(transition_ids)
        ):
            raise RuntimeError("Invalid real_transition_id coverage in checkpoint")
        prior = seen[transition_ids]
        if prior.any():
            prior_ids = transition_ids[prior]
            if not (
                np.array_equal(objective[prior_ids], actual_objective[prior])
                and np.array_equal(speed[prior_ids], costs[prior, 0])
                and np.array_equal(deadline[prior_ids], costs[prior, 1])
            ):
                raise RuntimeError("Overlapping immutable replay signals disagree")
        objective[transition_ids] = actual_objective
        speed[transition_ids] = costs[:, 0]
        deadline[transition_ids] = costs[:, 1]
        seen[transition_ids] = True
    if not seen.all() or not (
        np.isfinite(objective).all()
        and np.isfinite(speed).all()
        and np.isfinite(deadline).all()
    ):
        raise RuntimeError("Immutable replay reconstruction has a gap")
    return objective, speed, deadline


def build_reward_scale_audit() -> dict[str, Any]:
    configuration = load_configuration()
    raw = configuration.raw
    scalar = raw["scalarization"]
    lambda_speed = float(scalar["lambda_speed_star"])
    lambda_deadline = float(scalar["lambda_deadline_star"])
    normalizer = float(scalar["constrained_actor_normalizer"])
    budget = int(raw["real_transition_budget"])
    root = Path(raw["reference_root"])
    reconstructed: dict[int, tuple[np.ndarray, ...]] = {}
    by_seed: dict[str, Any] = {}
    for seed in TRAINING_SEEDS:
        objective, speed_cost, deadline_cost = _seed_signals(
            root,
            raw["reference_conditions"][str(seed)],
            budget=budget,
        )
        speed_term = lambda_speed * speed_cost
        deadline_term = lambda_deadline * deadline_cost
        reward_u = objective - speed_term - deadline_term
        reward_n = reward_u / normalizer
        reconstructed[seed] = (
            objective,
            speed_term,
            deadline_term,
            reward_n,
            reward_u,
        )
        by_seed[str(seed)] = {
            name: _statistics(values)
            for name, values in zip(
                (
                    "r_E",
                    "lambda_speed_c_speed",
                    "lambda_deadline_c_deadline",
                    "r_N",
                    "r_U",
                ),
                reconstructed[seed],
            )
        }
    pooled = {
        name: _statistics(
            np.concatenate([reconstructed[seed][index] for seed in TRAINING_SEEDS])
        )
        for index, name in enumerate(
            (
                "r_E",
                "lambda_speed_c_speed",
                "lambda_deadline_c_deadline",
                "r_N",
                "r_U",
            )
        )
    }
    relation_verified = all(
        np.allclose(rows[4], rows[3] * normalizer, rtol=1e-15, atol=1e-15)
        for rows in reconstructed.values()
    )
    return {
        "schema_version": 2,
        "source": (
            "All 900000 unique immutable transitions reconstructed from the six "
            "retained 50k-spaced checkpoints per TRANSITIONAL seed; "
            "real_transition_id coverage is exactly 0..299999 for each seed. No "
            "environment, policy, collector, or evaluator was run."
        ),
        "selected_candidate": "U",
        "lambda_speed_star": lambda_speed,
        "lambda_deadline_star": lambda_deadline,
        "constrained_actor_normalizer": normalizer,
        "candidate_equations": {
            "N": "(r_E - lambda_speed*c_speed - lambda_deadline*c_deadline) / Z",
            "U": "r_E - lambda_speed*c_speed - lambda_deadline*c_deadline",
        },
        "comparison": {
            "u_over_n_positive_scale_factor": normalizer,
            "elementwise_positive_scaling_verified": relation_verified,
            "floating_check_rtol": 1e-15,
            "floating_check_atol": 1e-15,
            "selection_used_performance_outcomes": False,
        },
        "by_seed": by_seed,
        "pooled": pooled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "reward_scale.json")
    args = parser.parse_args()
    payload = build_reward_scale_audit()
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
