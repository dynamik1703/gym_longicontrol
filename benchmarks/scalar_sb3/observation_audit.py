"""Record declared and empirically observed input ranges before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.scalar_sac.experiment import _base_environment

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration


def audit(configuration) -> dict:
    environment = _base_environment(configuration)
    generator = np.random.default_rng(73_001)
    observed_min = np.full(environment.observation_space.shape, np.inf)
    observed_max = np.full(environment.observation_space.shape, -np.inf)
    count = 0
    try:
        for seed in configuration.track_splits.development_calibration:
            observation, _ = environment.reset(seed=seed)
            while True:
                values = np.asarray(observation, dtype=np.float64)
                observed_min = np.minimum(observed_min, values)
                observed_max = np.maximum(observed_max, values)
                count += 1
                action = np.array([generator.uniform(-1.0, 1.0)])
                observation, _, terminated, truncated, _ = environment.step(action)
                if terminated or truncated:
                    break
    finally:
        environment.close()
    return {
        "schema_version": 1,
        "configuration_sha256": configuration_sha256(configuration),
        "track_split": "development-2000-2008-v1",
        "policy": "uniform random actions from numpy seed 73001",
        "observation_count": count,
        "declared_low": environment.observation_space.low.tolist(),
        "declared_high": environment.observation_space.high.tolist(),
        "observed_min": observed_min.tolist(),
        "observed_max": observed_max.tolist(),
        "vec_normalize_used": False,
        "reason": (
            "The historical eight-feature observation is already scaled to [0, 1]."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(load_configuration(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
