"""Interpretable reference policies for the fixed benchmark tracks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gym_longicontrol.domain.metrics import EpisodeMetrics

from .config import DEFAULT_CONFIG_PATH, configuration_sha256, load_configuration
from .evaluation import EpisodeEvaluation, EpisodeRecorder, EvaluationSummary

REFERENCE_SCHEMA_VERSION = 1
RANDOM_POLICY_SEED = 73_000


def privileged_speed_action(
    environment: Any,
    info: Mapping[str, Any],
    *,
    margin_m_s: float,
    braking_m_s2: float,
) -> np.ndarray:
    """Track-aware speed action used only as an oracle-style physical anchor."""

    base = environment.unwrapped
    position_m = float(info["position_m"])
    velocity_m_s = float(info["velocity_m_s"])
    speed_caps = [max(0.0, float(info["speed_limit_m_s"]) - margin_m_s)]
    for position, limit in zip(base.track.positions_m, base.track.limits_m_s):
        if position > position_m:
            distance = max(0.0, float(position - position_m) - 2.0)
            target = max(0.0, float(limit) - margin_m_s)
            speed_caps.append(
                float(np.sqrt(target**2 + 2.0 * braking_m_s2 * distance))
            )
    target_speed = min(speed_caps)
    desired_acceleration = float(
        np.clip(2.0 * (target_speed - velocity_m_s), -3.0, 3.0)
    )
    coast = base.vehicle.acceleration_from_action(velocity_m_s, 0.0)
    if desired_acceleration >= coast:
        action_limit = base.vehicle.acceleration_from_action(velocity_m_s, 1.0)
        action = (desired_acceleration - coast) / max(action_limit - coast, 1e-12)
    else:
        action_limit = base.vehicle.acceleration_from_action(velocity_m_s, -1.0)
        action = -(coast - desired_acceleration) / max(coast - action_limit, 1e-12)
    return np.array([np.clip(action, -1.0, 1.0)])


@dataclass(frozen=True)
class ReferencePolicyResult:
    policy_id: str
    description: str
    privileged_track_access: bool
    random_policy_seed: int | None
    episodes: tuple[EpisodeEvaluation, ...]
    summary: EvaluationSummary


@dataclass(frozen=True)
class ReferenceResults:
    benchmark_name: str
    configuration_sha256: str
    evaluation_set_id: str
    task: dict[str, float]
    policies: tuple[ReferencePolicyResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": REFERENCE_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReferenceResults:
        if raw.get("schema_version") != REFERENCE_SCHEMA_VERSION:
            raise ValueError("Unsupported reference result schema version")
        from gym_longicontrol.domain.task import TaskSpecification

        task = TaskSpecification(**raw["task"])
        policies = []
        for item in raw["policies"]:
            episodes = tuple(EpisodeEvaluation(**row) for row in item["episodes"])
            summary = EvaluationSummary(**item["summary"])
            if summary != EvaluationSummary.from_episodes(episodes, task):
                raise ValueError("Stored reference summary does not match episodes")
            policies.append(
                ReferencePolicyResult(
                    policy_id=item["policy_id"],
                    description=item["description"],
                    privileged_track_access=item["privileged_track_access"],
                    random_policy_seed=item["random_policy_seed"],
                    episodes=episodes,
                    summary=summary,
                )
            )
        return cls(
            benchmark_name=raw["benchmark_name"],
            configuration_sha256=raw["configuration_sha256"],
            evaluation_set_id=raw["evaluation_set_id"],
            task=dict(raw["task"]),
            policies=tuple(policies),
        )


def _environment(configuration):
    import gymnasium as gym

    import gym_longicontrol  # noqa: F401

    return gym.make(
        configuration.environment_id,
        max_episode_steps=configuration.max_episode_steps,
    )


def _evaluate(configuration, policy_id, description, privileged, action_factory):
    environment = _environment(configuration)
    episodes = []
    try:
        for seed in configuration.evaluation_seeds:
            _observation, info = environment.reset(seed=seed)
            action_policy = action_factory(environment, seed)
            recorder = EpisodeRecorder()
            while True:
                action = action_policy(info)
                _, _, terminated, truncated, info = environment.step(action)
                recorder.observe(action, info)
                if terminated or truncated:
                    metrics = EpisodeMetrics(**info["episode_metrics"])
                    episodes.append(
                        recorder.finalize(seed, metrics, configuration.task)
                    )
                    break
    finally:
        environment.close()
    values = tuple(episodes)
    return ReferencePolicyResult(
        policy_id=policy_id,
        description=description,
        privileged_track_access=privileged,
        random_policy_seed=(RANDOM_POLICY_SEED if policy_id == "random" else None),
        episodes=values,
        summary=EvaluationSummary.from_episodes(values, configuration.task),
    )


def evaluate_references(configuration) -> ReferenceResults:
    definitions = (
        (
            "fast-compliant-oracle",
            "0.5 m/s margin; full future-track access",
            True,
            lambda environment, _seed: lambda info: privileged_speed_action(
                environment, info, margin_m_s=0.5, braking_m_s2=0.75
            ),
        ),
        (
            "conservative-oracle",
            "3.0 m/s margin; full future-track access",
            True,
            lambda environment, _seed: lambda info: privileged_speed_action(
                environment, info, margin_m_s=3.0, braking_m_s2=0.75
            ),
        ),
        (
            "random",
            "uniform actions in [-1, 1]; no privileged track access",
            False,
            lambda _environment, seed: _random_policy(seed),
        ),
    )
    policies = tuple(
        _evaluate(configuration, policy_id, description, privileged, factory)
        for policy_id, description, privileged, factory in definitions
    )
    return ReferenceResults(
        benchmark_name=configuration.name,
        configuration_sha256=configuration_sha256(configuration),
        evaluation_set_id=configuration.evaluation_set_id,
        task=asdict(configuration.task),
        policies=policies,
    )


def _random_policy(seed: int):
    generator = np.random.default_rng(RANDOM_POLICY_SEED + seed)
    return lambda _info: np.array([generator.uniform(-1.0, 1.0)])


def save_reference_results(path: str | Path, results: ReferenceResults) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(results.to_dict(), stream, indent=2, sort_keys=True)
        stream.write("\n")
    return destination


def load_reference_results(path: str | Path) -> ReferenceResults:
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("Reference result root must be an object")
    return ReferenceResults.from_dict(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configuration = load_configuration(args.config)
    results = evaluate_references(configuration)
    path = save_reference_results(args.output, results)
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a script
    raise SystemExit(main())
