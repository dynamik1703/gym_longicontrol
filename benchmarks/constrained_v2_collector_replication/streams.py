"""Runtime wrapper for the frozen route-clean canonical training streams."""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any

import gymnasium as gym

from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.scalar_sac.experiment import _base_environment

from .split import stream_rows


class CanonicalTrainingTrackWrapper(gym.Wrapper):
    """Consume the precomputed accepted CANO stream in immutable order."""

    def __init__(self, environment: gym.Env, training_seed: int):
        super().__init__(environment)
        self.training_seed = int(training_seed)
        self.rows = copy.deepcopy(stream_rows(self.training_seed))
        self.cursor = 0
        self.reset_records: list[dict[str, Any]] = []
        self.current_exposure: dict[str, Any] | None = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise ValueError("Canonical training reset seed is stream-owned")
        if self.cursor >= len(self.rows):
            raise RuntimeError("Frozen 865-task canonical stream exhausted")
        frozen = self.rows[self.cursor]
        result = self.env.reset(seed=int(frozen["reset_seed"]), options=options)
        fingerprint = fingerprint_track(
            self.env.unwrapped.track,
            v2_semantics(),
            numeric_seed=int(frozen["reset_seed"]),
        )
        if (
            fingerprint.route_sha256 != frozen["route_sha256"]
            or fingerprint.full_task_sha256 != frozen["full_task_sha256"]
        ):
            raise RuntimeError("Runtime canonical task differs from frozen stream")
        row = {"stream": f"CANONICAL_{self.training_seed}", **copy.deepcopy(frozen)}
        self.reset_records.append(row)
        self.current_exposure = row
        self.cursor += 1
        return result


class ExplicitFixtureTrackWrapper(CanonicalTrainingTrackWrapper):
    """Preflight-only explicit tasks; never used by a scientific policy."""

    def __init__(self, environment: gym.Env, reset_seeds: tuple[int, ...]):
        gym.Wrapper.__init__(self, environment)
        self.training_seed = -1
        self.rows = []
        for index, reset_seed in enumerate(reset_seeds):
            # Fingerprint after reset in ``reset``; placeholders are filled below.
            self.rows.append(
                {
                    "nominal_policy_seed": -1,
                    "accepted_task_index": index,
                    "raw_candidate_index": index,
                    "reset_seed": int(reset_seed),
                    "route_sha256": None,
                    "full_task_sha256": None,
                }
            )
        self.cursor = 0
        self.reset_records = []
        self.current_exposure = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise ValueError("Fixture reset seed is wrapper-owned")
        if self.cursor >= len(self.rows):
            raise RuntimeError("Preflight fixture stream exhausted")
        frozen = self.rows[self.cursor]
        result = self.env.reset(seed=int(frozen["reset_seed"]), options=options)
        fingerprint = fingerprint_track(
            self.env.unwrapped.track,
            v2_semantics(),
            numeric_seed=int(frozen["reset_seed"]),
        )
        frozen["route_sha256"] = fingerprint.route_sha256
        frozen["full_task_sha256"] = fingerprint.full_task_sha256
        row = {"stream": "RESOURCE_FIXTURE", **copy.deepcopy(frozen)}
        self.reset_records.append(row)
        self.current_exposure = row
        self.cursor += 1
        return result


def find_track_wrapper(environment: Any) -> CanonicalTrainingTrackWrapper:
    current = environment
    while not isinstance(current, CanonicalTrainingTrackWrapper):
        if not hasattr(current, "env"):
            raise RuntimeError("Canonical training-track wrapper is absent")
        current = current.env
    return current


class TrainingRecorder(gym.Wrapper):
    """Record physical task exposure and actor-version stationarity."""

    def __init__(self, environment: gym.Env):
        super().__init__(environment)
        self.episode_step = 0
        self.transition_id = 0
        self.completed_episodes = 0
        self.used_exposures: list[dict[str, Any]] = []
        self.last_completed_episode: Any | None = None
        self.last_completed_exposure: dict[str, Any] | None = None
        self.last_completed_actor_versions: list[int] = []
        self.current_actor_version = 0
        self.current_episode_actor_versions: list[int] = []
        self._current_exposure_used = False

    def set_actor_version(self, update_count: int) -> None:
        self.current_actor_version = int(update_count)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.episode_step = 0
        self.current_episode_actor_versions = []
        self._current_exposure_used = False
        return result

    def step(self, action):
        wrapper = find_track_wrapper(self.env)
        if wrapper.current_exposure is None:
            raise RuntimeError("A transition has no canonical task exposure")
        if not self._current_exposure_used:
            self.used_exposures.append(copy.deepcopy(wrapper.current_exposure))
            self._current_exposure_used = True
        observation, reward, terminated, truncated, raw_info = self.env.step(action)
        self.episode_step += 1
        self.current_episode_actor_versions.append(self.current_actor_version)
        info = dict(raw_info)
        info["real_transition_id"] = self.transition_id
        info["actor_update_version"] = self.current_actor_version
        self.transition_id += 1
        if terminated or truncated:
            completed = getattr(self.env, "last_completed_episode", None)
            if completed is None:
                raise RuntimeError("Completed physical episode diagnostics were lost")
            self.completed_episodes += 1
            self.last_completed_episode = completed
            self.last_completed_exposure = copy.deepcopy(wrapper.current_exposure)
            self.last_completed_actor_versions = list(
                self.current_episode_actor_versions
            )
        return observation, reward, terminated, truncated, info


def actor_stationarity(versions: list[int]) -> dict[str, Any]:
    if not versions:
        return {
            "transitions": 0,
            "distinct_actor_versions": 0,
            "stationary_policy_segments": 0,
            "maximum_stationary_run_transitions": 0,
            "mean_stationary_run_transitions": 0.0,
        }
    runs: list[int] = []
    previous = versions[0]
    length = 1
    for value in versions[1:]:
        if value == previous:
            length += 1
        else:
            runs.append(length)
            previous = value
            length = 1
    runs.append(length)
    return {
        "transitions": len(versions),
        "distinct_actor_versions": len(set(versions)),
        "stationary_policy_segments": len(runs),
        "maximum_stationary_run_transitions": max(runs),
        "mean_stationary_run_transitions": sum(runs) / len(runs),
    }


def exposure_summary(recorder: TrainingRecorder) -> dict[str, Any]:
    rows = recorder.used_exposures
    routes = Counter(row["route_sha256"] for row in rows)
    full = Counter(row["full_task_sha256"] for row in rows)
    return {
        "task_exposures": len(rows),
        "unique_routes": len(routes),
        "unique_full_tasks": len(full),
        "duplicate_routes": len(rows) - len(routes),
        "duplicate_full_tasks": len(rows) - len(full),
        "maximum_route_multiplicity": max(routes.values(), default=0),
        "maximum_full_task_multiplicity": max(full.values(), default=0),
    }


def build_training_environment(
    configuration: Any,
    *,
    training_seed: int,
    max_simulator_steps: int,
) -> TrainingRecorder:
    base = _base_environment(configuration.v2)
    tracked = CanonicalTrainingTrackWrapper(base, training_seed)
    dense = DenseDeadlineTaskWrapper(
        tracked,
        task=configuration.v2.task,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
        max_simulator_steps=max_simulator_steps,
    )
    return TrainingRecorder(dense)


def build_fixture_environment(
    configuration: Any,
    *,
    reset_seeds: tuple[int, ...],
    max_simulator_steps: int,
) -> TrainingRecorder:
    base = _base_environment(configuration.v2)
    tracked = ExplicitFixtureTrackWrapper(base, reset_seeds)
    dense = DenseDeadlineTaskWrapper(
        tracked,
        task=configuration.v2.task,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
        max_simulator_steps=max_simulator_steps,
    )
    return TrainingRecorder(dense)
