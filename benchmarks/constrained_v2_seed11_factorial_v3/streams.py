"""Exact runtime training streams and transition/episode instrumentation."""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np

from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import ALL_NUMERICALLY_RESERVED
from benchmarks.scalar_sac.experiment import _base_environment

CURRENT_DOMAIN_TAG = 0x4D564531


class CurrentTrainingSeedStream:
    """The exact checkpointable current matched Real-Only reset stream."""

    def __init__(self, training_seed: int):
        self.training_seed = int(training_seed)
        self.rng = np.random.default_rng(
            np.random.SeedSequence([self.training_seed, CURRENT_DOMAIN_TAG])
        )
        self.draw_count = 0

    def next_seed(self) -> int:
        while True:
            candidate = int(self.rng.integers(0, 2**31 - 1))
            if candidate not in ALL_NUMERICALLY_RESERVED:
                self.draw_count += 1
                return candidate

    def state_dict(self) -> dict[str, Any]:
        return {
            "training_seed": self.training_seed,
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "draw_count": self.draw_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["training_seed"]) != self.training_seed:
            raise ValueError("Current stream training seed changed")
        self.rng.bit_generator.state = state["rng"]
        self.draw_count = int(state["draw_count"])


class _FingerprintingTrackWrapper(gym.Wrapper):
    stream_name: str

    def __init__(self, environment: gym.Env):
        super().__init__(environment)
        self.reset_records: list[dict[str, Any]] = []
        self.current_exposure: dict[str, Any] | None = None

    def _record(self, *, reset_seed: int | None, draw_index: int) -> None:
        base = self.env.unwrapped
        fingerprint = fingerprint_track(
            base.track, v2_semantics(), numeric_seed=reset_seed
        )
        row = {
            "stream": self.stream_name,
            "episode_index": len(self.reset_records),
            "reset_seed": reset_seed,
            "rng_draw_index": int(draw_index),
            "route_sha256": fingerprint.route_sha256,
            "full_task_sha256": fingerprint.full_task_sha256,
        }
        self.reset_records.append(row)
        self.current_exposure = row


class HistoricalTrainingTrackWrapper(_FingerprintingTrackWrapper):
    """Continuous seed-11 RNG with historical used draws 0, 2, 4, ... ."""

    stream_name = "T_HIST"

    def __init__(self, environment: gym.Env, training_seed: int):
        super().__init__(environment)
        self.training_seed = int(training_seed)
        self.initialized = False
        self.next_used_draw_index = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise ValueError("Historical training reset seed is stream-owned")
        if not self.initialized:
            result = self.env.reset(seed=self.training_seed, options=options)
            self.initialized = True
        else:
            self.env.reset(seed=None, options=options)
            result = self.env.reset(seed=None, options=options)
        self._record(reset_seed=None, draw_index=self.next_used_draw_index)
        self.next_used_draw_index += 2
        return result


class CurrentTrainingTrackWrapper(_FingerprintingTrackWrapper):
    """One explicit reset seed from the exact current stream per episode."""

    stream_name = "T_CURRENT"

    def __init__(self, environment: gym.Env, training_seed: int):
        super().__init__(environment)
        self.stream = CurrentTrainingSeedStream(training_seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            raise ValueError("Current training reset seed is stream-owned")
        reset_seed = self.stream.next_seed()
        result = self.env.reset(seed=reset_seed, options=options)
        self._record(reset_seed=reset_seed, draw_index=self.stream.draw_count - 1)
        return result


def find_track_wrapper(environment: Any) -> _FingerprintingTrackWrapper:
    current = environment
    while not isinstance(current, _FingerprintingTrackWrapper):
        if not hasattr(current, "env"):
            raise RuntimeError("Training track wrapper is absent")
        current = current.env
    return current


class TrainingRecorder(gym.Wrapper):
    """Attach transition IDs and persist task exposure at first physical use."""

    def __init__(self, environment: gym.Env):
        super().__init__(environment)
        self.episode_step = 0
        self.transition_id = 0
        self.completed_episodes = 0
        self.used_exposures: list[dict[str, Any]] = []
        self.last_completed_episode: Any | None = None
        self.last_completed_exposure: dict[str, Any] | None = None
        self._current_exposure_used = False

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.episode_step = 0
        self._current_exposure_used = False
        return result

    def step(self, action):
        wrapper = find_track_wrapper(self.env)
        if wrapper.current_exposure is None:
            raise RuntimeError("A transition has no training-track exposure")
        if not self._current_exposure_used:
            self.used_exposures.append(copy.deepcopy(wrapper.current_exposure))
            self._current_exposure_used = True
        observation, reward, terminated, truncated, raw_info = self.env.step(action)
        self.episode_step += 1
        info = dict(raw_info)
        info["real_transition_id"] = self.transition_id
        self.transition_id += 1
        if terminated or truncated:
            completed = getattr(self.env, "last_completed_episode", None)
            if completed is None:
                raise RuntimeError("Completed physical episode diagnostics were lost")
            self.completed_episodes += 1
            self.last_completed_episode = completed
            self.last_completed_exposure = copy.deepcopy(wrapper.current_exposure)
        return observation, reward, terminated, truncated, info


def build_track_wrapper(
    environment: gym.Env, *, stream_name: str, training_seed: int
) -> _FingerprintingTrackWrapper:
    if stream_name == "T_HIST":
        return HistoricalTrainingTrackWrapper(environment, training_seed)
    if stream_name == "T_CURRENT":
        return CurrentTrainingTrackWrapper(environment, training_seed)
    raise ValueError(f"Unknown training stream: {stream_name}")


def build_training_environment(
    configuration: Any,
    *,
    stream_name: str,
    training_seed: int,
    max_simulator_steps: int,
) -> TrainingRecorder:
    base = _base_environment(configuration.v2)
    tracked = build_track_wrapper(
        base, stream_name=stream_name, training_seed=training_seed
    )
    dense = DenseDeadlineTaskWrapper(
        tracked,
        task=configuration.v2.task,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
        max_simulator_steps=max_simulator_steps,
    )
    return TrainingRecorder(dense)
