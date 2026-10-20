import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.gcsl_canonical.config import (  # noqa: E402
    load_configuration,
    require_main_training_authorized,
    require_validation_authorized,
)
from benchmarks.gcsl_canonical.execution import RunLease  # noqa: E402
from benchmarks.gcsl_canonical.progress import checkpoint_progress  # noqa: E402
from benchmarks.gcsl_canonical.runner import PolicyRunner  # noqa: E402


class TinyEpisodeEnvironment:
    """Pickle-safe deterministic fixture; no LongiControl split is opened."""

    def __init__(self, episode_length=5):
        self.episode_length = int(episode_length)
        self.step_index = 0
        self.reset_seed = None
        self.closed = False

    def reset(self, seed=None):
        self.step_index = 0
        self.reset_seed = int(seed) if seed is not None else None
        observation = np.zeros(8, dtype=np.float64)
        return observation, {
            "position_m": 0.0,
            "elapsed_time_s": 0.0,
            "max_speed_violation_m_s": 0.0,
        }

    def step(self, action):
        assert np.asarray(action).shape == (1,)
        self.step_index += 1
        position = 10.0 * self.step_index
        terminated = False
        truncated = self.step_index == self.episode_length
        observation = np.full(8, self.step_index, dtype=np.float64)
        info = {
            "position_m": position,
            "elapsed_time_s": 0.1 * self.step_index,
            "max_speed_violation_m_s": 0.0,
        }
        if truncated:
            info["episode_metrics"] = {
                "completed": False,
                "travel_time_s": 0.1 * self.step_index,
                "energy_kwh": 0.01 * self.step_index,
                "speed_violation_count": 0,
                "max_speed_violation_m_s": 0.0,
                "integrated_speed_violation_m": 0.0,
            }
        return observation, 999.0, terminated, truncated, info

    def close(self):
        self.closed = True


def make_runner(*, episode_length=5):
    return PolicyRunner(
        training_seed=11,
        attempt_id="attempt-fixture",
        configuration_sha256="a" * 64,
        source_hashes={"source": "b" * 64},
        environment=TinyEpisodeEnvironment(episode_length),
        verify_track_content=False,
    )


def assert_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_equal(first, second)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=0, abs=0)
    else:
        assert left == right


def restore(path: Path, sha256: str):
    return PolicyRunner.restore(
        path,
        expected_sha256=sha256,
        configuration_sha256="a" * 64,
        source_hashes={"source": "b" * 64},
        attempt_id="attempt-fixture",
        verify_track_content=False,
    )


def test_exact_resume_mid_episode(tmp_path):
    uninterrupted = make_runner(episode_length=5)
    uninterrupted.collect_only()
    uninterrupted.collect_only()
    path = tmp_path / "mid-episode.pkl"
    metadata = uninterrupted.save(path)
    resumed = restore(path, metadata["sha256"])
    try:
        for _ in range(3):
            uninterrupted.advance()
            resumed.advance()
        assert_equal(uninterrupted.exact_snapshot(), resumed.exact_snapshot())
    finally:
        uninterrupted.close()
        resumed.close()


def test_exact_resume_on_pending_update_boundary(tmp_path):
    uninterrupted = make_runner(episode_length=20)
    uninterrupted.collect_only()
    uninterrupted.transition_count = 10_039
    uninterrupted.collect_only()
    assert uninterrupted.pending_update is True
    path = tmp_path / "update-boundary.pkl"
    metadata = uninterrupted.save(path)
    resumed = restore(path, metadata["sha256"])
    try:
        uninterrupted.perform_pending_update()
        resumed.perform_pending_update()
        assert uninterrupted.update_count == resumed.update_count == 1
        assert_equal(uninterrupted.exact_snapshot(), resumed.exact_snapshot())
    finally:
        uninterrupted.close()
        resumed.close()


def test_exact_resume_on_episode_termination_reset_boundary(tmp_path):
    uninterrupted = make_runner(episode_length=2)
    uninterrupted.collect_only()
    uninterrupted.collect_only()
    assert uninterrupted.pending_reset is True
    path = tmp_path / "reset-boundary.pkl"
    metadata = uninterrupted.save(path)
    resumed = restore(path, metadata["sha256"])
    try:
        uninterrupted.reset_pending_episode()
        resumed.reset_pending_episode()
        uninterrupted.collect_only()
        resumed.collect_only()
        assert_equal(uninterrupted.exact_snapshot(), resumed.exact_snapshot())
    finally:
        uninterrupted.close()
        resumed.close()


def test_checkpoint_rejects_attempt_or_provenance_change(tmp_path):
    runner = make_runner()
    path = tmp_path / "checkpoint.pkl"
    metadata = runner.save(path)
    runner.close()
    with pytest.raises(RuntimeError, match="Attempt ID"):
        PolicyRunner.restore(
            path,
            expected_sha256=metadata["sha256"],
            configuration_sha256="a" * 64,
            source_hashes={"source": "b" * 64},
            attempt_id="different-attempt",
            verify_track_content=False,
        )
    with pytest.raises(RuntimeError, match="provenance"):
        PolicyRunner.restore(
            path,
            expected_sha256=metadata["sha256"],
            configuration_sha256="c" * 64,
            source_hashes={"source": "b" * 64},
            attempt_id="attempt-fixture",
            verify_track_content=False,
        )


def test_duplicate_lease_is_blocked(tmp_path):
    with RunLease(tmp_path):
        with pytest.raises(RuntimeError, match="duplicate"):
            with RunLease(tmp_path):
                pass


def test_authorization_is_fail_closed_for_training_and_validation():
    configuration = load_configuration()
    with pytest.raises(PermissionError, match="MAIN_TRAINING_NOT_AUTHORIZED"):
        require_main_training_authorized(configuration)
    with pytest.raises(PermissionError, match="FROZEN_UNOPENED"):
        require_validation_authorized(configuration)


def test_progress_only_formats_an_existing_scheduled_event():
    rendered = checkpoint_progress(
        seed=11,
        transitions=100_000,
        updates=2_250,
        completed_episodes=7,
        canonical_successes=2,
        diagnostic_rsr=4,
        maximum_replay_progress=0.75,
        canonical_hindsight_tuples=32,
        walltime_s=123.4,
        healthy=True,
    )
    assert rendered == "\n".join(
        (
            "G11 | 100k / 300k | 33.3%",
            "updates: 2,250 / 7,250",
            "completed episodes: 7",
            "canonical training successes: 2",
            "diagnostic RSR: 4/18",
            "max replay progress: 0.750000",
            "canonical hindsight tuples: 32",
            "walltime: 123.4s",
            "status: HEALTHY",
        )
    )
    assert json.loads(Path("benchmarks/gcsl_canonical/canonical.json").read_text())[
        "diagnostic_checkpoints"
    ] == [
        50_000,
        100_000,
        150_000,
        200_000,
        250_000,
        300_000,
    ]
