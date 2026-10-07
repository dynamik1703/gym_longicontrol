from pathlib import Path

import numpy as np
import pytest

from benchmarks.contrastive_rl.checkpointing import file_sha256
from benchmarks.contrastive_rl.config import (
    DEVELOPMENT_CHECKPOINTS,
    DEVELOPMENT_TRACKS,
    PLANNED_COMPLETE_UPDATE_CYCLES,
    PLANNED_DEPTHS,
    PLANNED_SEEDS,
    PLANNED_TRANSITIONS_PER_POLICY,
    SEALED_PAPER_TRACKS,
    VALIDATION_TRACKS,
)
from benchmarks.contrastive_rl.execution import (
    EXPECTED_CONFIGURATION_SHA256,
    SCIENTIFIC_SHA256,
    ExclusiveStudyLock,
    TrainingTrackStream,
    assert_paper_tracks_sealed,
    authorize_fresh_restart,
    initialize_study_root,
    interrupt_policy,
    new_manifest,
    permit_exact_resume,
    policy_key,
    scheduled_update_transitions,
    should_update,
    start_policy,
    validate_resume_provenance,
    validate_validation_gate,
)
from benchmarks.contrastive_rl.replay import TransitionReplay


def test_frozen_schedule_has_exact_boundaries_and_cycle_count():
    schedule = scheduled_update_transitions()
    assert len(schedule) == PLANNED_COMPLETE_UPDATE_CYCLES == 7_250
    assert schedule[0] == 10_040
    assert schedule[-1] == 300_000
    assert all(second - first == 40 for first, second in zip(schedule, schedule[1:]))
    assert not any(should_update(step) for step in (0, 9_999, 10_000, 10_039))
    assert should_update(10_040)
    assert should_update(300_000)
    assert not should_update(300_001)


def test_frozen_matrix_has_six_matched_policies():
    manifest = new_manifest({"configuration_sha256": "x" * 64})
    assert PLANNED_DEPTHS == (4, 16)
    assert PLANNED_SEEDS == (11, 29, 47)
    assert len(manifest["policies"]) == 6
    for depth in PLANNED_DEPTHS:
        for seed in PLANNED_SEEDS:
            item = manifest["policies"][policy_key(depth, seed)]
            assert item["status"] == "NOT_STARTED"
            assert item["native_transitions"] == 0
            assert item["complete_update_cycles"] == 0


def test_training_track_stream_is_reproducible_separate_and_restorable():
    first = TrainingTrackStream(11)
    second = TrainingTrackStream(11)
    values = [first.next_seed() for _ in range(32)]
    assert values == [second.next_seed() for _ in range(32)]
    forbidden = set(range(1000, 1009))
    forbidden.update(DEVELOPMENT_TRACKS)
    forbidden.update(VALIDATION_TRACKS)
    forbidden.update(SEALED_PAPER_TRACKS)
    assert forbidden.isdisjoint(values)
    state = first.state
    expected_next = first.next_seed()
    restored = TrainingTrackStream(11, state)
    assert restored.next_seed() == expected_next
    assert (
        restored.provenance["claim_identical_realized_tracks_across_policies"] is False
    )


def _append_episode(replay, episode_id, length, *, terminated=False, truncated=False):
    for step in range(length):
        final = step == length - 1
        replay.append(
            state=np.full(3, episode_id + step / 10),
            action=np.array([step / 10]),
            source_outcome=np.array([step, max(step - 1, 0), step, 0.0]),
            outcome=np.array([step + 1, step, step + 1, 0.0]),
            historical_reward=10_000.0 + step,
            terminated=bool(final and terminated),
            truncated=bool(final and truncated),
            episode_id=episode_id,
            episode_step=step,
        )


def test_replay_samples_strict_futures_without_crossing_episode_boundaries():
    replay = TransitionReplay(16, state_dim=3)
    _append_episode(replay, 0, 3, terminated=True)
    _append_episode(replay, 1, 2, truncated=True)
    sampled = replay.sample(256, gamma=0.99, rng=np.random.default_rng(7))
    assert np.all(sampled.future_indices > sampled.source_indices)
    assert np.array_equal(
        replay.episode_ids[sampled.source_indices],
        replay.episode_ids[sampled.future_indices],
    )
    assert set(sampled.source_indices).isdisjoint({2, 4})
    assert 2 in sampled.future_indices
    assert 4 in sampled.future_indices
    assert replay.terminated[sampled.future_indices].any()
    assert replay.truncated[sampled.future_indices].any()


def test_replay_round_trip_preserves_all_rows_and_sampling_rng():
    replay = TransitionReplay(8, state_dim=3)
    _append_episode(replay, 0, 4, terminated=True)
    restored = TransitionReplay.from_state(replay.to_state())
    for field in (
        "states",
        "actions",
        "source_outcomes",
        "outcomes",
        "historical_rewards",
        "terminated",
        "truncated",
        "episode_ids",
        "episode_steps",
    ):
        np.testing.assert_array_equal(
            getattr(replay, field)[: replay.size],
            getattr(restored, field)[: restored.size],
        )
    first_rng = np.random.default_rng(99)
    second_rng = np.random.default_rng(99)
    first = replay.sample(16, gamma=0.99, rng=first_rng)
    second = restored.sample(16, gamma=0.99, rng=second_rng)
    np.testing.assert_array_equal(first.source_indices, second.source_indices)
    np.testing.assert_array_equal(first.future_indices, second.future_indices)


def test_replay_rejects_post_terminal_rows_and_noncontiguous_steps():
    replay = TransitionReplay(4, state_dim=3)
    _append_episode(replay, 0, 1, terminated=True)
    with pytest.raises(ValueError, match="post-terminal"):
        replay.append(
            state=np.zeros(3),
            action=np.zeros(1),
            source_outcome=np.zeros(4),
            outcome=np.zeros(4),
            historical_reward=0.0,
            terminated=False,
            truncated=False,
            episode_id=0,
            episode_step=1,
        )


def test_exclusive_lock_and_duplicate_study_root_are_refused(tmp_path):
    root, manifest = initialize_study_root(tmp_path / "study", {"frozen": True})
    marker = root / "preserve.txt"
    marker.write_text("do not erase", encoding="utf-8")
    with pytest.raises(FileExistsError, match="cannot evade"):
        initialize_study_root(root, {"frozen": True})
    assert marker.read_text(encoding="utf-8") == "do not erase"
    with ExclusiveStudyLock(root, manifest["study_id"]):
        with pytest.raises(RuntimeError, match="ACTIVE.lock"):
            with ExclusiveStudyLock(root, manifest["study_id"]):
                pass
    assert not (root / "ACTIVE.lock").exists()


def test_policy_state_machine_refuses_duplicate_and_preserves_attempt_accounting():
    manifest = new_manifest({"frozen": True})
    item = start_policy(manifest, 4, 11)
    assert item["status"] == "RUNNING"
    with pytest.raises(RuntimeError, match="second copy"):
        start_policy(manifest, 4, 11)
    interrupt_policy(item, transition_count=123, update_cycles=4, reason="test")
    assert item["status"] == "INTERRUPTED"
    assert item["attempts"][0]["native_transitions"] == 123
    with pytest.raises(RuntimeError, match="explicit resume"):
        start_policy(manifest, 4, 11)
    with pytest.raises(ValueError, match="authorization is incomplete"):
        authorize_fresh_restart(item, {"authorized_by": "nobody"})
    permit_exact_resume(item, "a" * 64)
    assert item["status"] == "RUNNING"


def test_resume_requires_current_configuration_science_and_execution_hashes():
    provenance = {
        "configuration_sha256": "a" * 64,
        "scientific_source_sha256": {"science.py": "b" * 64},
        "execution_source_sha256": {"runner.py": "c" * 64},
    }
    manifest = new_manifest(provenance)
    validate_resume_provenance(manifest, provenance)
    changed = {**provenance, "execution_source_sha256": {"runner.py": "d" * 64}}
    with pytest.raises(RuntimeError, match="execution_source_sha256"):
        validate_resume_provenance(manifest, changed)


def _completed_manifest(tmp_path: Path):
    manifest = new_manifest(
        {
            "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
            "execution_source_sha256": {"runner.py": "e" * 64},
        }
    )
    for index, item in enumerate(manifest["policies"].values()):
        checkpoint = tmp_path / f"checkpoint-{index}.bin"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        item.update(
            {
                "status": "COMPLETED",
                "native_transitions": PLANNED_TRANSITIONS_PER_POLICY,
                "complete_update_cycles": PLANNED_COMPLETE_UPDATE_CYCLES,
                "development_checkpoints": [
                    {"transition_count": step} for step in DEVELOPMENT_CHECKPOINTS
                ],
                "final_checkpoint_path": checkpoint.name,
                "final_checkpoint_sha256": file_sha256(checkpoint),
                "configuration_sha256": EXPECTED_CONFIGURATION_SHA256,
                "scientific_source_sha256": dict(SCIENTIFIC_SHA256),
                "execution_source_sha256": {"runner.py": "e" * 64},
            }
        )
    return manifest


def test_validation_gate_blocks_early_and_requires_every_frozen_artifact(tmp_path):
    manifest = new_manifest({"frozen": True})
    with pytest.raises(RuntimeError, match="all six"):
        validate_validation_gate(tmp_path, manifest)
    complete = _completed_manifest(tmp_path)
    validate_validation_gate(tmp_path, complete)
    complete["policies"][policy_key(4, 11)]["complete_update_cycles"] -= 1
    with pytest.raises(RuntimeError, match="7,250"):
        validate_validation_gate(tmp_path, complete)


def test_validation_is_single_use_and_paper_tracks_are_always_sealed(tmp_path):
    manifest = _completed_manifest(tmp_path)
    manifest["validation_opened"] = True
    with pytest.raises(RuntimeError, match="already"):
        validate_validation_gate(tmp_path, manifest)
    with pytest.raises(PermissionError, match="4000-4017"):
        assert_paper_tracks_sealed()


def test_frozen_scientific_hashes_match_projected_goal_preparation_commit():
    root = Path(__file__).resolve().parents[2]
    for relative, expected in SCIENTIFIC_SHA256.items():
        assert file_sha256(root / relative) == expected, relative
