import json
from pathlib import Path

from benchmarks.model_based_rl.config import (
    MODEL_CONDITIONS,
    RESERVED_TRACKS,
    load_configuration,
)
from benchmarks.model_based_rl.execution import (
    TrainingTrackStream,
    initial_manifest,
    validation_ready,
)


def test_frozen_matrix_and_v2_parity():
    configuration = load_configuration()
    assert tuple(configuration.raw["conditions"]) == MODEL_CONDITIONS
    assert configuration.raw["training_seeds"] == [11, 29, 47]
    assert configuration.raw["real_transition_checkpoints"][-1] == 300_000
    assert configuration.imagination.horizon == 1
    assert configuration.imagination.real_batch_size == 128
    assert configuration.imagination.model_batch_size == 128
    assert configuration.v2.algorithm.n_step == 2
    assert configuration.v2.algorithm.update_per_step == 0.1
    assert configuration.raw["authorization"] == {
        "main_training_authorized": False,
        "main_training_enabled": False,
        "validation_authorized": False,
        "paper_test_authorized": False,
    }


def test_configuration_hash_matches_machine_readable_payload():
    configuration = load_configuration()
    raw = json.loads(
        Path("benchmarks/model_based_rl/canonical.json").read_text(encoding="utf-8")
    )
    assert configuration.raw == raw
    assert len(configuration.configuration_sha256) == 64


def test_training_track_stream_is_reproducible_and_excludes_splits():
    left = TrainingTrackStream(11)
    right = TrainingTrackStream(11)
    left_values = [left.next_seed() for _ in range(1000)]
    right_values = [right.next_seed() for _ in range(1000)]
    assert left_values == right_values
    assert not (set(left_values) & RESERVED_TRACKS)


def test_validation_needs_all_six_exactly_completed():
    manifest = initial_manifest()
    assert not validation_ready(manifest)
    for run in manifest["runs"].values():
        run.update(
            {
                "state": "COMPLETED",
                "real_transitions": 300_000,
                "rl_gradient_updates": 30_000,
                "final_checkpoint_sha256": "a" * 64,
                "development_complete": True,
            }
        )
    assert validation_ready(manifest)
    next(iter(manifest["runs"].values()))["real_transitions"] -= 1
    assert not validation_ready(manifest)
