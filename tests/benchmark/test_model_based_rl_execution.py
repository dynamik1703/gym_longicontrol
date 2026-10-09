import pytest

from benchmarks.model_based_rl.checkpointing import (
    atomic_torch_save,
    file_sha256,
    load_checkpoint,
)
from benchmarks.model_based_rl.config import load_configuration
from benchmarks.model_based_rl.execution import (
    RunLease,
    assert_paper_test_blocked,
    load_or_create_manifest,
)
from benchmarks.model_based_rl.runner import main as runner_main
from benchmarks.model_based_rl.training import run_policy


def checkpoint_payload():
    return {
        "model_condition": "physics",
        "policy": {},
        "policy_optimizers": {},
        "lagrangian_states": [],
        "real_rl_replay": [],
        "real_model_replay": [],
        "synthetic_replay": [],
        "environment_state": {},
        "track_stream_state": {},
        "counters": {},
        "diagnostics": [],
        "hashes": {},
        "rng_states": {},
    }


def test_atomic_checkpoint_hash_and_corruption_detection(tmp_path):
    path = tmp_path / "checkpoint.pt"
    digest = atomic_torch_save(path, checkpoint_payload())
    assert digest == file_sha256(path)
    assert load_checkpoint(path, expected_sha256=digest)["model_condition"] == "physics"
    with pytest.raises(RuntimeError, match="mismatch"):
        load_checkpoint(path, expected_sha256="0" * 64)


def test_duplicate_run_protection_and_interruption_record(tmp_path):
    run_id = "physics-seed-11"
    with pytest.raises(RuntimeError, match="boom"):
        with RunLease(tmp_path, run_id, resume=False):
            raise RuntimeError("boom")
    _path, manifest = load_or_create_manifest(tmp_path)
    assert manifest["runs"][run_id]["state"] == "INTERRUPTED"
    assert len(manifest["runs"][run_id]["attempts"]) == 1
    with pytest.raises(RuntimeError, match="Cannot start"):
        with RunLease(tmp_path, run_id, resume=False):
            pass


def test_only_frozen_main_matrix_is_authorized_and_paper_test_is_blocked(tmp_path):
    with pytest.raises(ValueError, match="Unauthorized training seed"):
        run_policy(
            condition="physics", training_seed=13, output_root=tmp_path
        )
    with pytest.raises(PermissionError, match="sealed"):
        assert_paper_test_blocked()
    configuration = load_configuration()
    assert configuration.raw["authorization"] == {
        "main_training_authorized": True,
        "main_training_enabled": True,
        "validation_authorized": True,
        "paper_test_authorized": False,
    }


def test_status_is_read_only(tmp_path, capsys):
    assert runner_main(["status", "--output-root", str(tmp_path)]) == 0
    assert "NOT_STARTED" in capsys.readouterr().out
    assert not (tmp_path / "manifest.json").exists()
