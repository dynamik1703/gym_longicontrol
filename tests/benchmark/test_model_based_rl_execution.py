import pytest

from benchmarks.model_based_rl.checkpointing import (
    atomic_torch_save,
    atomic_torch_save_with_metrics,
    file_sha256,
    load_checkpoint,
)
from benchmarks.model_based_rl.config import load_configuration
from benchmarks.model_based_rl.execution import (
    RunLease,
    assert_paper_test_blocked,
    atomic_json,
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


def test_failed_atomic_checkpoint_does_not_replace_verified_file(
    tmp_path, monkeypatch
):
    import torch

    path = tmp_path / "checkpoint.pt"
    original = atomic_torch_save(path, checkpoint_payload())

    def fail_during_save(_payload, stream):
        stream.write(b"incomplete")
        raise OSError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_during_save)
    with pytest.raises(OSError, match="simulated"):
        atomic_torch_save_with_metrics(path, checkpoint_payload())
    assert file_sha256(path) == original
    assert not (tmp_path / ".checkpoint.pt.tmp").exists()


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


def test_zero_work_restart_requires_and_consumes_explicit_authorization(tmp_path):
    run_id = "physics-seed-11"
    with pytest.raises(RuntimeError, match="first failure"):
        with RunLease(tmp_path, run_id, resume=False):
            raise RuntimeError("first failure")
    manifest_path, manifest = load_or_create_manifest(tmp_path)
    manifest["runs"][run_id]["fresh_restart_authorizations"] = [
        {
            "authorized_at": "2026-10-09T00:00:00+00:00",
            "reason": "collector_api_incompatibility",
            "consumed_at": None,
        }
    ]
    atomic_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="second failure"):
        with RunLease(tmp_path, run_id, resume=False, restart=True):
            raise RuntimeError("second failure")
    _path, restarted = load_or_create_manifest(tmp_path)
    run = restarted["runs"][run_id]
    assert run["state"] == "INTERRUPTED"
    assert len(run["attempts"]) == 2
    assert run["attempts"][0]["error"] == "first failure"
    assert run["attempts"][1]["restart"] is True
    assert run["fresh_restart_authorizations"][0]["consumed_at"] is not None
    with pytest.raises(PermissionError, match="unused explicit authorization"):
        with RunLease(tmp_path, run_id, resume=False, restart=True):
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


def test_runner_forwards_explicit_fresh_restart(tmp_path, monkeypatch, capsys):
    observed = {}

    def fake_run_policy(**kwargs):
        observed.update(kwargs)
        return tmp_path / "checkpoint.pt"

    monkeypatch.setattr(
        "benchmarks.model_based_rl.runner.assert_freeze_ancestor", lambda: None
    )
    monkeypatch.setattr(
        "benchmarks.model_based_rl.runner.run_policy", fake_run_policy
    )
    assert runner_main(
        [
            "restart",
            "--condition",
            "learned",
            "--seed",
            "11",
            "--output-root",
            str(tmp_path),
        ]
    ) == 0
    assert observed["restart"] is True
    assert observed["condition"] == "learned"
    assert observed["training_seed"] == 11
    assert "checkpoint.pt" in capsys.readouterr().out
