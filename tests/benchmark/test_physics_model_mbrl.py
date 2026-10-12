from __future__ import annotations

import json
import plistlib
from pathlib import Path

import pytest

from benchmarks.model_based_rl.checkpointing import file_sha256
from benchmarks.model_based_rl.config import load_configuration as load_parent
from benchmarks.model_based_rl.execution import scientific_hashes
from benchmarks.physics_model_mbrl.config import (
    PHYSICS_CONDITION,
    TRAINING_SEEDS,
    load_configuration,
)
from benchmarks.physics_model_mbrl.detached_launcher import (
    LABEL,
    create_launchd_job,
)
from benchmarks.physics_model_mbrl.execution import (
    assert_paper_test_blocked,
    initial_manifest,
    load_or_create_manifest,
    validation_ready,
)
from benchmarks.physics_model_mbrl.final_evaluation import _exclusive_failure_mode
from benchmarks.physics_model_mbrl.freeze import verify_study_freeze
from benchmarks.physics_model_mbrl.matrix_worker import FROZEN_PLAN, run_matrix
from benchmarks.physics_model_mbrl.training import run_policy


def test_protocol_selects_only_byte_frozen_parent_physics_condition():
    configuration = load_configuration()
    parent = load_parent()
    assert configuration.raw["condition"] == PHYSICS_CONDITION
    assert tuple(configuration.raw["training_seeds"]) == TRAINING_SEEDS
    assert configuration.raw["parent_configuration_sha256"] == (
        parent.configuration_sha256
    )
    assert parent.imagination.horizon == 1
    assert parent.imagination.warmup_real_transitions == 10_000
    assert parent.imagination.refresh_real_transitions == 250
    assert parent.imagination.synthetic_transitions_per_refresh == 2_500
    assert parent.imagination.real_batch_size == 128
    assert parent.imagination.model_batch_size == 128
    assert parent.v2.algorithm.n_step == 2


def test_historical_scientific_hashes_remain_frozen():
    frozen = json.loads(
        Path("benchmarks/model_based_rl/scientific_freeze.json").read_text(
            encoding="utf-8"
        )
    )["files"]
    assert scientific_hashes() == frozen
    assert (
        file_sha256("benchmarks/model_based_rl/physics_model.py")
        == frozen["benchmarks/model_based_rl/physics_model.py"]
    )


def test_dedicated_study_freeze_matches_sources():
    assert verify_study_freeze()


def test_manifest_contains_exactly_three_physics_runs_and_sealed_splits(tmp_path):
    manifest = initial_manifest()
    assert set(manifest["runs"]) == {f"physics-seed-{seed}" for seed in TRAINING_SEEDS}
    assert all(item["condition"] == "physics" for item in manifest["runs"].values())
    assert manifest["validation"]["state"] == "SEALED"
    assert manifest["paper_test"]["state"] == "SEALED"
    path, persisted = load_or_create_manifest(tmp_path)
    assert path.is_file()
    assert persisted["study"] == "physics-model-constrained-rl-v1"


def test_validation_requires_all_three_exactly_complete():
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


def test_paper_tracks_remain_hard_blocked():
    with pytest.raises(PermissionError, match="remain sealed"):
        assert_paper_test_blocked()


def test_physics_matrix_is_serial_and_stops_on_failure(tmp_path, monkeypatch):
    calls = []

    class Process:
        pid = 123

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))

        def wait(self):
            return 9

        def poll(self):
            return 9

    monkeypatch.setattr(
        "benchmarks.physics_model_mbrl.matrix_worker.subprocess.Popen", Process
    )
    assert (
        run_matrix(
            repository=tmp_path,
            output_root=tmp_path / "runs",
            python="/fixture/python",
        )
        == 9
    )
    assert FROZEN_PLAN == (("physics", 11), ("physics", 29), ("physics", 47))
    assert len(calls) == 1
    assert "benchmarks.physics_model_mbrl.runner" in calls[0][0]
    status = json.loads(
        (tmp_path / "runs" / "launcher-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "STOPPED_ON_WORKER_FAILURE"


def test_launchd_identity_and_runtime_are_isolated(tmp_path):
    plist, metadata = create_launchd_job(
        repository=tmp_path,
        output_root=tmp_path / "physics-runs",
        python="/fixture/python",
        pythonpath="fixture-path",
    )
    with plist.open("rb") as stream:
        payload = plistlib.load(stream)
    assert payload["Label"] == LABEL
    assert payload["KeepAlive"] is False
    assert payload["EnvironmentVariables"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "benchmarks.physics_model_mbrl.matrix_worker" in payload["ProgramArguments"]
    assert metadata["output_root"].endswith("physics-runs")


def test_training_adapter_changes_only_manifest_routing(tmp_path, monkeypatch):
    observed = {}

    def fake_parent_run(**kwargs):
        from benchmarks.model_based_rl import execution as parent_execution
        from benchmarks.model_based_rl import training as parent_training

        manifest_path, manifest = parent_execution.load_or_create_manifest(tmp_path)
        observed.update(kwargs)
        observed["manifest"] = manifest
        observed["manifest_path"] = manifest_path
        observed["checkpoint_builder"] = parent_training._make_checkpoint_payload
        return tmp_path / "checkpoint.pt"

    monkeypatch.setattr(
        "benchmarks.physics_model_mbrl.training.parent_training.run_policy",
        fake_parent_run,
    )
    result = run_policy(training_seed=11, output_root=tmp_path)
    assert result == tmp_path / "checkpoint.pt"
    assert observed["condition"] == "physics"
    assert set(observed["manifest"]["runs"]) == {
        "physics-seed-11",
        "physics-seed-29",
        "physics-seed-47",
    }
    assert observed["checkpoint_builder"].__name__ == (
        "_checkpoint_with_observational_diagnostics"
    )


def test_training_adapter_refuses_fresh_restart(tmp_path):
    with pytest.raises(PermissionError, match="Fresh restart"):
        run_policy(training_seed=11, output_root=tmp_path, restart=True)


@pytest.mark.parametrize(
    ("episode", "expected"),
    [
        (
            {
                "feasible": True,
                "completed": True,
                "travel_time_s": 100.0,
                "max_speed_violation_m_s": 0.0,
            },
            "feasible",
        ),
        (
            {
                "feasible": False,
                "completed": False,
                "travel_time_s": 180.0,
                "max_speed_violation_m_s": 1.0,
            },
            "incomplete",
        ),
        (
            {
                "feasible": False,
                "completed": True,
                "travel_time_s": 141.0,
                "max_speed_violation_m_s": 0.0,
            },
            "deadline_only",
        ),
        (
            {
                "feasible": False,
                "completed": True,
                "travel_time_s": 130.0,
                "max_speed_violation_m_s": 0.1,
            },
            "speed_only",
        ),
        (
            {
                "feasible": False,
                "completed": True,
                "travel_time_s": 141.0,
                "max_speed_violation_m_s": 0.1,
            },
            "deadline_and_speed",
        ),
    ],
)
def test_exclusive_failure_modes(episode, expected):
    assert _exclusive_failure_mode(episode, 140.0) == expected
