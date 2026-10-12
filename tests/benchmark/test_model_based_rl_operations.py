"""Execution-only recovery administration and detached-launch tests."""

from __future__ import annotations

import json
import plistlib

from benchmarks.model_based_rl.checkpointing import file_sha256
from benchmarks.model_based_rl.detached_launcher import LABEL, create_launchd_job
from benchmarks.model_based_rl.execution import (
    RunLease,
    atomic_json,
    load_or_create_manifest,
)
from benchmarks.model_based_rl.matrix_worker import FROZEN_PLAN, run_matrix
from benchmarks.model_based_rl.recovery_admin import (
    ATTEMPT_TWO_ID,
    ATTEMPT_TWO_PID,
    preserve_attempt_two_and_authorize_fresh_restart,
)
from benchmarks.model_based_rl.termination_admin import (
    ATTEMPT_ID,
    CLASSIFICATION,
    close_resource_infeasible_study,
)


def test_verified_resume_reuses_same_attempt(tmp_path):
    run_id = "learned-seed-11"
    manifest_path, manifest = load_or_create_manifest(tmp_path)
    attempt = {
        "attempt_number": 1,
        "attempt_id": "same-attempt",
        "started_at": "2026-10-10T00:00:00+00:00",
        "state": "INTERRUPTED",
        "recovery_class": "VERIFIED_CHECKPOINT_RESUMABLE",
    }
    run = manifest["runs"][run_id]
    run["state"] = "INTERRUPTED"
    run["attempts"] = [attempt]
    atomic_json(manifest_path, manifest)

    with RunLease(tmp_path, run_id, resume=True) as lease:
        assert lease.attempt_id == "same-attempt"
        assert len(lease.manifest["runs"][run_id]["attempts"]) == 1
        assert len(lease.attempt["resume_events"]) == 1
        lease.manifest["runs"][run_id]["state"] = "INTERRUPTED"

    _path, persisted = load_or_create_manifest(tmp_path)
    assert len(persisted["runs"][run_id]["attempts"]) == 1


def test_attempt_two_recovery_preserves_unknown_work_and_stale_lock(
    tmp_path, monkeypatch
):
    manifest_path, manifest = load_or_create_manifest(tmp_path)
    run = manifest["runs"]["learned-seed-11"]
    run["state"] = "RUNNING"
    run["attempts"] = [
        {
            "attempt_number": 2,
            "attempt_id": ATTEMPT_TWO_ID,
            "started_at": "2026-10-09T20:30:23.198144+00:00",
            "state": "RUNNING",
        }
    ]
    atomic_json(manifest_path, manifest)
    lock_path = tmp_path / "learned-seed-11" / ".run.lock"
    atomic_json(lock_path, {"host": "fixture", "pid": ATTEMPT_TWO_PID})
    monkeypatch.setattr(
        "benchmarks.model_based_rl.recovery_admin.process_alive", lambda _pid: False
    )
    monkeypatch.setattr(
        "benchmarks.model_based_rl.recovery_admin._other_runner_pids", lambda: []
    )

    snapshot = preserve_attempt_two_and_authorize_fresh_restart(tmp_path)
    assert snapshot.is_file()
    assert not lock_path.exists()
    _path, updated = load_or_create_manifest(tmp_path)
    run = updated["runs"]["learned-seed-11"]
    failed = run["attempts"][0]
    assert failed["classification"] == "HARD_FAILURE_UNRECOVERABLE"
    assert failed["durable_scientific_work"]["real_transitions"] == 0
    assert failed["actual_consumed_scientific_work"]["real_transitions"] is None
    assert run["actual_consumed_transition_count"] is None
    assert run["fresh_restart_authorizations"][0]["consumed_at"] is None


def test_launchd_job_is_one_shot_and_persists_process_output(tmp_path):
    plist_path, metadata = create_launchd_job(
        repository=tmp_path,
        output_root=tmp_path / "runs",
        python="/fixture/python",
        pythonpath="fixture-path",
    )
    with plist_path.open("rb") as stream:
        job = plistlib.load(stream)
    assert job["Label"] == LABEL
    assert job["RunAtLoad"] is True
    assert job["KeepAlive"] is False
    assert job["AbandonProcessGroup"] is False
    assert job["StandardOutPath"].endswith("launcher.stdout.log")
    assert job["StandardErrorPath"].endswith("launcher.stderr.log")
    assert "benchmarks.model_based_rl.matrix_worker" in job["ProgramArguments"]
    assert metadata["state"] == "PREPARED"


def test_serial_matrix_stops_after_first_worker_failure(tmp_path, monkeypatch):
    calls = []

    class Process:
        pid = 321

        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))

        def wait(self):
            return 17

        def poll(self):
            return 17

    monkeypatch.setattr(
        "benchmarks.model_based_rl.matrix_worker.subprocess.Popen", Process
    )
    result = run_matrix(
        repository=tmp_path,
        output_root=tmp_path / "runs",
        python="/fixture/python",
    )
    assert result == 17
    assert len(calls) == 1
    status = json.loads(
        (tmp_path / "runs" / "launcher-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "STOPPED_ON_WORKER_FAILURE"
    assert status["failure"]["exit_code"] == 17
    assert status["plan"] == [list(item) for item in FROZEN_PLAN]
    assert status["completed"] == []


def test_serial_matrix_records_signal_failure(tmp_path, monkeypatch):
    class Process:
        pid = 322

        def __init__(self, _command, **_kwargs):
            pass

        def wait(self):
            return -9

        def poll(self):
            return -9

    monkeypatch.setattr(
        "benchmarks.model_based_rl.matrix_worker.subprocess.Popen", Process
    )
    assert run_matrix(
        repository=tmp_path,
        output_root=tmp_path / "runs",
        python="/fixture/python",
    ) == 137
    status = json.loads(
        (tmp_path / "runs" / "launcher-status.json").read_text(encoding="utf-8")
    )
    assert status["failure"]["exit_code"] is None
    assert status["failure"]["terminating_signal"] == 9


def test_resource_termination_preserves_durable_and_observed_work(
    tmp_path, monkeypatch
):
    manifest_path, manifest = load_or_create_manifest(tmp_path)
    run = manifest["runs"]["learned-seed-11"]
    run["state"] = "INTERRUPTED"
    run["attempts"] = [
        {
            "attempt_number": 3,
            "attempt_id": ATTEMPT_ID,
            "state": "INTERRUPTED",
            "started_at": "2026-10-10T11:27:29+00:00",
        }
    ]
    atomic_json(manifest_path, manifest)

    run_root = tmp_path / "learned-seed-11"
    checkpoint = run_root / "recovery" / "step-060000" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified fixture")
    checkpoint_hash = file_sha256(checkpoint)
    atomic_json(
        run_root / "recovery" / "latest.json",
        {"sha256": checkpoint_hash, "transition_count": 60_000},
    )
    atomic_json(
        run_root / "heartbeat.json",
        {
            "attempt_id": ATTEMPT_ID,
            "real_transition_count": 60_500,
            "rl_update_count": 6_050,
            "model_update_count": 29_700_838,
            "model_refresh_count": 203,
            "synthetic_generated_count": 507_500,
            "synthetic_sampled_count": 646_528,
        },
    )
    atomic_json(
        tmp_path / "launcher-status.json",
        {
            "state": "STOPPED_ON_WORKER_FAILURE",
            "active": None,
            "failure": {"terminating_signal": 2},
        },
    )
    evidence = {
        "classification": CLASSIFICATION,
        "decision_at_utc": "2026-10-12T15:52:35Z",
        "safe_boundary_at_utc": "2026-10-12T16:16:21Z",
        "execution": {"stopped_pids": [20438, 20439]},
        "durable_state": {
            "real_transitions": 60_000,
            "checkpoint_relative_path": (
                "learned-seed-11/recovery/step-060000/checkpoint.pt"
            ),
            "checkpoint_sha256": checkpoint_hash,
            "counters": {"real_transitions": 60_000},
        },
        "observed_safe_boundary": {
            "real_transitions": 60_500,
            "rl_gradient_updates": 6_050,
            "model_gradient_updates": 29_700_838,
            "model_refreshes": 203,
            "synthetic_generated": 507_500,
            "synthetic_sampled": 646_528,
        },
        "additional_consumed_after_checkpoint": {"real_transitions": 500},
        "resource_consumption": {"attempt_3_wall_clock_hours": 52.8},
    }
    evidence_path = tmp_path / "evidence.json"
    atomic_json(evidence_path, evidence)
    monkeypatch.setattr(
        "benchmarks.model_based_rl.termination_admin.process_alive",
        lambda _pid: False,
    )

    close_resource_infeasible_study(
        output_root=tmp_path, evidence_path=evidence_path
    )

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    closed = updated["runs"]["learned-seed-11"]
    assert updated["state"] == CLASSIFICATION
    assert closed["state"] == CLASSIFICATION
    assert closed["durable_transition_count"] == 60_000
    assert closed["actual_consumed_transition_count"] == 60_500
    assert closed["attempts"][-1]["classification"] == CLASSIFICATION
    assert all(
        updated["runs"][run_id]["state"] == "NOT_STARTED"
        for run_id in (
            "learned-seed-29",
            "learned-seed-47",
            "physics-seed-11",
            "physics-seed-29",
            "physics-seed-47",
        )
    )
    assert updated["validation"]["state"] == "SEALED"
    assert updated["paper_test"]["state"] == "SEALED"
    assert not (run_root / ".run.lock").exists()
