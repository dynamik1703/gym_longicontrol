"""Execution-only recovery administration and detached-launch tests."""

from __future__ import annotations

import json
import plistlib

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
