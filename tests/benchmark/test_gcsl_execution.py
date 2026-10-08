import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks.gcsl.config import (  # noqa: E402
    DEVELOPMENT_CHECKPOINTS,
    DEVELOPMENT_TRACKS,
    PLANNED_SEEDS,
    PLANNED_UPDATE_CYCLES,
    SEALED_PAPER_TRACKS,
    VALIDATION_TRACKS,
    scheduled_update_transitions,
    should_update,
)
from benchmarks.gcsl.evaluation import evaluate_policy  # noqa: E402
from benchmarks.gcsl.execution import (  # noqa: E402
    ExclusiveStudyLock,
    TrainingTrackStream,
    assert_paper_tracks_sealed,
    authorize_fresh_restart,
    file_sha256,
    new_manifest,
    start_policy,
    validate_validation_gate,
)
from benchmarks.gcsl.learner import GCSLLearner  # noqa: E402

ROOT = Path("benchmarks/gcsl")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class OneStepEnvironment:
    def reset(self, seed=None):
        self.seed = seed
        return np.zeros(8), {
            "position_m": 0.0,
            "elapsed_time_s": 0.0,
            "max_speed_violation_m_s": 0.0,
        }

    def step(self, action):
        return (
            np.zeros(8),
            999.0,
            True,
            False,
            {
                "position_m": 1000.0,
                "elapsed_time_s": 100.0,
                "max_speed_violation_m_s": 0.0,
                "episode_metrics": {
                    "completed": True,
                    "travel_time_s": 100.0,
                    "energy_kwh": 1.0,
                    "speed_violation_count": 0,
                    "max_speed_violation_m_s": 0.0,
                    "integrated_speed_violation_m": 0.0,
                },
            },
        )

    def close(self):
        pass


def test_deliverables_and_authorization_flags_are_complete_and_off():
    expected = {
        "README.md",
        "SOURCE_AUDIT.md",
        "DESIGN.md",
        "PROTOCOL.md",
        "RESOURCE_REPORT.md",
        "canonical.json",
        "upstream.json",
        "preparation_status.json",
        "policy.py",
        "replay.py",
        "learner.py",
        "diagnostics.py",
        "evaluation.py",
        "checkpointing.py",
        "execution.py",
        "runner.py",
    }
    assert expected <= {path.name for path in ROOT.iterdir()}
    status = load("preparation_status.json")
    canonical = load("canonical.json")
    assert status["execution_infrastructure_ready"] is True
    assert status["ready_for_main_training"] is True
    assert status["main_training_authorized"] is False
    assert status["main_training_enabled"] is False
    assert canonical["authorization"] == {
        "execution_infrastructure_ready": True,
        "main_training_authorized": False,
        "main_training_enabled": False,
        "ready_for_main_training": True,
    }


def test_upstream_pin_source_semantics_and_license_conclusion():
    upstream = load("upstream.json")
    assert upstream["paper"]["audited_version"] == "v4"
    assert upstream["paper"]["openreview"].endswith("rALA0Xo6yNJ")
    assert upstream["implementation"]["commit"] == (
        "cfae5609cee79e5a2228fb7653451023c41a64cb"
    )
    assert upstream["license"]["conclusion"] == "NO_EXPLICIT_TOP_LEVEL_LICENSE"


def test_exact_update_schedule_and_budget():
    updates = scheduled_update_transitions()
    assert len(updates) == PLANNED_UPDATE_CYCLES == 7250
    assert updates[0] == 10_040
    assert updates[-1] == 300_000
    assert should_update(10_040)
    assert not should_update(10_000)
    assert not should_update(10_041)


def test_training_track_stream_excludes_every_research_split():
    stream = TrainingTrackStream(11)
    generated = [stream.next_seed() for _ in range(2000)]
    reserved = set(
        (
            *range(1000, 1009),
            *DEVELOPMENT_TRACKS,
            *VALIDATION_TRACKS,
            *SEALED_PAPER_TRACKS,
        )
    )
    assert not reserved.intersection(generated)


def test_study_lock_and_duplicate_policy_start_protection(tmp_path):
    manifest = new_manifest({})
    start_policy(manifest, 11)
    with pytest.raises(RuntimeError, match="RUNNING"):
        start_policy(manifest, 11)
    tmp_path.mkdir(exist_ok=True)
    with ExclusiveStudyLock(tmp_path):
        with pytest.raises(RuntimeError, match="ACTIVE.lock"):
            with ExclusiveStudyLock(tmp_path):
                pass


def test_fresh_restart_requires_explicit_complete_authorization():
    item = {"status": "INTERRUPTED", "attempts": []}
    with pytest.raises(RuntimeError, match="authorization"):
        authorize_fresh_restart(item, {"authorized_by": "user"})
    authorize_fresh_restart(
        item,
        {
            "authorized_by": "user",
            "authorized_at": "2026-10-08T00:00:00Z",
            "reason": "hardware loss",
            "prior_attempt_hash": "a" * 64,
        },
    )
    assert item["status"] == "NOT_STARTED"


def test_validation_and_paper_gates(tmp_path):
    manifest = new_manifest({})
    with pytest.raises(RuntimeError, match="all policies"):
        validate_validation_gate(tmp_path, manifest)
    checkpoint = tmp_path / "final.ckpt"
    checkpoint.write_bytes(b"frozen")
    for seed in PLANNED_SEEDS:
        manifest["policies"][f"seed-{seed}"].update(
            {
                "status": "COMPLETED",
                "native_transitions": 300_000,
                "complete_update_cycles": 7_250,
                "development_checkpoints": [
                    {"transition_count": step} for step in DEVELOPMENT_CHECKPOINTS
                ],
                "final_checkpoint_path": "final.ckpt",
                "final_checkpoint_sha256": file_sha256(checkpoint),
            }
        )
    validate_validation_gate(tmp_path, manifest)
    with pytest.raises(PermissionError, match="sealed"):
        assert_paper_tracks_sealed()


def test_deterministic_development_evaluation_does_not_change_policy():
    learner = GCSLLearner(seed=8)
    before = {
        key: value.detach().clone()
        for key, value in learner.policy.state_dict().items()
    }
    result = evaluate_policy(
        learner,
        DEVELOPMENT_TRACKS,
        split="development",
        environment_factory=OneStepEnvironment,
    )
    assert result["summary"]["canonical_successes"] == 9
    for key, value in learner.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    with pytest.raises(PermissionError, match="paper-final"):
        evaluate_policy(
            learner,
            SEALED_PAPER_TRACKS,
            split="paper-final",
            environment_factory=OneStepEnvironment,
        )
    with pytest.raises(PermissionError, match="declared validation"):
        evaluate_policy(
            learner,
            DEVELOPMENT_TRACKS,
            split="validation",
            environment_factory=OneStepEnvironment,
        )


def test_resource_probe_obeyed_preparation_caps():
    measurement = load("resource_measurements.json")
    assert measurement["training_performed"] is False
    assert measurement["synthetic_updates"] <= 100
    assert measurement["simulator"]["transitions"] <= 1000
    assert measurement["simulator"]["used_for_learning"] is False
    assert measurement["validation_tracks_used"] is False
    assert measurement["paper_tracks_used"] is False
