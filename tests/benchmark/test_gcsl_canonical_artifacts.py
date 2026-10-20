import json
from pathlib import Path

from benchmarks.gcsl_canonical.config import load_configuration
from benchmarks.gcsl_canonical.execution import source_hashes

ROOT = Path("benchmarks/gcsl_canonical")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_readiness_is_complete_but_all_scientific_authorization_is_off():
    status = load("preparation_status.json")
    assert status["all_scientific_readiness_flags_true"] is True
    assert all(status["readiness"].values())
    assert status["authorization"] == {
        "main_training_authorized": False,
        "main_training_enabled": False,
        "validation_authorized": False,
        "paper_test_authorized": False,
    }
    assert status["preparation_interactions"]["policies_trained"] == 0
    assert status["preparation_interactions"]["policies_evaluated"] == 0


def test_source_hash_lock_matches_every_execution_file():
    status = load("preparation_status.json")
    assert status["scientific_source_hashes"] == source_hashes()
    assert status["scientific_configuration_sha256"] == (
        load_configuration().configuration_sha256
    )


def test_primary_source_pin_and_license_conclusion_are_exact():
    upstream = load("upstream.json")
    assert upstream["paper"]["audited_version"] == "v4"
    assert upstream["paper"]["openreview"].endswith("rALA0Xo6yNJ")
    assert upstream["implementation"] == {
        "commit": "cfae5609cee79e5a2228fb7653451023c41a64cb",
        "commit_date": "2020-09-17T12:56:31-07:00",
        "repository": "https://github.com/dibyaghosh/gcsl",
        "tree": "b5672e00ed73787e85c193fab79f1aa51edbd9c3",
    }
    assert upstream["license"]["conclusion"] == "NO_EXPLICIT_TOP_LEVEL_LICENSE"


def test_resource_record_obeys_preparation_and_hard_caps():
    resource = load("resource_measurements.json")
    preparation = resource["preparation_only"]
    assert preparation["main_training_performed"] is False
    assert preparation["policy_evaluation_performed"] is False
    assert (
        preparation["cumulative_simulator_transitions_across_benchmark_passes"] <= 5000
    )
    assert preparation["gcsl_diagnostic_episodes"] == 0
    assert preparation["gcsl_validation_episodes"] == 0
    assert preparation["constrained_validation_episodes"] == 0
    assert preparation["paper_content_instantiated"] is False
    assert resource["hard_gates"]["computational_feasibility_verified"] is True


def test_constrained_reference_is_contextual_verified_and_unevaluated():
    reference = load("constrained_reference_verification.json")
    assert reference["reference_constrained_verified"] is True
    assert reference["strict_algorithmic_control"] is False
    assert set(reference["seeds"]) == {"11", "29", "47"}
    assert all(
        row["real_transitions"] == 300_000
        and row["rl_gradient_updates"] == 30_000
        and len(row["sha256"]) == 64
        for row in reference["seeds"].values()
    )
    assert reference["future_evaluation"] == {
        "split": "GCSL_VALIDATION_V1",
        "state": "NOT_AUTHORIZED",
        "episodes_run_during_preparation": 0,
    }
