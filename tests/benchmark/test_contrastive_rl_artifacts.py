import json
from pathlib import Path

from benchmarks.contrastive_rl.config import (
    PLANNED_COMPLETE_UPDATE_CYCLES,
    ReferenceCoreConfig,
)

ROOT = Path("benchmarks/contrastive_rl")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_preparation_deliverables_are_complete_without_fake_results():
    expected = {
        "README.md",
        "SOURCE_AUDIT.md",
        "DESIGN.md",
        "PROTOCOL.md",
        "RESOURCE_REPORT.md",
        "EXECUTION.md",
        "execution_schema.json",
        "preparation_status.json",
        "canonical.json",
        "upstream.json",
        "projected_adapter.py",
        "runner.py",
        "replay.py",
        "checkpointing.py",
        "diagnostics.py",
        "evaluation.py",
        "execution.py",
        "projection_measurements.json",
        "resource_measurements_projected.json",
    }
    assert expected <= {path.name for path in ROOT.iterdir()}
    assert not (ROOT / "RESULTS.md").exists()


def test_semantic_readiness_is_separate_from_execution_authorization():
    status = load("preparation_status.json")
    config = load("canonical.json")
    assert status["reference_core_verified"] is True
    assert status["task_mapping_verified"] is True
    assert status["technical_semantic_readiness"] is True
    assert status["execution_infrastructure_ready"] is True
    assert status["ready_for_main_training"] is True
    assert status["main_training_authorized"] is False
    assert status["main_training_enabled"] is False
    assert (
        status["future_protocol_status"]
        == "FROZEN_EXECUTION_READY_BUT_NOT_AUTHORIZED"
    )
    assert config["status"] == "FROZEN_PRETRAINING_DESIGN_EXECUTION_DISABLED"
    assert config["task_mapping_verified"] is True
    assert config["main_training_authorized"] is False
    assert config["main_training_enabled"] is False


def test_pinned_sources_and_planned_factorial_budget():
    upstream = load("upstream.json")
    config = load("canonical.json")
    assert upstream["paper"]["audited_version"] == "v4"
    assert (
        upstream["implementation"]["commit"]
        == "17acb519ddc4325c8662b1f8c68ed6a5f31857fc"
    )
    future = config["planned_protocol"]
    assert future["seeds"] == [11, 29, 47]
    assert config["reference_core"]["depths_inside_residual_blocks"] == [4, 16]
    assert config["reference_core"]["batch_size"] == 256
    assert config["representations"]["projected_goal_dim"] == 3
    assert config["representations"]["canonical_command"] == [1.0, 1.0, 1.0]
    assert future["native_transitions_per_policy"] == 300_000
    assert future["prefill_transitions"] == 10_000
    assert future["first_update_after_transition"] == 10_040
    assert future["final_update_after_transition"] == 300_000
    assert future["complete_update_cycles_per_policy"] == 7_250
    assert PLANNED_COMPLETE_UPDATE_CYCLES == 7_250
    assert ReferenceCoreConfig().goal_dim == 3
    assert future["total_native_transitions"] == 1_800_000
    assert future["paper_tracks_sealed"] == list(range(4000, 4018))


def test_execution_schema_is_ready_but_cannot_authorize_training():
    schema = load("execution_schema.json")
    assert schema["frozen_configuration_sha256"] == (
        "659e139034d9f3aed25a07d5244bc6ac4c86ce63dd0394f315ca5341ae0a78fc"
    )
    assert schema["matrix"] == {
        "depths": [4, 16],
        "training_seeds": [11, 29, 47],
        "policy_count": 6,
    }
    assert schema["schedule"]["complete_update_cycles"] == 7_250
    assert schema["authorization"]["execution_infrastructure_ready"] is True
    assert schema["authorization"]["ready_for_main_training"] is True
    assert schema["authorization"]["main_training_authorized"] is False
    assert schema["authorization"]["main_training_enabled"] is False
    assert schema["validation"]["opened"] is False
    assert schema["paper_final"]["opened"] is False


def test_resource_probe_stayed_within_preparation_caps():
    measurements = load("resource_measurements.json")
    projected = load("resource_measurements_projected.json")
    status = load("preparation_status.json")
    assert measurements["training_performed"] is False
    assert measurements["validation_tracks_used"] is False
    assert measurements["paper_tracks_used"] is False
    assert measurements["simulator"]["transitions"] <= 2_000
    assert measurements["simulator"]["used_for_learning"] is False
    for depth in measurements["depths"]:
        assert depth["synthetic_update_iterations_total_including_compile"] <= 100
        assert depth["finite_final_metrics"] is True
    assert projected["simulator"]["transitions"] == 0
    for depth in projected["depths"]:
        assert depth["synthetic_update_iterations_total_including_compile"] <= 20
        assert depth["finite_final_metrics"] is True
    assert [item["parameters"]["total_trainable"] for item in projected["depths"]] == [
        839_299,
        3_226_243,
    ]
    assert status["preparation_usage"]["longicontrol_policy_training_transitions"] == 0
    assert (
        status["preparation_usage"]["total_longicontrol_simulator_transitions"]
        == 1_503
    )
    assert (
        status["preparation_usage"][
            "execution_infrastructure_additional_simulator_transitions"
        ]
        == 3
    )
    assert status["preparation_usage"]["validation_tracks_used"] is False
    assert status["preparation_usage"]["paper_tracks_used"] is False


def test_projection_collision_check_is_bounded_and_not_training():
    measurement = load("projection_measurements.json")
    assert measurement["native_simulator_transitions"] <= 500
    assert measurement["used_for_learning"] is False
    assert measurement["training_performed"] is False
    assert measurement["validation_tracks_used"] is False
    assert measurement["paper_tracks_used"] is False
    assert (
        measurement["projected_equivalent_pair_rate"]
        > measurement["raw_equivalent_pair_rate"]
    )
