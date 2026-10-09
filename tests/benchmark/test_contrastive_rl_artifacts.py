import json
from pathlib import Path

from benchmarks.contrastive_rl.config import (
    PLANNED_COMPLETE_UPDATE_CYCLES,
    ReferenceCoreConfig,
)

ROOT = Path("benchmarks/contrastive_rl")


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_study_deliverables_include_compact_final_results():
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
        "RESULTS.md",
        "results.json",
        "validation_episodes.json",
        "execution_manifest.json",
        "finalize.py",
    }
    assert expected <= {path.name for path in ROOT.iterdir()}
    assert {
        "development-rsr.png",
        "validation-requirements.png",
        "collision-vs-score-gap.png",
    } <= {path.name for path in (ROOT / "plots").iterdir()}


def test_completed_study_disables_duplicate_training_but_preserves_authorization():
    status = load("preparation_status.json")
    config = load("canonical.json")
    assert status["reference_core_verified"] is True
    assert status["task_mapping_verified"] is True
    assert status["technical_semantic_readiness"] is True
    assert status["execution_infrastructure_ready"] is True
    assert status["status"] == "STUDY_COMPLETED"
    assert status["ready_for_main_training"] is False
    assert status["main_training_authorized"] is True
    assert status["main_training_enabled"] is False
    assert status["future_protocol_status"] == "COMPLETED_VALIDATION_FROZEN"
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


def test_execution_schema_records_completed_single_validation():
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
    assert schema["authorization"]["ready_for_main_training"] is False
    assert schema["authorization"]["main_training_authorized"] is True
    assert schema["authorization"]["main_training_enabled"] is False
    assert schema["validation"]["opened"] is True
    assert schema["validation"]["completed"] is True
    assert schema["validation"]["episode_count"] == 54
    assert schema["paper_final"]["opened"] is False
    assert schema["final_result"]["depth_4_successes"] == 0
    assert schema["final_result"]["depth_16_successes"] == 1


def test_final_results_match_frozen_budget_and_validation_accounting():
    results = load("results.json")
    manifest = load("execution_manifest.json")
    episodes = load("validation_episodes.json")

    assert results["status"] == "COMPLETED"
    assert results["canonical_configuration_sha256"] == (
        "659e139034d9f3aed25a07d5244bc6ac4c86ce63dd0394f315ca5341ae0a78fc"
    )
    assert results["validation_episode_count"] == len(episodes) == 54
    assert results["paper_tracks_used"] is False
    assert results["accounting"] == {
        "complete_update_cycles": 43_500,
        "development_simulator_transitions": 462_172,
        "native_simulator_transitions": 1_800_000,
        "validation_simulator_transitions": 59_633,
    }
    assert len(results["policies"]) == 6
    assert all(
        policy["native_transitions"] == 300_000 for policy in results["policies"]
    )
    assert all(
        policy["complete_update_cycles"] == 7_250 for policy in results["policies"]
    )
    assert all(policy["attempt_count"] == 1 for policy in results["policies"])
    assert all(policy["interruption_count"] == 0 for policy in results["policies"])

    assert manifest["status"] == "COMPLETED"
    assert manifest["validation_opened"] is True
    assert manifest["validation"]["status"] == "COMPLETED"
    assert manifest["validation"]["episode_count"] == 54
    assert manifest["paper_tracks_opened"] is False


def test_final_validation_outcomes_and_energy_semantics_are_exact():
    results = load("results.json")
    depth_4 = results["depth_results"]["4"]
    depth_16 = results["depth_results"]["16"]

    assert depth_4["success_count"] == 0
    assert depth_4["episode_count"] == 27
    assert depth_4["per_seed_success"] == {"11": 0, "29": 0, "47": 0}
    assert depth_4["mean_feasible_energy_kwh"] is None
    assert depth_4["feasible_energy_count"] == 0

    assert depth_16["success_count"] == 1
    assert depth_16["episode_count"] == 27
    assert depth_16["per_seed_success"] == {"11": 0, "29": 1, "47": 0}
    assert depth_16["feasible_energy_count"] == 1
    assert depth_16["mean_feasible_energy_kwh"] == 0.13554744407086844

    expected_history = {
        "Scalar SB3 SAC": 5,
        "Action-repeat scalar": 9,
        "Constrained V2": 21,
        "Requirement-conditioned, canonical 140 s": 12,
        "Binary Success": 0,
        "LLM reward": 1,
        "Goal-conditioned SAC, no HER": 0,
        "Goal-conditioned SAC + HER": 0,
    }
    assert {
        name: row["successes"] for name, row in results["historical_context"].items()
    } == expected_history


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
        status["preparation_usage"]["total_longicontrol_simulator_transitions"] == 1_503
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
