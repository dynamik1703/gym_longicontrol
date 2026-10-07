import json
from pathlib import Path

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
        "preparation_status.json",
        "canonical.json",
        "upstream.json",
    }
    assert expected <= {path.name for path in ROOT.iterdir()}
    assert not (ROOT / "RESULTS.md").exists()


def test_main_training_is_explicitly_disabled_and_protocol_is_draft():
    status = load("preparation_status.json")
    config = load("canonical.json")
    assert status["reference_core_verified"] is True
    assert status["task_mapping_verified"] is False
    assert status["ready_for_main_training"] is False
    assert status["main_training_enabled"] is False
    assert status["future_protocol_status"] == "DRAFT"
    assert config["status"] == "DRAFT_NOT_EXECUTABLE"
    assert config["main_training_enabled"] is False


def test_pinned_sources_and_planned_factorial_budget():
    upstream = load("upstream.json")
    config = load("canonical.json")
    assert upstream["paper"]["audited_version"] == "v4"
    assert (
        upstream["implementation"]["commit"]
        == "17acb519ddc4325c8662b1f8c68ed6a5f31857fc"
    )
    future = config["future_protocol_if_unblocked"]
    assert future["seeds"] == [11, 29, 47]
    assert config["reference_core"]["depths_inside_residual_blocks"] == [4, 16]
    assert future["native_transitions_per_policy"] == 300_000
    assert future["total_native_transitions"] == 1_800_000
    assert future["paper_tracks_sealed"] == list(range(4000, 4018))


def test_resource_probe_stayed_within_preparation_caps():
    measurements = load("resource_measurements.json")
    status = load("preparation_status.json")
    assert measurements["training_performed"] is False
    assert measurements["validation_tracks_used"] is False
    assert measurements["paper_tracks_used"] is False
    assert measurements["simulator"]["transitions"] <= 2_000
    assert measurements["simulator"]["used_for_learning"] is False
    for depth in measurements["depths"]:
        assert depth["synthetic_update_iterations_total_including_compile"] <= 100
        assert depth["finite_final_metrics"] is True
    assert status["preparation_usage"]["longicontrol_policy_training_transitions"] == 0
