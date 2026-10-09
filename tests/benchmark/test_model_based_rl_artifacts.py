import json
from pathlib import Path

from benchmarks.constrained_rl_v2.config import (
    configuration_sha256 as v2_configuration_sha256,
)
from benchmarks.constrained_rl_v2.config import load_configuration as load_v2
from benchmarks.model_based_rl.execution import SCIENTIFIC_FILES, scientific_hashes

ROOT = Path("benchmarks/model_based_rl")


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_required_compact_artifacts_are_present_and_consistent():
    required = {
        "README.md",
        "SOURCE_AUDIT.md",
        "DESIGN.md",
        "PROTOCOL.md",
        "RESOURCE_REPORT.md",
        "canonical.json",
        "upstream.json",
        "physics_parity.json",
        "learned_model_probe.json",
        "model_disabled_parity.json",
        "resource_measurements.json",
        "preparation_interactions.json",
        "preparation_status.json",
    }
    assert required <= {path.name for path in ROOT.iterdir()}
    assert read_json("physics_parity.json")["verified"] is True
    assert read_json("model_disabled_parity.json")["verified"] is True
    interactions = read_json("preparation_interactions.json")
    assert interactions["simulator_transitions"] == 4995
    assert interactions["simulator_transitions"] <= interactions["maximum_allowed"]
    assert interactions["validation_transitions"] == 0
    assert interactions["paper_test_transitions"] == 0


def test_reviewed_execution_authorization_preserves_sealed_evaluation_state():
    status = read_json("preparation_status.json")
    assert status["scientific_design_frozen"] is True
    assert status["physics_model_verified"] is True
    assert status["learned_model_verified"] is True
    assert status["execution_infrastructure_ready"] is True
    assert status["ready_for_main_training"] is True
    assert status["main_training_authorized"] is True
    assert status["main_training_enabled"] is True
    assert status["validation_opened"] is False
    assert status["paper_test_opened"] is False


def test_frozen_v2_and_source_provenance():
    canonical = read_json("canonical.json")
    assert v2_configuration_sha256(load_v2()) == canonical["frozen_v2"][
        "configuration_sha256"
    ]
    upstream = read_json("upstream.json")["sources"]
    assert upstream["mbpo"]["revision"] == (
        "ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129"
    )
    assert upstream["td_mpc2"]["revision"] == (
        "e9f59321933cbc8e11a002b842adc7d4ffae8ff1"
    )
    assert upstream["dreamerv3"]["revision"] == (
        "e01491fad6434b2245a3b8ca201dd7faedcc458c"
    )
    assert upstream["fsrl"]["revision"] == (
        "e056fc9498d5d037869533da7cf976acf462f918"
    )


def test_all_scientific_sources_have_sha256_and_no_raw_runs_are_tracked():
    hashes = scientific_hashes()
    assert set(hashes) == set(SCIENTIFIC_FILES)
    assert all(len(value) == 64 for value in hashes.values())
    assert not (ROOT / "runs").exists()
    assert not tuple(ROOT.rglob("*.pt"))
    assert not tuple(ROOT.rglob("*.pth"))
