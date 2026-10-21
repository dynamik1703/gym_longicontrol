from __future__ import annotations

from benchmarks.hybrid_rl_validation_v2.constants import (
    EXPECTED_ACTOR_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
)
from benchmarks.hybrid_rl_validation_v2.integrity import (
    load_frozen_policies,
    verify_frozen_scientific_sources,
    verify_incident_evidence,
    verify_test_isolation,
)


def test_incident_counts_are_permanent_and_exact():
    evidence = verify_incident_evidence()
    assert evidence["passed"]
    assert all(evidence["checks"].values())


def test_scientific_sources_and_all_original_controllers_are_unchanged():
    sources = verify_frozen_scientific_sources()
    assert sources["passed"]
    assert sources["original_canonical_unchanged"]
    assert all(sources["controller_source_unchanged"].values())


def test_checkpoint_and_actor_hashes_load_read_only():
    policies, agents = load_frozen_policies(return_agents=True)
    assert set(policies) == {"H11", "H29", "H47"}
    for controller_id, policy in policies.items():
        assert policy["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256[controller_id]
        assert policy["actor_parameter_sha256"] == EXPECTED_ACTOR_SHA256[controller_id]
        assert policy["loaded_read_only"]
        assert agents[controller_id].update_count == 30_000


def test_preparation_tests_contain_no_reserved_runtime_path():
    isolation = verify_test_isolation()
    assert isolation["passed"]
    assert not isolation["violations"]
