from __future__ import annotations

import ast
import hashlib

import pytest

from benchmarks.physical_reference_contextual import runner


def test_contextual_protocol_is_explicitly_post_hoc_and_not_independent():
    protocol = runner.PROTOCOL.read_text(encoding="utf-8")
    assert (
        "This is a post-hoc contextual physical-reference analysis on previously"
        in protocol
    )
    assert "not a new independent held-out generalization benchmark" in protocol


def test_immutable_validation_artifacts_are_complete_and_exact():
    expected = {
        "sameinfo": ("SAMEINFO_VALIDATION_V1", 9_500_001),
        "gcsl": ("GCSL_VALIDATION_V1", 9_700_000),
    }
    for name, (evaluation_id, first_seed) in expected.items():
        panel = runner._immutable_panel(name)
        assert panel["evaluation_id"] == evaluation_id
        assert len(panel["tasks"]) == 18
        assert len(panel["rl_rows"]) == 108
        assert panel["tasks"][0]["numeric_seed"] == first_seed
        assert panel["ordered_sha256"] == runner.PANELS[name]["ordered_sha256"]


def test_runtime_task_fingerprints_match_immutable_ledgers():
    for name in runner.PANELS:
        for task in runner._immutable_panel(name)["tasks"]:
            runner._verify_runtime_task(task)


def test_frozen_reference_source_and_configuration_hashes_match():
    integrity = runner._reference_integrity()
    assert integrity["verified"]
    assert (
        integrity["scientific_configuration_sha256"]
        == runner.EXPECTED_REFERENCE_CONFIGURATION_SHA256
    )
    assert integrity["source_sha256_expected"] == integrity["source_sha256_observed"]


def test_contextual_runner_imports_no_rl_actor_or_learner():
    tree = ast.parse(runner.Path(runner.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = ("agent", "learner", "policy", "checkpointing", "torch")
    assert not any(part in name.split(".") for name in imported for part in forbidden)


def test_paired_energy_is_feasible_only_and_uses_causal_minus_oracle():
    oracle = [
        {
            "track_seed": 1,
            "canonical_success": True,
            "signed_net_energy_kwh": 0.2,
            "travel_time_s": 100.0,
        },
        {
            "track_seed": 2,
            "canonical_success": False,
            "signed_net_energy_kwh": 0.1,
            "travel_time_s": 180.0,
        },
    ]
    causal = [
        {
            "track_seed": 1,
            "canonical_success": True,
            "signed_net_energy_kwh": 0.25,
            "travel_time_s": 90.0,
        },
        {
            "track_seed": 2,
            "canonical_success": True,
            "signed_net_energy_kwh": 0.3,
            "travel_time_s": 110.0,
        },
    ]
    result = runner._physical_pair_analysis(oracle, causal)
    assert result["both_references_feasible_count"] == 1
    assert result["energy_causal_minus_oracle_kwh"]["mean"] == pytest.approx(0.05)
    assert result["travel_time_causal_minus_oracle_s"]["mean"] == -10.0


def test_completed_contextual_artifacts_have_exact_accounting():
    if not runner.RESULTS.exists():
        pytest.skip("contextual execution has not been authorized/completed")
    payload = runner.load_json(runner.RESULTS)
    accounting = payload["execution_accounting"]
    assert accounting["scientific_controller_task_pairs"] == 72
    assert accounting["duplicate_scientific_pairs"] == 0
    assert accounting["oracle_replay_checks"] == 36
    assert accounting["rl_policy_episodes_executed"] == 0
    assert payload["factual_interpretation"][
        "both_references_full_feasibility_on_both_panels"
    ]


def test_completed_contextual_output_hashes_and_pairs_are_unique():
    if not runner.RESULTS.exists():
        pytest.skip("contextual execution has not been authorized/completed")
    integrity = runner.load_json(runner.INTEGRITY)
    for name, expected in integrity["output_sha256"].items():
        observed = hashlib.sha256((runner.HERE / name).read_bytes()).hexdigest()
        assert observed == expected
    paths = sorted(runner.EPISODES.rglob("seed-*-*.json"))
    identities = set()
    for path in paths:
        envelope = runner.load_json(path)
        identity = (
            envelope["evaluation_id"],
            envelope["task"]["numeric_seed"],
            envelope["controller"],
        )
        assert identity not in identities
        identities.add(identity)
    assert len(paths) == len(identities) == 72
