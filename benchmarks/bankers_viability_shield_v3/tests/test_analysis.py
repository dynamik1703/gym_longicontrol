"""Analysis logic against fabricated records, no task construction or episodes."""

import copy
import importlib

import pytest

from benchmarks.bankers_viability_shield_v3.serialization import atomic_json, file_hash


@pytest.mark.parametrize(
    "lose_coverage,expected",
    [
        (False, "BANKERS_V3_READY_FOR_MASKED_RL_DESIGN"),
        (True, "BANKERS_V3_SPEED_SAFE_PROGRESS_INSUFFICIENT"),
    ],
)
def test_preregistered_summary(tmp_path, monkeypatch, lose_coverage, expected):
    module = importlib.import_module("benchmarks.bankers_viability_shield_v3.report")
    monkeypatch.setattr(module, "HERE", tmp_path)
    monkeypatch.setattr(module, "verify_sources", lambda _: None)
    identity = {"source": "fabricated_test"}
    pairs = {}
    for seed in range(18):
        for method in module.METHODS:
            key = f"V3_ANALYSIS_UNIT_{seed}_{method}"
            failure = (
                lose_coverage and seed == 0 and method == "BANKERS_MASKED_CAUSAL60"
            )
            result = {
                "method": method,
                "task": {"numeric_seed": seed},
                "feasible": not failure,
                "metrics": {
                    "completed": True,
                    "travel_time_s": 141.0 if failure else 100.0,
                    "energy_kwh": -0.1,
                    "max_speed_violation_m_s": 0.0,
                    "speed_violation_count": 0,
                    "integrated_speed_violation_m": 0.0,
                },
                "actual_progress_m": 1000.0,
                "evidence": {"file": key},
            }
            path = tmp_path / "sealed_results" / f"{key}.json"
            atomic_json(path, result, immutable=True)
            pairs[key] = {
                "state": "COMPLETED",
                "result": result,
                "result_file_sha256": file_hash(path),
            }
    ledger = tmp_path / "pair_ledger.json"
    atomic_json(ledger, {"pairs": pairs})
    atomic_json(
        tmp_path / "development_seal.json", {"ledger_sha256": file_hash(ledger)}
    )
    atomic_json(
        tmp_path / "execution_manifest.json",
        {"identity": identity, "ordered_panel_sha256": "test"},
    )
    certificate = {
        "minimum_velocity_slack_m_s": 1.0,
        "rejection_reason": "ACCEPTED",
        "accepted": True,
    }
    row = {
        "authority": {
            "count": 21,
            "distinct_velocities": 21,
            "velocity_span_m_s": 0.5,
            "positive_acceleration_choices": 10,
            "negative_acceleration_choices": 10,
        },
        "safe_action_mask": [True] * 21,
        "executed_action": 0.5,
        "proposal_action": 1.0,
        "selected_certificate": certificate,
        "certificates": [certificate] * 21,
        "causal60_proposal_certificate": None,
        "timing": {"mask_s": 0.001, "final_s": 0.001},
        "diagnostic_sampled": False,
    }
    monkeypatch.setattr(module, "read_evidence", lambda *_: [copy.deepcopy(row)])
    module.report()
    import json

    result = json.loads((tmp_path / "results.json").read_text())
    assert result["classification"] == expected
    assert result["controllers"]["CAUSAL60"]["raw_rsr"] == 18
    assert result["matched_causal60"]["additional_masked_failures"] == (
        ["0"] if lose_coverage else []
    )
