import pytest

from benchmarks.paper_synthesis_v1.evidence import (
    assert_headlines,
    evidence_rows,
    generated_outputs,
    method_rows,
    paired_rows,
)


def test_mandatory_historical_results_reconcile():
    assert_headlines()


def test_method_registry_does_not_conflate_checkpoints_with_seeds():
    rows = method_rows()
    assert len({row["method_id"] for row in rows}) == len(rows)
    assert all("checkpoint" not in str(row["training_seeds"]).lower() for row in rows)
    predictor = next(row for row in rows if row["method_id"] == "counterfactual_predictor")
    assert predictor["algorithm_family"] == "diagnostic supervised predictor"
    assert predictor["real_native_training_transitions"] == 0


def test_hybrid_paired_savings_and_shared_route_denominator():
    rows = {row["comparison_id"]: row for row in paired_rows()}
    expected = {
        "h11_vs_time_aware": 1.0351750338147012,
        "h29_vs_time_aware": 1.2450393062050755,
        "h47_vs_time_aware": 1.7001296818846239,
    }
    for key, value in expected.items():
        assert rows[key]["distinct_task_count"] == 35
        assert rows[key]["paired_count"] == 35
        assert rows[key]["arithmetic_mean_of_per_task_percentage_saving"] == pytest.approx(value)


def test_causal_mpc_and_temporal_headlines_are_recomputed():
    rows = {row["comparison_id"]: row for row in paired_rows()}
    assert rows["adaptive_vs_time_aware"][
        "arithmetic_mean_of_per_task_percentage_saving"
    ] == pytest.approx(5.421834807082291)
    assert rows["adaptive_vs_fixed"][
        "arithmetic_mean_of_per_task_percentage_saving"
    ] == pytest.approx(0.8011839951300701)
    assert rows["continue_vs_reset"][
        "arithmetic_mean_of_per_task_percentage_saving"
    ] == pytest.approx(-0.5620832885179694)


def test_comparison_basis_is_explicit_everywhere():
    allowed = {
        "PAIRED_SAME_TASKS",
        "PAIRED_SAME_STREAM",
        "INDEPENDENT_PANEL",
        "CONTEXTUAL_ONLY",
        "NOT_COMPARABLE",
    }
    assert all(row["comparison_basis"] in allowed for row in evidence_rows())
    assert all(row["comparison_basis"] in allowed for row in method_rows())


def test_generated_tables_are_derived_from_machine_records():
    outputs = generated_outputs()
    assert any(path.name == "method_registry.json" for path in outputs)
    assert any(path.name == "master_evidence_matrix.csv" for path in outputs)
