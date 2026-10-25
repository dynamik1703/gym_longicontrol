import pytest

from benchmarks.feedback_chunk_diagnostic_v2.evaluation import (
    applicable,
    not_applicable,
)


def test_applicability_encodings_are_unambiguous():
    assert applicable(False) == {"status": "APPLICABLE", "value": False}
    assert not_applicable("not computed") == {
        "status": "NOT_APPLICABLE",
        "reason": "not computed",
        "value": None,
    }


def test_unknown_controller_is_rejected_before_reset():
    from benchmarks.feedback_chunk_diagnostic_v2.evaluation import evaluate_controller

    with pytest.raises(PermissionError, match="not registered"):
        evaluate_controller(numeric_seed=1, controller_id="UNKNOWN")
