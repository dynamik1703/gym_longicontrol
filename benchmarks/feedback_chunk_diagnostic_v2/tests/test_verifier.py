
import pytest

from benchmarks.feedback_chunk_diagnostic_v2.preflight import _schema_gate
from benchmarks.feedback_chunk_diagnostic_v2.verifier import verify_native_evidence


def test_explicit_not_applicable_schema_and_independent_accounting():
    assert all(_schema_gate().values())


def test_missing_evidence_is_not_silently_treated_as_zero():
    # Exercise the strict empty/missing path without depending on a simulator reset.
    with pytest.raises(RuntimeError, match="empty"):
        verify_native_evidence([])
