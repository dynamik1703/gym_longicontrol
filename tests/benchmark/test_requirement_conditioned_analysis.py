from __future__ import annotations

import pytest

from benchmarks.requirement_conditioned.analysis import (
    _average_ranks,
    _decision_gate,
    spearman_correlation,
)


def test_average_ranks_use_average_for_ties():
    assert _average_ranks([4.0, 2.0, 2.0, 8.0]).tolist() == [3.0, 1.5, 1.5, 4.0]


def test_spearman_correlation_direction_and_undefined_constant():
    assert spearman_correlation([1, 2, 3], [2, 4, 8]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [8, 4, 2]) == pytest.approx(-1.0)
    assert spearman_correlation([1, 2, 3], [4, 4, 4]) is None


def _criteria(*, rsr=True, time=True, energy=True, interpolation=True):
    return {
        "overall_rsr": {"passed": rsr},
        "minimum_requirement_rsr": {"passed": rsr},
        "minimum_seed_rsr": {"passed": rsr},
        "travel_time_pairwise_monotonicity": {"passed": time},
        "energy_pairwise_monotonicity": {"passed": energy},
        "requirement_sensitive_group_rate": {"passed": time},
        "interpolation_rsr_gap": {"passed": interpolation},
        "interpolation_directional_rate": {"passed": interpolation},
    }


def _margins(tight=0.6, loose=0.6):
    return {
        "20": {"requirement_satisfaction_rate": tight},
        "30": {"requirement_satisfaction_rate": loose},
        "40": {"requirement_satisfaction_rate": loose},
        "50": {"requirement_satisfaction_rate": loose},
        "60": {"requirement_satisfaction_rate": loose},
    }


@pytest.mark.parametrize(
    ("criteria", "margins", "expected"),
    (
        (_criteria(), _margins(), "F"),
        (_criteria(energy=False), _margins(), "A"),
        (_criteria(time=False), _margins(), "B"),
        (_criteria(interpolation=False), _margins(), "C"),
        (_criteria(rsr=False), _margins(tight=0.0, loose=0.7), "D"),
        (_criteria(rsr=False), _margins(tight=0.0, loose=0.1), "E"),
    ),
)
def test_decision_gates_are_reachable(criteria, margins, expected):
    assert _decision_gate(criteria, margins)[0] == expected
