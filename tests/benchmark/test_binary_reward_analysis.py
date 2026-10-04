from __future__ import annotations

import pytest

from benchmarks.binary_reward.analysis import decision_gate


def _criteria(passed=True):
    return {
        name: {"passed": passed}
        for name in (
            "pooled_validation_rsr",
            "minimum_seed_rsr",
            "seed_count_at_half_rsr",
            "minimum_training_successes_per_seed",
            "maximum_peak_to_final_rsr_drop",
        )
    }


def _seeds(*values):
    return {
        str(seed): {"requirement_satisfaction_rate": value}
        for seed, value in zip((11, 29, 47), values)
    }


def _training(successes):
    return {"pooled_successful_training_episode_count": successes}


@pytest.mark.parametrize(
    ("criteria", "seeds", "training", "feasible", "expected"),
    (
        (_criteria(), _seeds(0.6, 0.6, 0.6), _training(10), 18, "A"),
        (_criteria(False), _seeds(0.0, 0.0, 0.0), _training(0), 0, "D"),
        (_criteria(False), _seeds(0.0, 0.2, 0.7), _training(8), 8, "C"),
        (_criteria(False), _seeds(0.3, 0.4, 0.4), _training(8), 10, "B"),
        (_criteria(False), _seeds(0.0, 0.1, 0.2), _training(8), 2, "E"),
        (_criteria(False), _seeds(0.1, 0.1, 0.1), _training(0), 3, "F"),
    ),
)
def test_preregistered_decision_gates(criteria, seeds, training, feasible, expected):
    assert decision_gate(criteria, seeds, training, feasible)[0] == expected
