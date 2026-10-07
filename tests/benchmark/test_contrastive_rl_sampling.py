import numpy as np
import pytest

from benchmarks.contrastive_rl.sampling import (
    equivalent_goal_rate,
    future_probabilities,
    in_batch_reference_indices,
    lags_to_seconds,
    sample_future_pairs,
)


def test_future_probabilities_match_discounted_temporal_rule():
    probabilities = future_probabilities([1, 2, 3], gamma=0.5)
    np.testing.assert_allclose(probabilities, np.array([4, 2, 1]) / 7)
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    np.testing.assert_allclose(lags_to_seconds([1, 3], dt_s=0.2), [0.2, 0.6])


def test_future_pairs_are_strict_and_never_cross_episode_boundaries():
    pairs = sample_future_pairs(
        [10, 10, 10, 11, 11, 12], gamma=0.99, rng=np.random.default_rng(7)
    )
    ids = np.array([10, 10, 10, 11, 11, 12])
    assert np.all(pairs.future_indices > pairs.source_indices)
    np.testing.assert_array_equal(
        ids[pairs.source_indices], ids[pairs.future_indices]
    )
    np.testing.assert_array_equal(pairs.source_indices, [0, 1, 3])


def test_future_pairs_do_not_cross_discarded_collector_transition():
    pairs = sample_future_pairs(
        [5, 5, 5, 5],
        step_indices=[0, 1, 3, 4],
        gamma=0.99,
        rng=np.random.default_rng(4),
    )
    assert set(zip(pairs.source_indices, pairs.future_indices, strict=True)) <= {
        (0, 1),
        (2, 3),
    }


def test_discarded_state_is_not_sampled_as_a_future():
    pairs = sample_future_pairs(
        [1, 1, 1],
        eligible=np.array([True, True, False]),
        gamma=0.99,
        rng=np.random.default_rng(1),
    )
    np.testing.assert_array_equal(pairs.source_indices, [0])
    np.testing.assert_array_equal(pairs.future_indices, [1])


def test_real_terminal_outcome_can_be_future_but_never_a_source_after_termination():
    pairs = sample_future_pairs(
        [8, 8],
        episode_ends=[False, True],
        gamma=0.99,
        rng=np.random.default_rng(2),
    )
    np.testing.assert_array_equal(pairs.source_indices, [0])
    np.testing.assert_array_equal(pairs.future_indices, [1])


def test_post_terminal_row_with_same_episode_identity_is_rejected():
    with pytest.raises(ValueError, match="post-terminal"):
        sample_future_pairs(
            [8, 8, 8],
            episode_ends=[False, True, False],
            gamma=0.99,
            rng=np.random.default_rng(2),
        )


def test_reference_columns_and_duplicate_goal_diagnostic():
    np.testing.assert_array_equal(
        in_batch_reference_indices(3),
        [[0, 1, 2], [0, 1, 2], [0, 1, 2]],
    )
    goals = np.array([[1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert equivalent_goal_rate(goals) == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("gamma", [0.0, 1.0, -0.1])
def test_invalid_discount_is_rejected(gamma):
    with pytest.raises(ValueError):
        future_probabilities([1], gamma=gamma)
