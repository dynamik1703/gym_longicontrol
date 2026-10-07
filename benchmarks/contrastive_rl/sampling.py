"""Explicit episode-safe future-pair and reference-goal semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FuturePairs:
    source_indices: np.ndarray
    future_indices: np.ndarray
    lags: np.ndarray


def future_probabilities(lags, *, gamma: float) -> np.ndarray:
    """Normalize the upstream gamma**lag distribution for positive lags."""

    lag_array = np.asarray(lags, dtype=np.int64)
    if lag_array.ndim != 1 or not len(lag_array) or (lag_array <= 0).any():
        raise ValueError("lags must be a nonempty vector of positive integers")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    weights = np.power(gamma, lag_array, dtype=np.float64)
    return weights / weights.sum()


def sample_future_pairs(
    episode_ids,
    *,
    gamma: float,
    rng: np.random.Generator,
    eligible=None,
    step_indices=None,
    episode_ends=None,
) -> FuturePairs:
    """Sample one strict future from the same episode for each valid source.

    Episode IDs and consecutive physical step indices define trajectory
    segments. A discontinuity therefore cannot be crossed even if discarded
    collector data happens to retain the same episode ID. Sources without a
    later eligible state are omitted. This explicit interface replaces
    upstream's fixed-length/Brax seed convention; conditional temporal weights
    remain ``gamma**lag``.
    """

    ids = np.asarray(episode_ids)
    if ids.ndim != 1:
        raise ValueError("episode_ids must be rank one")
    steps = (
        np.arange(len(ids), dtype=np.int64)
        if step_indices is None
        else np.asarray(step_indices, dtype=np.int64)
    )
    if steps.shape != ids.shape:
        raise ValueError("step_indices must match episode_ids")
    allowed = (
        np.ones(len(ids), dtype=bool)
        if eligible is None
        else np.asarray(eligible)
    )
    if allowed.shape != ids.shape or allowed.dtype != np.bool_:
        raise ValueError("eligible must be a boolean vector matching episode_ids")
    ends = (
        np.zeros(len(ids), dtype=bool)
        if episode_ends is None
        else np.asarray(episode_ends)
    )
    if ends.shape != ids.shape or ends.dtype != np.bool_:
        raise ValueError("episode_ends must be a boolean vector matching episode_ids")
    if len(ids) > 1 and np.any(ends[:-1] & (ids[:-1] == ids[1:])):
        raise ValueError("post-terminal rows must use a new physical episode ID")

    segment = np.zeros(len(ids), dtype=np.int64)
    for index in range(1, len(ids)):
        segment[index] = segment[index - 1] + int(
            ids[index] != ids[index - 1]
            or steps[index] != steps[index - 1] + 1
            or ends[index - 1]
        )

    sources: list[int] = []
    futures: list[int] = []
    lags: list[int] = []
    for source in range(len(ids)):
        candidates = np.flatnonzero(
            (np.arange(len(ids)) > source)
            & (segment == segment[source])
            & allowed
        )
        if not allowed[source] or ends[source] or not len(candidates):
            continue
        candidate_lags = candidates - source
        future = int(
            rng.choice(
                candidates,
                p=future_probabilities(candidate_lags, gamma=gamma),
            )
        )
        sources.append(source)
        futures.append(future)
        lags.append(future - source)
    return FuturePairs(
        source_indices=np.asarray(sources, dtype=np.int64),
        future_indices=np.asarray(futures, dtype=np.int64),
        lags=np.asarray(lags, dtype=np.int64),
    )


def lags_to_seconds(lags, *, dt_s: float) -> np.ndarray:
    lag_array = np.asarray(lags, dtype=np.int64)
    if (lag_array < 0).any() or not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("lags must be nonnegative and dt_s finite and positive")
    return lag_array.astype(np.float64) * dt_s


def in_batch_reference_indices(batch_size: int) -> np.ndarray:
    """Return all goal columns used as the reference set by upstream InfoNCE."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return np.broadcast_to(np.arange(batch_size), (batch_size, batch_size)).copy()


def equivalent_goal_rate(goals, *, atol: float = 0.0) -> float:
    """Fraction of unordered pairs that are identical/equivalent."""

    array = np.asarray(goals, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("goals must be a rank-two array")
    if len(array) < 2:
        return 0.0
    equal = np.all(
        np.isclose(array[:, None, :], array[None, :, :], atol=atol, rtol=0.0),
        axis=-1,
    )
    upper = equal[np.triu_indices(len(array), k=1)]
    return float(np.mean(upper))
