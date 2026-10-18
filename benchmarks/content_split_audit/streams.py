"""Reset-only reconstruction helpers; these functions never call ``env.step``."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from gym_longicontrol.domain.state import SimulationConfig
from gym_longicontrol.domain.track import StochasticTrackGenerator, Track

from .fingerprints import TaskFingerprint, fingerprint_track

DEVELOPMENT_SEEDS = tuple(range(2000, 2009))
VALIDATION_SEEDS = tuple(range(3000, 3009))
PAPER_SEEDS = tuple(range(4000, 4018))
ALL_NUMERICALLY_RESERVED = frozenset(
    (*range(1000, 1009), *DEVELOPMENT_SEEDS, *VALIDATION_SEEDS, *PAPER_SEEDS)
)


class SealedSplitError(ValueError):
    """Raised before any Paper-final content can be constructed."""


def track_from_seed(seed: int) -> Track:
    """Construct static content exactly as the first reset with ``seed`` would."""

    if int(seed) in PAPER_SEEDS:
        raise SealedSplitError("Paper-final task construction is forbidden")
    rng = np.random.default_rng(int(seed))
    return StochasticTrackGenerator(SimulationConfig().track_length_m)(rng)


def explicit_seed_stream(
    training_seed: int,
    *,
    domain_tag: int,
    count: int,
    upper_bound: int = 2**31 - 1,
    excluded: frozenset[int] = ALL_NUMERICALLY_RESERVED,
) -> list[int]:
    rng = np.random.default_rng(np.random.SeedSequence([training_seed, domain_tag]))
    result: list[int] = []
    while len(result) < count:
        candidate = int(rng.integers(0, upper_bound))
        if candidate not in excluded:
            result.append(candidate)
    return result


def continuous_tracks(
    initial_seed: int, used_draw_indices: Sequence[int]
) -> list[Track]:
    if not used_draw_indices:
        return []
    if list(used_draw_indices) != sorted(set(used_draw_indices)):
        raise ValueError("used_draw_indices must be strictly increasing")
    generator = StochasticTrackGenerator(SimulationConfig().track_length_m)
    rng = np.random.default_rng(initial_seed)
    wanted = iter(used_draw_indices)
    target = next(wanted)
    tracks: list[Track] = []
    for draw_index in range(used_draw_indices[-1] + 1):
        track = generator(rng)
        if draw_index == target:
            tracks.append(track)
            try:
                target = next(wanted)
            except StopIteration:
                break
    return tracks


def fingerprint_rows(
    tracks: Iterable[Track],
    task_semantics: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    numeric_seeds: Sequence[int | None],
    draw_indices: Sequence[int | None] | None = None,
) -> list[dict[str, Any]]:
    tracks = list(tracks)
    if len(tracks) != len(numeric_seeds):
        raise ValueError("tracks and numeric_seeds must have equal lengths")
    semantics = (
        list(task_semantics)
        if not isinstance(task_semantics, Mapping)
        else [task_semantics] * len(tracks)
    )
    if len(semantics) != len(tracks):
        raise ValueError("one task-semantic mapping is required per track")
    if draw_indices is None:
        draw_indices = [None] * len(tracks)
    rows = []
    for index, (track, description, seed, draw_index) in enumerate(
        zip(tracks, semantics, numeric_seeds, draw_indices)
    ):
        fingerprint = fingerprint_track(track, description, numeric_seed=seed)
        rows.append(
            {
                "episode_index": index,
                "reset_seed": seed,
                "rng_draw_index": draw_index,
                "route_sha256": fingerprint.route_sha256,
                "full_task_sha256": fingerprint.full_task_sha256,
            }
        )
    return rows


def duplicate_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(field: str) -> dict[str, Any]:
        counts = Counter(row[field] for row in rows)
        distribution = Counter(counts.values())
        return {
            "unique": len(counts),
            "duplicate_episode_rate": (len(rows) - len(counts)) / len(rows),
            "maximum_multiplicity": max(counts.values()),
            "multiplicity_distribution": {
                str(key): distribution[key] for key in sorted(distribution)
            },
        }

    return {
        "generated_episodes": len(rows),
        "route": summarize("route_sha256"),
        "full_task": summarize("full_task_sha256"),
    }


def collisions(
    training_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    stream: str,
) -> list[dict[str, Any]]:
    route_lookup: dict[str, list[Mapping[str, Any]]] = {}
    full_lookup: dict[str, list[Mapping[str, Any]]] = {}
    for row in reference_rows:
        route_lookup.setdefault(str(row["route_sha256"]), []).append(row)
        full_lookup.setdefault(str(row["full_task_sha256"]), []).append(row)
    result = []
    for training in training_rows:
        matches = {
            int(row["reset_seed"])
            for row in route_lookup.get(str(training["route_sha256"]), [])
        } | {
            int(row["reset_seed"])
            for row in full_lookup.get(str(training["full_task_sha256"]), [])
        }
        for reference_seed in sorted(matches):
            route_match = any(
                int(row["reset_seed"]) == reference_seed
                for row in route_lookup.get(str(training["route_sha256"]), [])
            )
            full_match = any(
                int(row["reset_seed"]) == reference_seed
                for row in full_lookup.get(str(training["full_task_sha256"]), [])
            )
            result.append(
                {
                    "training_stream": stream,
                    "training_episode_index": training["episode_index"],
                    "training_reset_seed": training["reset_seed"],
                    "training_rng_draw_index": training["rng_draw_index"],
                    "split": split,
                    "reference_seed": reference_seed,
                    "route_sha256_match": route_match,
                    "full_task_sha256_match": full_match,
                    "route_sha256": training["route_sha256"],
                    "full_task_sha256": training["full_task_sha256"],
                }
            )
    return result


def fingerprint_only(fingerprint: TaskFingerprint) -> dict[str, Any]:
    return {
        "route_sha256": fingerprint.route_sha256,
        "full_task_sha256": fingerprint.full_task_sha256,
    }
