"""Synthetic adversarial correctness scenarios, separate from performance tasks."""

from __future__ import annotations

import numpy as np

from gym_longicontrol.domain.track import Track


def make_track(positions, limits):
    return Track(
        np.asarray(positions, dtype=float), np.asarray(limits, dtype=float) / 3.6
    )


def fixtures():
    return {
        "constant_20": make_track([0], [20]),
        "constant_100": make_track([0], [100]),
        "60_to_20": make_track([0, 300], [60, 20]),
        "consecutive_40_reductions": make_track([0, 400, 500], [100, 60, 20]),
        "exact_150_preview": make_track([0, 150], [60, 20]),
        "two_preview_signs": make_track([0, 100, 200, 300, 400], [40, 60, 40, 60, 20]),
        "final_near_1000": make_track([0, 999], [60, 20]),
        "repeated_adversarial": make_track(
            list(range(0, 1000, 100)), [60, 20, 60, 20, 60, 20, 60, 20, 60, 20]
        ),
    }
