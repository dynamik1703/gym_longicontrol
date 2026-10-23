from __future__ import annotations

import numpy as np
import pytest

from benchmarks.feedback_chunk_rl_v1.chunk_decoder import (
    RESIDUAL_SCALE,
    ResidualChunk,
    combined_proposal,
    method_spec,
)


def test_method_contracts_are_exact() -> None:
    chunk = method_spec("CHUNK3_RESIDUAL_SAC")
    residual = method_spec("RESIDUAL1S_SAC")
    assert (chunk.action_dim, chunk.actor_interval_native_steps) == (3, 30)
    assert (residual.action_dim, residual.actor_interval_native_steps) == (1, 10)
    assert (chunk.target_entropy, residual.target_entropy) == (-3.0, -1.0)
    assert RESIDUAL_SCALE == 0.25


def test_three_phases_are_distinct_and_not_resampled() -> None:
    chunk = ResidualChunk(np.array([-0.75, 0.0, 0.625]), 90, 4)
    assert [chunk.phase(step) for step in (90, 99, 100, 109, 110, 119)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert [chunk.correction(step) for step in (90, 100, 110)] == [
        -0.75,
        0.0,
        0.625,
    ]
    with pytest.raises(RuntimeError):
        chunk.phase(120)


def test_combined_proposal_is_scaled_and_clipped() -> None:
    assert combined_proposal(0.4, -0.5) == 0.275
    assert combined_proposal(0.9, 1.0) == 1.0
    assert combined_proposal(-0.9, -1.0) == -1.0


def test_chunk_cancellation_is_sticky() -> None:
    chunk = ResidualChunk(np.zeros(3), 0, 0)
    chunk.cancel(native_step=7, reason="FIRST")
    chunk.cancel(native_step=8, reason="SECOND")
    assert chunk.cancelled
    assert chunk.cancellation_reason == "FIRST"
    assert chunk.cancellation_native_step == 7
