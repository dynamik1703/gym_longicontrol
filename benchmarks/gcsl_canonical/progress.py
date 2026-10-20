"""Concise summaries of already scheduled GCSL events."""

from __future__ import annotations

from typing import Any


def checkpoint_progress(
    *,
    seed: int,
    transitions: int,
    updates: int,
    completed_episodes: int,
    canonical_successes: int,
    diagnostic_rsr: int,
    maximum_replay_progress: float | None,
    canonical_hindsight_tuples: int,
    walltime_s: float,
    healthy: bool,
) -> str:
    progress = 100.0 * transitions / 300_000
    maximum = (
        "n/a" if maximum_replay_progress is None else f"{maximum_replay_progress:.6f}"
    )
    return "\n".join(
        (
            f"G{seed} | {transitions // 1000}k / 300k | {progress:.1f}%",
            f"updates: {updates:,} / 7,250",
            f"completed episodes: {completed_episodes}",
            f"canonical training successes: {canonical_successes}",
            f"diagnostic RSR: {diagnostic_rsr}/18",
            f"max replay progress: {maximum}",
            f"canonical hindsight tuples: {canonical_hindsight_tuples:,}",
            f"walltime: {walltime_s:.1f}s",
            f"status: {'HEALTHY' if healthy else 'CHECK_REQUIRED'}",
        )
    )


def status_progress(payload: dict[str, Any]) -> str:
    return checkpoint_progress(
        seed=int(payload["training_seed"]),
        transitions=int(payload["real_transitions"]),
        updates=int(payload["supervised_updates"]),
        completed_episodes=int(payload["completed_episodes"]),
        canonical_successes=int(payload["canonical_training_successes"]),
        diagnostic_rsr=int(payload.get("latest_diagnostic_rsr", 0)),
        maximum_replay_progress=payload.get("maximum_replay_progress"),
        canonical_hindsight_tuples=int(payload["canonical_hindsight_tuples"]),
        walltime_s=float(payload["walltime_s"]),
        healthy=payload.get("status") == "HEALTHY",
    )
