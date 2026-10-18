import json
from pathlib import Path

from benchmarks.constrained_v2_repro_audit.harness import (
    collector_buffer_fixture,
    collector_reset_counts,
    current_track_fingerprints,
    divergence_ladder,
    historical_track_fingerprints,
    initial_policy_fingerprints,
    replay_fixture,
    summarize_track_overlap,
    update_schedule_fixture,
)


def test_collectors_consume_different_reset_counts() -> None:
    counts = collector_reset_counts()

    assert (
        counts["historical_fast_collector"]["resets_after_one_terminal_transition"] == 3
    )
    assert counts["current_collector"]["resets_after_one_terminal_transition"] == 2


def test_collectors_store_same_fixed_episode_transitions() -> None:
    fixture = collector_buffer_fixture()

    assert fixture["identical"] is True
    assert fixture["historical_fast_collector"]["done"] == [False, False, True]


def test_training_track_streams_diverge_at_first_episode() -> None:
    historical = historical_track_fingerprints(11, count=3)
    current = current_track_fingerprints(11, count=3)

    overlap = summarize_track_overlap(historical, current)
    assert overlap["first_divergence_index"] == 0
    assert overlap["same_at_same_index"] == 0
    assert [row["explicit_reset_seed"] for row in current] == [
        620023146,
        9646336,
        258433255,
    ]


def test_bounded_ladder_stops_at_first_track_divergence() -> None:
    ladder = divergence_ladder(audit_seed=0)

    assert ladder["levels"][0]["equal"] is True
    assert ladder["earliest_divergence_level"] == 2
    assert ladder["levels"][1]["equal"] is False
    assert ladder["longicontrol_simulator_transitions"] == 0
    assert all(row["status"] == "NOT_RUN" for row in ladder["levels"][3:])


def test_initial_constructors_match_under_one_runtime() -> None:
    fingerprints = initial_policy_fingerprints(11)

    assert fingerprints["identical"] is True
    assert fingerprints["historical_constructor"]["lagrangian"] == [0.0, 0.0]
    assert fingerprints["historical_constructor"]["alpha"] == 1.0


def test_replay_sampling_and_two_step_returns_are_repeatable() -> None:
    fixture = replay_fixture()

    assert fixture["repeat_identical"] is True
    assert fixture["sampled_indices"] == [4, 6, 2, 4, 1, 5]
    assert fixture["successor_indices"] == [1, 3, 3, 7]
    assert fixture["reward"] == [0.0, 2.0, 3.0, 6.0]
    assert fixture["speed_cost"] == [0.0, 0.2, 0.3, 0.6]
    assert fixture["deadline_cost"] == [0.0, 0.4, 0.6, 1.2]
    assert (
        fixture["two_step_returns_sha256"]
        == "1d9cc296f3459f21090f1cbce849933d045bc8de3b8416bc20f8fb1c6053768d"
    )


def test_update_schedule_places_pid_at_different_learner_states() -> None:
    fixture = update_schedule_fixture()

    assert fixture["historical"]["updates_after_pid"] == 180
    assert fixture["current"]["first_policy_change_step"] == 256
    assert fixture["current"]["updates_before_pid"] == 179
    assert fixture["current"]["updates_after_pid_on_terminal_step"] == 1


def test_machine_readable_audit_result_is_complete() -> None:
    path = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "constrained_v2_repro_audit"
        / "results.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))

    assert result["headline"]["primary_classification"] == "C"
    assert result["longicontrol_simulator_transitions_consumed"] == 0
    assert result["configuration"]["scientific_parameter_differences"] == []
    assert result["provenance"]["shared_source_comparison"]["all_identical"]
    assert result["collector_buffer_fixture"]["identical"]
    assert (
        result["bounded_end_to_end_fingerprint_ladder"]["earliest_divergence_level"]
        == 2
    )
    assert all(
        len(result["rng_and_tracks"]["by_seed"][str(seed)][key]) == 100
        for seed in (11, 29, 47)
        for key in (
            "historical_first_100_used_track_fingerprints",
            "current_first_100_used_track_fingerprints",
            "historical_first_100_episode_lengths",
            "current_first_100_episode_lengths",
        )
    )
