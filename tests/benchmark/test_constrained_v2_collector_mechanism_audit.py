import json
from pathlib import Path

from benchmarks.constrained_v2_collector_mechanism_audit import analyze

HERE = Path(analyze.__file__).resolve().parent
RESULTS = json.loads((HERE / "results.json").read_text(encoding="utf-8"))


def test_quantile_uses_linear_empirical_interpolation():
    assert analyze.quantile([0.0, 10.0], 0.25) == 2.5


def test_behavior_categories_are_operational_and_exclusive():
    base = {
        "feasible": False,
        "completed": False,
        "energy_kwh": 0.0,
        "max_speed_violation_m_s": 0.0,
        "travel_time_s": 180.0,
    }
    assert (
        analyze.episode_behavior_state(base)
        == "STANDSTILL_OR_NEAR_STANDSTILL"
    )
    assert (
        analyze.episode_behavior_state({**base, "energy_kwh": 0.1})
        == "CRAWLING_OR_PARTIAL"
    )
    assert (
        analyze.episode_behavior_state(
            {**base, "completed": True, "max_speed_violation_m_s": 0.1}
        )
        == "COMPLETES_WITH_SPEED_VIOLATION"
    )
    assert (
        analyze.episode_behavior_state({**base, "completed": True})
        == "COMPLETES_TOO_SLOW"
    )
    assert analyze.episode_behavior_state({**base, "feasible": True}) == "FEASIBLE"


def test_frozen_result_and_zero_interaction_contract():
    frozen = RESULTS["frozen_empirical_result"]
    assert frozen["classification"] == "SEED_HETEROGENEOUS"
    assert frozen["decision"] == "DO_NOT_FREEZE_COLLECTOR"
    assert RESULTS["source"]["full_commit_sha"] == analyze.SOURCE_STUDY_COMMIT
    zero = RESULTS["zero_new_scientific_interaction"]
    assert zero["new_training_transitions"] == 0
    assert zero["new_rl_updates"] == 0
    assert zero["new_policy_evaluation_episodes"] == 0
    assert zero["new_development_episodes"] == 0
    assert zero["new_validation_episodes"] == 0
    assert zero["new_paper_episodes"] == 0
    assert zero["environment_rollouts"] == 0
    assert zero["neural_network_forward_passes"] == 0


def test_checkpoint_and_episode_timelines_are_complete():
    expected = {
        "E11_episodic_seed11": 243,
        "T11_transitional_seed11": 233,
        "E29_episodic_seed29": 236,
        "T29_transitional_seed29": 238,
        "E47_episodic_seed47": 236,
        "T47_transitional_seed47": 206,
    }
    for condition, count in expected.items():
        assert len(RESULTS["checkpoint_timelines"][condition]) == 7
        assert len(RESULTS["episode_timelines"][condition]) == count
    for seed, shared_count in {"11": 233, "29": 236, "47": 206}.items():
        assert (
            len(RESULTS["paired_episode_deltas_episodic_minus_transitional"][seed])
            == shared_count
        )


def test_checkpoint_counters_preserve_episodic_backlogs():
    e11 = RESULTS["checkpoint_timelines"]["E11_episodic_seed11"]
    t11 = RESULTS["checkpoint_timelines"]["T11_transitional_seed11"]
    assert e11[1]["gradient_updates"] == 4962
    assert t11[1]["gradient_updates"] == 5000
    assert e11[-1]["gradient_updates"] == t11[-1]["gradient_updates"] == 30_000


def test_pid_exposure_and_nonstationarity_are_exact():
    summaries = RESULTS["condition_summaries"]
    for seed in analyze.SEEDS:
        episodic, _ = analyze.condition_for_name(seed, "EPISODIC")
        transitional, _ = analyze.condition_for_name(seed, "TRANSITIONAL")
        assert summaries[episodic]["pid_lagrange"][
            "post_episode_lambda_update_exposure"
        ]["updates"] == 30_000
        assert summaries[episodic]["within_episode_nonstationarity"][
            "total_actor_updates_while_episode_active"
        ] == 0
        assert summaries[transitional]["within_episode_nonstationarity"][
            "total_actor_updates_while_episode_active"
        ] > 29_000


def test_missing_replay_samples_are_not_inferred():
    for summary in RESULTS["condition_summaries"].values():
        replay = summary["replay_freshness"]
        assert replay["sample_age_distribution"] == "NOT_RECOVERABLE"
        assert replay["sample_fraction_last_256"] == "NOT_RECOVERABLE"
        assert replay["sample_fraction_last_1000"] == "NOT_RECOVERABLE"
        assert replay["sample_fraction_last_5000"] == "NOT_RECOVERABLE"


def test_sign_reversal_conclusion_is_descriptive_only():
    reversal = RESULTS["sign_reversal_analysis"]
    assert reversal["rsr_episodic_minus_transitional"] == {
        "11": 3,
        "29": -4,
        "47": -7,
    }
    matches = [
        row["feature"]
        for row in reversal["features"]
        if row["all_three_signs_match_rsr"]
    ]
    assert matches == ["final_lambda_deadline"]
    assert "endogenous" in reversal["conclusion"]


def test_stop_decision_and_execution_convention_are_distinct():
    gate = RESULTS["decision_gate"]
    assert gate["decision"] == "STOP_COLLECTOR_RESEARCH"
    assert gate["new_collector_ablation_authorized"] is False
    convention = gate["canonical_execution_convention"]
    assert convention["collector"] == "TRANSITIONAL"
    assert convention["collector_status"] == (
        "selected as an execution convention, not as a demonstrated superior "
        "collector"
    )
    assert gate["next_major_scientific_study"] == (
        "Same-information Scalar vs Constrained"
    )


def test_finalizer_recovery_integrity_is_confirmed():
    audit = RESULTS["finalizer_recovery_audit"]
    assert audit["classification"] == "REPORTING_ONLY_RECOVERY_CONFIRMED"
    assert all(audit["checks"].values())
    assert all(audit["final_checkpoint_hash_checks"].values())
    assert all(all(row.values()) for row in audit["actor_hash_checks"].values())


def test_analysis_code_and_selected_source_hashes_match():
    assert RESULTS["analysis"]["analysis_code_sha256"] == analyze.sha256_file(
        Path(analyze.__file__)
    )
    source_root = Path(RESULTS["source"]["worktree"])
    hashes = RESULTS["source"]["input_artifact_sha256"]
    selected = (
        "benchmarks/constrained_v2_collector_replication/results.json",
        "benchmarks/constrained_v2_collector_replication/freeze.json",
        "runs/constrained-v2-collector-replication/manifest.json",
    )
    for relative in selected:
        assert hashes[relative] == analyze.sha256_file(source_root / relative)
    checkpoint_hashes = [
        key for key in hashes if key.endswith("/checkpoint.pt")
    ]
    assert len(checkpoint_hashes) == 36
