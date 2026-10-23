import json

import pytest

from benchmarks.paper_synthesis_v1.feasibility import (
    PAPER_NUMERIC_SEEDS,
    PaperSeedAccessError,
    assert_not_paper_seed,
    build_audit,
    classify_task,
    conservative_discrete_lower_bound,
    load_historical_tasks,
)


def test_track_10200029_has_formal_discrete_proof():
    tasks = load_historical_tasks()
    full_hash = "6f0d03873d9039e863cb40cecfb024aac0341f95e878272a86b1f1805801b883"
    task = tasks[full_hash]["canonical_task"]
    classification, bound = classify_task(task)
    assert classification == "PROVEN_INFEASIBLE"
    assert bound is not None and bound.lower_bound_s is not None
    assert float(bound.lower_bound_s) > 140.0
    assert float(bound.continuous_limit_only_s) == pytest.approx(144.0)


def test_continuous_only_bound_is_not_a_classifier():
    tasks = load_historical_tasks()
    task = tasks["6f0d03873d9039e863cb40cecfb024aac0341f95e878272a86b1f1805801b883"][
        "canonical_task"
    ]
    altered = json.loads(json.dumps(task))
    altered["environment_parameters"]["integration"]["name"] = "unknown"
    bound = conservative_discrete_lower_bound(altered)
    assert bound.supported is False
    assert classify_task(altered)[0] == "UNRESOLVED"


def test_missing_content_and_failed_episode_remain_unresolved():
    assert classify_task(None)[0] == "UNRESOLVED"
    tasks = load_historical_tasks()
    task = next(iter(tasks.values()))["canonical_task"]
    failure = {
        "completed": False,
        "travel_time_s": 180.0,
        "max_speed_violation_m_s": 0.0,
        "route_sha256": task["route_sha256"],
        "full_task_sha256": task["_audit_full_task_sha256"],
        "source": "synthetic-test-record",
    }
    assert classify_task(task, witness_records=[failure])[0] != "FEASIBLE_WITNESS"


def test_aggregate_outcome_is_not_promoted_to_witness():
    tasks = load_historical_tasks()
    task = next(iter(tasks.values()))["canonical_task"]
    aggregate = {
        "completed": True,
        "travel_time_s": 100.0,
        "max_speed_violation_m_s": 0.0,
        "route_sha256": task["route_sha256"],
        "full_task_sha256": task["_audit_full_task_sha256"],
        "source": "aggregate-only-test",
        "aggregate_only": True,
    }
    assert classify_task(task, witness_records=[aggregate])[0] != "FEASIBLE_WITNESS"


def test_route_duplicates_do_not_create_independent_tasks():
    audit = build_audit()
    assert audit["coverage"]["unique_routes"] == len(
        {row["route_sha256"] for row in audit["tasks"]}
    )
    assert (
        audit["coverage"]["study_task_occurrences"]
        >= audit["coverage"]["unique_routes"]
    )


@pytest.mark.parametrize("seed", sorted(PAPER_NUMERIC_SEEDS))
def test_paper_numeric_identities_are_rejected_before_task_access(seed):
    with pytest.raises(PaperSeedAccessError):
        assert_not_paper_seed(seed)
