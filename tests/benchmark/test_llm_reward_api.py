from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from benchmarks.llm_reward.candidate_validation import (
    ingest_candidate,
    load_reward_function,
    repair_request_payload,
    source_sha256,
    validate_candidate_source,
    validate_visible_rationale,
)
from benchmarks.llm_reward.history import (
    initialize_history,
    load_history,
    new_history,
    save_history,
    start_search,
)
from benchmarks.llm_reward.protocol import load_protocol, protocol_sha256
from benchmarks.llm_reward.reward_api import (
    REWARD_CONTEXT_FIELDS,
    CandidateRewardWrapper,
    RewardContext,
    RewardOutput,
)
from benchmarks.scalar_sac.experiment import _base_environment
from benchmarks.scalar_sb3.config import load_configuration as load_scalar_sb3

# TEST FIXTURE - NOT AN LLM CANDIDATE. It exists only in ephemeral pytest paths.
VALID_REWARD_SOURCE = '''\
"""TEST FIXTURE - NOT AN LLM CANDIDATE."""
import math
from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def _bounded(value):
    return math.tanh(value)


def compute_reward(ctx: RewardContext) -> RewardOutput:
    progress = _bounded(ctx.delta_position_m / 10.0)
    energy = -abs(ctx.step_energy_kwh)
    return RewardOutput(
        reward=progress + energy,
        components={"progress": progress, "energy": energy},
    )
'''


def _context(**updates):
    values = {
        "position_m": 10.0,
        "previous_position_m": 9.0,
        "delta_position_m": 1.0,
        "velocity_m_s": 5.0,
        "previous_velocity_m_s": 4.9,
        "acceleration_m_s2": 1.0,
        "previous_acceleration_m_s2": 0.5,
        "action": 0.2,
        "speed_limit_m_s": 13.9,
        "future_speed_limit_1_m_s": 8.3,
        "future_speed_limit_2_m_s": 16.7,
        "distance_to_future_limit_1_m": 75.0,
        "distance_to_future_limit_2_m": 150.0,
        "step_energy_kwh": 0.001,
        "cumulative_energy_kwh": 0.01,
        "elapsed_time_s": 1.0,
        "dt_s": 0.1,
        "route_length_m": 1000.0,
        "time_budget_s": 140.0,
        "completed": False,
        "episode_ended": False,
    }
    return RewardContext(**{**values, **updates})


def _open_history(tmp_path):
    protocol = load_protocol()
    path = tmp_path / "search_history.json"
    initialize_history(path, protocol)
    save_history(path, start_search(load_history(path, protocol)), protocol)
    return protocol, path


def test_protocol_is_exact_and_copies_frozen_scalar_sac():
    protocol = load_protocol()
    scalar = load_scalar_sb3()
    assert asdict(protocol.sac) == asdict(scalar.sac)
    assert protocol.search_budget.candidate_counts_by_generation == (5, 3, 3)
    assert protocol.search_budget.maximum_candidates == 11
    assert protocol.screening.training_seed == 11
    assert protocol.screening.simulator_transitions_per_candidate == 50_000
    assert protocol.screening.evaluation_checkpoints == (10_000, 25_000, 50_000)
    assert protocol.screening.development_tracks == tuple(range(2000, 2009))
    assert protocol.final_evaluation.training_seeds == (11, 29, 47)
    assert protocol.final_evaluation.simulator_transitions_per_seed == 300_000
    assert len(protocol_sha256(protocol)) == 64


def test_reward_context_has_only_the_documented_immutable_whitelist():
    assert REWARD_CONTEXT_FIELDS == (
        "position_m",
        "previous_position_m",
        "delta_position_m",
        "velocity_m_s",
        "previous_velocity_m_s",
        "acceleration_m_s2",
        "previous_acceleration_m_s2",
        "action",
        "speed_limit_m_s",
        "future_speed_limit_1_m_s",
        "future_speed_limit_2_m_s",
        "distance_to_future_limit_1_m",
        "distance_to_future_limit_2_m",
        "step_energy_kwh",
        "cumulative_energy_kwh",
        "elapsed_time_s",
        "dt_s",
        "route_length_m",
        "time_budget_s",
        "completed",
        "episode_ended",
    )
    context = _context()
    with pytest.raises(FrozenInstanceError):
        context.position_m = 20.0
    with pytest.raises(TypeError):
        RewardContext(**{**asdict(context), "track_seed": 2000})
    with pytest.raises(ValueError, match="finite"):
        _context(velocity_m_s=float("nan"))


def test_reward_output_requires_finite_scalar_and_finite_named_components():
    output = RewardOutput(1, {"progress": np.float64(0.5)})
    assert output.reward == 1.0
    assert dict(output.components) == {"progress": 0.5}
    with pytest.raises(TypeError):
        output.components["energy"] = 0.0
    with pytest.raises(ValueError, match="finite scalar"):
        RewardOutput(float("inf"), {})
    with pytest.raises(ValueError, match="finite scalar"):
        RewardOutput(0.0, {"energy": float("nan")})
    with pytest.raises(ValueError, match="component name"):
        RewardOutput(0.0, {"Bad Name": 1.0})


def test_candidate_static_and_runtime_validation_and_hashing(tmp_path):
    report = validate_candidate_source(VALID_REWARD_SOURCE)
    assert report.valid
    assert report.source_sha256 == source_sha256(VALID_REWARD_SOURCE)
    assert report.smoke_component_names == ("energy", "progress")
    assert report.source_line_count > 5
    assert report.ast_node_count > report.numeric_constant_count > 0
    path = tmp_path / "reward.py"
    path.write_text(VALID_REWARD_SOURCE, encoding="utf-8")
    function, loaded = load_reward_function(path)
    first = function(_context())
    second = function(_context())
    assert first.reward == second.reward
    assert dict(first.components) == dict(second.components)
    assert loaded == report


@pytest.mark.parametrize(
    ("source", "error"),
    (
        ("import os\ndef compute_reward(ctx): return 0\n", "only 'import math'"),
        (
            "def compute_reward(ctx):\n    return open('x').read()\n",
            "forbidden identifier",
        ),
        (
            "import random\ndef compute_reward(ctx): return random.random()\n",
            "only 'import math'",
        ),
        (
            "value = 1\ndef compute_reward(ctx): return value\n",
            "top-level state",
        ),
        (
            "def compute_reward(ctx):\n    return ctx.track_seed\n",
            "not whitelisted",
        ),
        (
            "def compute_reward(ctx):\n    return RewardOutput(3000, {})\n",
            "protected track identifier",
        ),
        (
            "from benchmarks.scalar_sb3 import results\n"
            "def compute_reward(ctx): return RewardOutput(0, {})\n",
            "reward_api type import",
        ),
        (
            "def compute_reward(ctx):\n    return 1.0\n",
            "must return RewardOutput",
        ),
        (
            "def compute_reward(ctx, memory=[]):\n    return RewardOutput(0.0, {})\n",
            "defaults, variadics",
        ),
        (
            "def compute_reward(ctx):\n"
            "    while True:\n"
            "        pass\n"
            "    return RewardOutput(0.0, {})\n",
            "loops are forbidden",
        ),
        ("def compute_reward(:\n", "syntax error"),
    ),
)
def test_candidate_rejects_forbidden_or_invalid_code(source, error):
    report = validate_candidate_source(source)
    assert not report.valid
    assert error in " ".join(report.errors)


def test_candidate_wrapper_replaces_only_reward_and_exposes_components():
    protocol = load_protocol()
    observed = []

    def fixture_reward(ctx):
        observed.append(ctx)
        return RewardOutput(ctx.delta_position_m, {"progress": ctx.delta_position_m})

    environment = CandidateRewardWrapper(
        _base_environment(protocol),
        compute_reward=fixture_reward,
        task=protocol.task,
        candidate_id="test-fixture",
        source_sha256="0" * 64,
    )
    try:
        observation, _ = environment.reset(seed=2)
        assert observation.shape == (8,)
        next_observation, reward, terminated, truncated, info = environment.step(
            np.array([0.5])
        )
        assert next_observation.shape == observation.shape
        assert reward == pytest.approx(observed[0].delta_position_m)
        assert not terminated and not truncated
        assert info["llm_reward_components"] == {"progress": reward}
        assert "historical_reward" in info
        assert observed[0].time_budget_s == 140.0
        assert observed[0].dt_s == 0.1
    finally:
        environment.close()


def test_ingestion_assigns_hash_metadata_and_effort_counts(tmp_path):
    protocol, history_path = _open_history(tmp_path)
    source = tmp_path / "incoming.py"
    source.write_text(VALID_REWARD_SOURCE, encoding="utf-8")
    generations = tmp_path / "generations"
    candidate_id, report = ingest_candidate(
        source,
        rationale="Uses bounded progress and an energy term.",
        generation=0,
        history_path=history_path,
        protocol=protocol,
        generations_root=generations,
    )
    assert candidate_id == "g0-c01"
    assert report.valid
    history = load_history(history_path, protocol)
    candidate = history["candidates"][0]
    assert candidate["source_sha256"] == source_sha256(VALID_REWARD_SOURCE)
    assert candidate["engineering_effort"] == {
        "reward_component_count": 2,
        "numeric_constant_count": report.numeric_constant_count,
        "source_line_count": report.source_line_count,
        "ast_node_count": report.ast_node_count,
    }
    metadata = json.loads(
        (generations / "g0" / "g0-c01.json").read_text(encoding="utf-8")
    )
    assert metadata["source_sha256"] == candidate["source_sha256"]
    assert history["engineering_effort"]["generated_reward_candidate_count"] == 1


def test_one_repair_is_allowed_then_technical_failure_consumes_slot(tmp_path):
    protocol, history_path = _open_history(tmp_path)
    invalid = tmp_path / "invalid.py"
    invalid.write_text("import os\n", encoding="utf-8")
    generations = tmp_path / "generations"
    candidate_id, report = ingest_candidate(
        invalid,
        rationale="A deliberately invalid test fixture.",
        generation=0,
        history_path=history_path,
        protocol=protocol,
        generations_root=generations,
    )
    assert candidate_id == "g0-c01" and not report.valid
    repaired_id, repaired = ingest_candidate(
        invalid,
        rationale="A deliberately invalid test fixture.",
        generation=0,
        history_path=history_path,
        protocol=protocol,
        generations_root=generations,
        repair_candidate_id=candidate_id,
    )
    assert repaired_id == candidate_id and not repaired.valid
    history = load_history(history_path, protocol)
    assert history["candidates"][0]["status"] == "technical_failure"
    assert history["candidates"][0]["repair_attempt_count"] == 1
    valid = tmp_path / "valid.py"
    valid.write_text(VALID_REWARD_SOURCE, encoding="utf-8")
    next_id, _ = ingest_candidate(
        valid,
        rationale="A separate deterministic test fixture.",
        generation=0,
        history_path=history_path,
        protocol=protocol,
        generations_root=generations,
    )
    assert next_id == "g0-c02"
    with pytest.raises(RuntimeError, match="Only a repair-required"):
        ingest_candidate(
            valid,
            rationale="A separate deterministic test fixture.",
            generation=0,
            history_path=history_path,
            protocol=protocol,
            generations_root=generations,
            repair_candidate_id=candidate_id,
        )


def test_visible_rationale_is_sanitized():
    assert validate_visible_rationale("Uses nonlinear progress.")
    with pytest.raises(ValueError, match="forbidden research information"):
        validate_visible_rationale("Copy the Constrained V2 result.")
    with pytest.raises(ValueError, match="protected track"):
        validate_visible_rationale("Optimize track 4000 specifically.")


def test_repair_payload_contains_only_checker_errors():
    report = validate_candidate_source("import os\n")
    payload = repair_request_payload("g0-c01", report)
    assert set(payload) == {
        "candidate_id",
        "status",
        "repair_feedback_scope",
        "errors",
    }
    assert payload["status"] == "repair_required"
    assert payload["repair_feedback_scope"] == "validation-errors-only"
    assert payload["errors"] == list(report.errors)


def test_repository_history_is_pristine_and_not_started():
    protocol = load_protocol()
    history = load_history(
        "benchmarks/llm_reward/search_history.json",
        protocol,
    )
    assert history == new_history(protocol)
    assert history["status"] == "NOT_STARTED"
    assert history["candidates"] == []
    assert history["engineering_effort"]["screening_rl_transitions"] == 0
    assert history["engineering_effort"]["final_training_transitions"] == 0
