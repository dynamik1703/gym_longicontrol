from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from benchmarks.constrained_rl_v2.config import load_configuration as load_v2
from benchmarks.constrained_rl_v2.costs import deadline_deficit_cost, deadline_state
from benchmarks.requirement_conditioned.adapter import collect_training_episode
from benchmarks.requirement_conditioned.config import (
    configuration_sha256,
    load_configuration,
)
from benchmarks.requirement_conditioned.experiment import _requirement_kind
from benchmarks.requirement_conditioned.feasibility import development_analysis
from benchmarks.requirement_conditioned.requirements import (
    BalancedMarginSampler,
    RequirementConditionedTaskWrapper,
)
from benchmarks.scalar_sac.experiment import _base_environment

ROOT = Path(__file__).parents[2]


def wrapped(*, margins=(20.0, 40.0, 60.0), sampler_seed=7):
    configuration = load_configuration()
    return RequirementConditionedTaskWrapper(
        _base_environment(configuration),
        margins_s=margins,
        sampler_seed=sampler_seed,
        time_scale_s=configuration.requirements.time_scale_s,
        maximum_t_min_start_s=(configuration.requirements.maximum_t_min_start_s),
        energy_scale_kwh=configuration.objective.energy_scale_kwh,
    )


def test_canonical_configuration_is_stable_and_self_hashing():
    configuration = load_configuration()
    assert configuration.requirements.training_margins_s == (20.0, 40.0, 60.0)
    assert configuration.requirements.interpolation_margins_s == (30.0, 50.0)
    assert len(configuration_sha256(configuration)) == 64


def test_algorithm_configuration_exactly_matches_constrained_v2():
    assert asdict(load_configuration().algorithm) == asdict(load_v2().algorithm)
    assert asdict(load_configuration().objective) == asdict(load_v2().objective)
    assert load_configuration().constraints.names == load_v2().constraints.names
    assert (
        load_configuration().constraints.cost_limits
        == load_v2().constraints.cost_limits
    )


def test_balanced_sampler_has_one_of_each_margin_per_block():
    sampler = BalancedMarginSampler((20.0, 40.0, 60.0), seed=11)
    values = [sampler.sample() for _ in range(9)]
    assert all(
        sorted(values[index : index + 3]) == [20.0, 40.0, 60.0]
        for index in range(0, 9, 3)
    )


def test_balanced_sampler_is_deterministic_and_seeded():
    first = BalancedMarginSampler((20.0, 40.0, 60.0), seed=29)
    second = BalancedMarginSampler((20.0, 40.0, 60.0), seed=29)
    third = BalancedMarginSampler((20.0, 40.0, 60.0), seed=47)
    first_values = [first.sample() for _ in range(12)]
    assert first_values == [second.sample() for _ in range(12)]
    assert first_values != [third.sample() for _ in range(12)]


def test_training_collection_pins_margin_across_collector_internal_resets():
    class FakeCollector:
        def __init__(self, environment):
            self.environment = environment
            self.reset_arguments = []

        def reset_env(self, arguments):
            self.reset_arguments.append(arguments)

        def collect(self, *, n_episode, gym_reset_kwargs):
            assert n_episode == 1
            self.reset_arguments.append(gym_reset_kwargs)
            self.environment.last_completed_episode = type(
                "Completed", (), {"simulator_steps": 17, "costs": (0.0, 0.0)}
            )()
            return {"n/st": 17}

    class FakeEnvironment:
        last_completed_episode = None

    environment = FakeEnvironment()
    collector = FakeCollector(environment)
    stats, completed = collect_training_episode(
        collector, environment, requirement_margin_s=40.0
    )
    expected = {"options": {"requirement_margin_s": 40.0}}
    assert collector.reset_arguments == [expected, expected]
    assert stats["n/st"] == 17
    assert completed is environment.last_completed_episode


def test_requirement_calculation_and_observation_shape():
    environment = wrapped()
    try:
        observation, info = environment.reset(
            seed=2000, options={"requirement_margin_s": 20.0}
        )
        assert observation.shape == (10,)
        assert environment.observation_space.contains(observation)
        assert info["requirement_t_min_start_s"] == pytest.approx(117.0)
        assert info["requirement_max_time_s"] == pytest.approx(137.0)
        assert observation[-2] == pytest.approx(137.0 / 180.0)
        assert observation[-1] == 0.0
    finally:
        environment.close()


def test_public_observation_space_remains_eight_dimensional():
    environment = _base_environment(load_configuration())
    try:
        assert environment.observation_space.shape == (8,)
        observation, _ = environment.reset(seed=2000)
        assert observation.shape == (8,)
    finally:
        environment.close()


def test_elapsed_time_feature_uses_post_step_time():
    environment = wrapped()
    try:
        observation, _ = environment.reset(
            seed=2000, options={"requirement_margin_s": 20.0}
        )
        observation, *_ = environment.step(np.array([0.0]))
        assert observation[-1] == pytest.approx(0.1 / 180.0)
    finally:
        environment.close()


def test_requirement_changes_only_augmented_feature_at_reset():
    environment = wrapped()
    try:
        tight, _ = environment.reset(seed=2001, options={"requirement_margin_s": 20.0})
        loose, _ = environment.reset(seed=2001, options={"requirement_margin_s": 60.0})
        np.testing.assert_array_equal(tight[:8], loose[:8])
        assert tight[-2] != loose[-2]
        assert tight[-1] == loose[-1] == 0.0
    finally:
        environment.close()


def test_markov_timing_feature_distinguishes_elapsed_time():
    environment = wrapped()
    try:
        observation, _ = environment.reset(
            seed=2002, options={"requirement_margin_s": 40.0}
        )
        later = environment._augment(observation[:8], 10.0)
        np.testing.assert_array_equal(observation[:9], later[:9])
        assert later[-1] - observation[-1] == pytest.approx(10.0 / 180.0)
    finally:
        environment.close()


def test_no_privileged_deadline_state_leaks_into_observation():
    environment = wrapped()
    try:
        observation, info = environment.reset(
            seed=2003, options={"requirement_margin_s": 20.0}
        )
        assert observation.shape == (10,)
        assert "deadline_slack_s" not in info
        assert "deadline_deficit_s" not in info
        assert "deadline_optimistic_remaining_time_s" not in info
    finally:
        environment.close()


def test_parameterized_deadline_cost_tight_is_not_less_than_loose():
    environment = wrapped()
    try:
        environment.reset(seed=2004, options={"requirement_margin_s": 20.0})
        base = environment.unwrapped
        tight = deadline_state(
            elapsed_time_s=100.0,
            position_m=400.0,
            track=base.track,
            track_length_m=base.config.track_length_m,
            deadline_s=120.0,
        )
        loose = deadline_state(
            elapsed_time_s=100.0,
            position_m=400.0,
            track=base.track,
            track_length_m=base.config.track_length_m,
            deadline_s=160.0,
        )
        assert tight[1] < loose[1]
        assert tight[2] >= loose[2]
        assert deadline_deficit_cost(
            deficit_s=tight[2], dt_s=0.1, normalization_s=120.0
        ) >= deadline_deficit_cost(deficit_s=loose[2], dt_s=0.1, normalization_s=160.0)
    finally:
        environment.close()


def test_wrapper_is_deterministic_for_seed_requirement_and_actions():
    first = wrapped()
    second = wrapped()
    try:
        observations = []
        for environment in (first, second):
            observation, _ = environment.reset(
                seed=3000, options={"requirement_margin_s": 30.0}
            )
            rollout = [observation]
            for _ in range(10):
                observation, *_ = environment.step(np.array([0.25]))
                rollout.append(observation)
            observations.append(np.asarray(rollout))
        np.testing.assert_array_equal(*observations)
    finally:
        first.close()
        second.close()


def test_episode_specific_task_controls_feasibility_boundary():
    environment = wrapped()
    try:
        environment.reset(seed=3001, options={"requirement_margin_s": 30.0})
        assert environment.current_task.max_time_s == pytest.approx(87.9)
        assert environment.current_task.max_speed_violation_m_s == 0.0
    finally:
        environment.close()


def test_seen_and_unseen_requirement_labels():
    configuration = load_configuration()
    assert _requirement_kind(configuration, 20.0) == "seen"
    assert _requirement_kind(configuration, 30.0) == "interpolation"
    assert _requirement_kind(configuration, 37.0) == "canonical-140"


def test_invalid_or_leaking_reset_options_are_rejected():
    environment = wrapped()
    try:
        with pytest.raises(ValueError):
            environment.reset(seed=2, options={"deadline_deficit_s": 0.0})
        with pytest.raises(ValueError):
            environment.reset(seed=2, options={"requirement_margin_s": 0.0})
    finally:
        environment.close()


def test_physics_artifact_matches_frozen_selection_and_validation_precheck():
    artifact = json.loads(
        (ROOT / "benchmarks/requirement_conditioned/feasibility.json").read_text()
    )
    assert artifact["training_margins_s"] == [20.0, 40.0, 60.0]
    assert artifact["interpolation_margins_s"] == [30.0, 50.0]
    assert artifact["maximum_development_deadline_s"] == pytest.approx(177.0)
    assert artifact["validation_precheck"][
        "all_tight_requirements_reached_by_fast_compliant_reference"
    ]
    assert not artifact["reserved_final_test_evaluated"]


def test_development_analysis_never_uses_reserved_tracks():
    result = development_analysis(load_configuration())
    seeds = {item["track_seed"] for item in result["tracks"]}
    validation = {
        item["track_seed"] for item in result["validation_precheck"]["tracks"]
    }
    assert seeds == set(range(2000, 2009))
    assert validation == set(range(3000, 3009))
    assert not (seeds | validation) & set(range(4000, 4018))
