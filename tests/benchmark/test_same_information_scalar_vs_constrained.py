"""Scientific-integrity tests for the same-information study."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.constrained_rl.costs import objective_reward, speed_cost
from benchmarks.constrained_rl_v2.costs import deadline_deficit_cost
from benchmarks.same_information_scalar_vs_constrained.checkpointing import (
    capture_rng_states,
    load_verified,
    validate_payload,
    write_verified,
)
from benchmarks.same_information_scalar_vs_constrained.config import (
    CHECKPOINTS,
    TRAINING_SEEDS,
    load_configuration,
    require_main_training_authorized,
    require_validation_authorized,
)
from benchmarks.same_information_scalar_vs_constrained.evaluation import evaluate_paper
from benchmarks.same_information_scalar_vs_constrained.initialization import (
    verify_initialization,
)
from benchmarks.same_information_scalar_vs_constrained.interpretation import classify
from benchmarks.same_information_scalar_vs_constrained.scalarization import (
    frozen_scalarization,
    n_step_scalar_return,
    scalar_reward,
)
from benchmarks.same_information_scalar_vs_constrained.signals import (
    SameInformationScalarWrapper,
    canonical_transition_signals,
)
from benchmarks.same_information_scalar_vs_constrained.split import (
    build_split_ledger,
    diagnostic_seeds,
    validation_seeds,
)


def test_frozen_matched_configuration_and_authorization():
    configuration = load_configuration()
    assert TRAINING_SEEDS == (11, 29, 47)
    assert CHECKPOINTS == (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)
    assert configuration.raw["collector"] == "TRANSITIONAL"
    assert configuration.raw["gradient_updates_per_policy"] == 30_000
    with pytest.raises(PermissionError, match="MAIN_TRAINING_NOT_AUTHORIZED"):
        require_main_training_authorized(configuration)
    with pytest.raises(PermissionError, match="VALIDATION.*SEALED"):
        require_validation_authorized(configuration)


def test_exact_reference_median_and_scalar_formula():
    scalar = frozen_scalarization()
    assert scalar.lambda_speed == np.median(
        [1.737065439763169, 1.4009421299572349, 1.3904230583041928]
    )
    assert scalar.lambda_deadline == np.median(
        [1.0015569080594906, 1.5752363435900516, 2.3950774369414662]
    )
    assert scalar.normalizer == pytest.approx(
        1 + scalar.lambda_speed + scalar.lambda_deadline
    )
    expected = (
        0.2514977651664276 * 0.4 - 0.35233381481173953 * 0.2 - 0.39616842002183283 * 0.1
    )
    assert scalar_reward(0.4, 0.2, 0.1) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        (
            dict(
                signed_step_energy_kwh=0.001,
                velocity_m_s=10.0,
                speed_limit_m_s=12.0,
                elapsed_time_s=100.0,
                optimistic_remaining_time_s=30.0,
            ),
            (-0.004, 0.0, 0.0),
        ),
        (
            dict(
                signed_step_energy_kwh=-0.0005,
                velocity_m_s=10.0,
                speed_limit_m_s=12.0,
                elapsed_time_s=120.0,
                optimistic_remaining_time_s=30.0,
            ),
            (0.002, 0.0, 0.1 * 10.0 / 140.0),
        ),
        (
            dict(
                signed_step_energy_kwh=0.0,
                velocity_m_s=15.0,
                speed_limit_m_s=12.0,
                elapsed_time_s=100.0,
                optimistic_remaining_time_s=30.0,
            ),
            (0.0, 0.3, 0.0),
        ),
        (
            dict(
                signed_step_energy_kwh=0.002,
                velocity_m_s=14.0,
                speed_limit_m_s=12.0,
                elapsed_time_s=130.0,
                optimistic_remaining_time_s=20.0,
            ),
            (-0.008, 0.2, 0.1 * 10.0 / 140.0),
        ),
    ),
)
def test_m_information_parity_fixture(values, expected):
    signals = canonical_transition_signals(**values)
    constrained = (
        objective_reward(values["signed_step_energy_kwh"], 0.25),
        speed_cost(max(0.0, values["velocity_m_s"] - values["speed_limit_m_s"]), 0.1),
        deadline_deficit_cost(
            deficit_s=max(
                0.0,
                values["elapsed_time_s"]
                + values["optimistic_remaining_time_s"]
                - 140.0,
            ),
            dt_s=0.1,
            normalization_s=140.0,
        ),
    )
    assert (
        signals.objective,
        signals.speed_cost,
        signals.deadline_cost,
    ) == pytest.approx(constrained)
    assert constrained == pytest.approx(expected)


def test_scalar_reward_has_no_hidden_or_omitted_term():
    baseline = scalar_reward(0.25, 0.0, 0.0)
    speed = scalar_reward(0.25, 0.1, 0.0)
    deadline = scalar_reward(0.25, 0.0, 0.1)
    scalar = frozen_scalarization()
    assert baseline - speed == pytest.approx(scalar.w_speed * 0.1)
    assert baseline - deadline == pytest.approx(scalar.w_deadline * 0.1)
    # Terminal/truncation status is deliberately absent from the scalar formula.
    assert scalar_reward(0.25, 0.1, 0.2) == scalar.reward(0.25, 0.1, 0.2)


class _SyntheticTransition(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def __init__(self):
        self.track = SimpleNamespace(
            positions_m=np.asarray([0.0]), limits_m_s=np.asarray([10.0])
        )
        self.config = SimpleNamespace(track_length_m=100.0)

    def reset(self, *, seed=None, options=None):
        del seed, options
        return np.zeros(8, dtype=np.float32), {"elapsed_time_s": 0.0}

    def step(self, action):
        del action
        return (
            np.ones(8, dtype=np.float32),
            999.0,
            False,
            True,
            {
                "elapsed_time_s": 0.1,
                "position_m": 0.0,
                "step_energy_kwh": 0.001,
                "speed_excess_m_s": 2.0,
                "episode_metrics": {
                    "completed": False,
                    "travel_time_s": 0.1,
                    "energy_kwh": 0.001,
                    "speed_violation_count": 1,
                    "max_speed_violation_m_s": 2.0,
                    "integrated_speed_violation_m": 0.2,
                },
            },
        )


def test_scalar_wrapper_preserves_observation_and_truncation_without_terminal_bonus():
    environment = SameInformationScalarWrapper(
        _SyntheticTransition(),
        task=load_configuration().constrained_v2.task,
        energy_scale_kwh=0.25,
        deadline_normalization_s=140.0,
    )
    environment.reset()
    observation, reward, terminated, truncated, info = environment.step([0.0])
    np.testing.assert_array_equal(observation, np.ones(8, dtype=np.float32))
    assert (terminated, truncated) == (False, True)
    assert info["same_information_signals"] == pytest.approx(
        {"r_E": -0.004, "c_speed": 0.2, "c_deadline": 0.0}
    )
    assert reward == scalar_reward(-0.004, 0.2, 0.0)
    assert info["scalar_reward_has_hidden_terms"] is False


def test_exact_n_step_two_return_and_terminal_handling():
    assert n_step_scalar_return(
        (1.0, 2.0), bootstrap=3.0, gamma=0.99, terminated=False
    ) == pytest.approx(1.0 + 0.99 * 2.0 + 0.99**2 * 3.0)
    assert n_step_scalar_return(
        (1.0, 2.0), bootstrap=3.0, gamma=0.99, terminated=True
    ) == pytest.approx(1.0 + 0.99 * 2.0)


def test_scalar_actor_and_reward_critic_pair_exactly_without_cost_critics():
    result = verify_initialization(load_configuration())
    assert result["passed"] is True
    for row in result["by_seed"].values():
        assert row["byte_identical"] is True
        assert row["scalar_invariants"] == {
            "critic_systems": 1,
            "double_q_heads": 2,
            "cost_critics": 0,
            "use_lagrangian": False,
            "pid_optimizers": 0,
            "automatic_alpha": True,
            "target_entropy": -1.0,
            "n_step": 2,
        }


def test_new_splits_are_frozen_route_clean_and_paper_fail_closed():
    ledger = build_split_ledger()
    assert ledger["route_clean_verified"] is True
    assert diagnostic_seeds() == (
        9_400_000,
        9_400_001,
        9_400_002,
        9_400_003,
        9_400_004,
        9_400_006,
        9_400_007,
        9_400_008,
        9_400_009,
        9_400_011,
        9_400_012,
        9_400_013,
        9_400_014,
        9_400_015,
        9_400_016,
        9_400_017,
        9_400_018,
        9_400_019,
    )
    assert validation_seeds() == (
        9_500_001,
        9_500_002,
        9_500_003,
        9_500_005,
        9_500_006,
        9_500_007,
        9_500_008,
        9_500_011,
        9_500_012,
        9_500_014,
        9_500_015,
        9_500_016,
        9_500_017,
        9_500_018,
        9_500_019,
        9_500_020,
        9_500_021,
        9_500_022,
    )
    assert ledger["paper"]["content_fingerprints_constructed"] == 0
    with pytest.raises(PermissionError, match="PAPER_TEST_SEALED"):
        evaluate_paper()


def test_scalar_recovery_round_trip_rejects_pid_state(tmp_path: Path):
    configuration = load_configuration()
    payload = {
        "checkpoint_schema_version": 1,
        "condition": "S11_sameinfo_scalar_seed11",
        "attempt_id": "fixture",
        "configuration_sha256": configuration.configuration_sha256,
        "source_hashes": {},
        "runtime_versions": {},
        "policy": {},
        "policy_optimizers": {},
        "replay_buffer": [],
        "environment_state": {},
        "collector_state": {},
        "counters": {"real_transitions": 10_000, "rl_gradient_updates": 1_000},
        "diagnostic_checkpoints_completed": [],
        "rng_states": capture_rng_states(),
        "walltime_state": {"completed_segments_s": 1.0},
    }
    path = tmp_path / "checkpoint.pt"
    metadata = write_verified(path, payload)
    restored = load_verified(
        path,
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256=configuration.configuration_sha256,
    )
    assert restored["counters"] == payload["counters"]
    with pytest.raises(ValueError, match="PID/Lagrange"):
        validate_payload({**payload, "lagrangian_states": []})


@pytest.mark.parametrize(
    ("deltas", "pooled", "expected"),
    (
        ((2, 2, 2), 6, "CONSTRAINED_ROBUST_ADVANTAGE"),
        ((-2, -2, -2), -6, "SCALAR_ROBUST_ADVANTAGE"),
        ((1, 0, -1), 0, "NO_MATERIAL_DIFFERENCE"),
        ((3, -3, 0), 0, "SEED_HETEROGENEOUS"),
    ),
)
def test_frozen_interpretation_gates(deltas, pooled, expected):
    assert classify(deltas, pooled) == expected
