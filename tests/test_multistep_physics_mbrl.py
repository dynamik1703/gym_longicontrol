"""Focused semantic and operational tests for exact-physics value expansion."""

from __future__ import annotations

import json
import plistlib
from pathlib import Path
from types import MethodType

import numpy as np
import pytest

from benchmarks.model_based_rl.adapter import model_step
from benchmarks.model_based_rl.model_disabled_parity import _fill_buffer
from benchmarks.model_based_rl.physics_model import PhysicsDynamicsModel
from benchmarks.multistep_physics_mbrl.adapter import (
    ModelState,
    RealEpisodePIDGate,
    RealPhysicsReplay,
    TrackContext,
)
from benchmarks.multistep_physics_mbrl.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_states,
    load_verified,
    restore_rng_states,
    write_verified_checkpoint,
)
from benchmarks.multistep_physics_mbrl.config import (
    CONDITIONS,
    MODEL_EXPANSION_STEPS,
    TRAINING_SEEDS,
    load_configuration,
)
from benchmarks.multistep_physics_mbrl.diagnostics import (
    TargetAccumulator,
    late_collapse,
    numerical_instability,
)
from benchmarks.multistep_physics_mbrl.evaluation import (
    evaluate_paper_test,
    evaluate_validation,
)
from benchmarks.multistep_physics_mbrl.execution import (
    initial_manifest,
    prepare_launchd_job,
    run_policy,
)
from benchmarks.multistep_physics_mbrl.physics_rollout import (
    batched_step,
    braking_horizon,
    pack_states,
    scalar_step,
)
from benchmarks.multistep_physics_mbrl.value_expansion import (
    parameter_snapshot,
    real_replay_update,
    recursive_soft_target,
    snapshots_equal,
)
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.track import Track
from gym_longicontrol.domain.vehicle import VehicleModel


def context(
    positions=(0.0, 100.0, 300.0),
    limits=(100.0 / 3.6, 60.0 / 3.6, 20.0 / 3.6),
) -> TrackContext:
    return TrackContext(
        Track(np.asarray(positions), np.asarray(limits)),
        sensor_range_m=150.0,
        track_length_m=1000.0,
        energy_factor=1.0,
    )


def test_configuration_freezes_clean_matched_protocol():
    configuration = load_configuration()
    assert tuple(configuration.raw["conditions"]) == CONDITIONS
    assert tuple(configuration.raw["training_seeds"]) == TRAINING_SEEDS
    assert configuration.model_expansion_steps == MODEL_EXPANSION_STEPS == 60
    assert configuration.raw["target"]["real_prefix_steps"] == 2
    assert configuration.raw["target"]["synthetic_replay"] is False
    assert configuration.raw["total_real_transitions"] == 1_800_000
    assert configuration.raw["preparation"] == {
        "maximum_simulator_transitions": 5000,
        "simulator_transitions_consumed": 1024,
        "allowed_track_seeds": [0],
    }
    assert configuration.raw["authorization"] == {
        "main_training_authorized": False,
        "main_training_enabled": False,
        "validation_authorized": False,
        "paper_test_authorized": False,
        "automatic_fresh_restart_authorized": False,
    }


def test_physical_horizon_is_exact_and_not_rounded_conventionally():
    direct = braking_horizon(100.0, 60.0)
    assert direct.steps == 60
    assert direct.time_s == 6.0
    assert direct.distance_m == pytest.approx(135.0764556315701, abs=1e-12)
    assert direct.distance_m < 150.0
    two_transition = braking_horizon(100.0, 20.0)
    assert two_transition.steps == 100
    assert two_transition.time_s == 10.0
    assert two_transition.distance_m == pytest.approx(179.8249471715907, abs=1e-12)


@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_scalar_vectorized_physics_parity(batch_size):
    vehicle = VehicleModel()
    config = SimulationConfig()
    rng = np.random.default_rng(811)
    contexts = [
        context(
            positions=(0.0, 120.0, 500.0),
            limits=(80.0 / 3.6, 40.0 / 3.6, 70.0 / 3.6),
        )
        for _ in range(batch_size)
    ]
    states = [
        ModelState(
            VehicleState(
                position_m=float(rng.uniform(0.0, 800.0)),
                velocity_m_s=float(rng.uniform(0.0, 30.0)),
                acceleration_m_s2=float(rng.uniform(-3.0, 3.0)),
                elapsed_time_s=float(rng.uniform(0.0, 130.0)),
                total_energy_kwh=float(rng.uniform(-0.1, 0.5)),
            ),
            episode_step=int(rng.integers(0, 1798)),
        )
        for _ in range(batch_size)
    ]
    actions = rng.uniform(-1.0, 1.0, size=batch_size)
    vector = batched_step(
        pack_states(states),
        actions,
        contexts=contexts,
        vehicle=vehicle,
        config=config,
    )
    for index, (state, action, track_context) in enumerate(
        zip(states, actions, contexts)
    ):
        scalar = scalar_step(
            state,
            float(action),
            context=track_context,
            vehicle=vehicle,
            config=config,
            source_real_transition_id=index,
        )
        np.testing.assert_allclose(
            vector.next_observation[index], scalar.next_observation, atol=1e-12
        )
        assert vector.objective[index] == pytest.approx(scalar.objective, abs=1e-12)
        np.testing.assert_allclose(vector.costs[index], scalar.costs, atol=1e-12)
        assert bool(vector.terminated[index]) is scalar.terminated
        assert bool(vector.truncated[index]) is scalar.truncated
        actual = vector.state.to_model_states()[index]
        assert actual.vehicle.position_m == pytest.approx(
            scalar.next_state.vehicle.position_m, abs=1e-12
        )
        assert actual.vehicle.velocity_m_s == pytest.approx(
            scalar.next_state.vehicle.velocity_m_s, abs=1e-12
        )


@pytest.mark.parametrize("model_steps", [1, 2])
def test_short_horizon_soft_targets_match_manual_equations(model_steps):
    gamma = 0.99
    alpha = 0.2
    real = np.asarray([[1.0, 0.1, 0.2], [2.0, 0.3, 0.4]])
    model = np.asarray(
        [[3.0, 0.5, 0.6], [4.0, 0.7, 0.8]][:model_steps]
    )
    log_prob = np.asarray([-0.7, -0.3][:model_steps])
    q = np.asarray([10.0, 11.0, 12.0])
    result, real_part, model_part, bootstrap = recursive_soft_target(
        real_metrics=real,
        model_metrics=model,
        model_log_probabilities=log_prob,
        bootstrap_q=q,
        bootstrap_log_probability=-0.25,
        alpha=alpha,
        gamma=gamma,
    )
    expected_real = real[0] + gamma * real[1]
    expected_model = sum(
        gamma ** (2 + index) * (row - alpha * log_prob[index])
        for index, row in enumerate(model)
    )
    expected_bootstrap = gamma ** (2 + model_steps) * (q - alpha * -0.25)
    np.testing.assert_allclose(real_part, expected_real, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(model_part, expected_model, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(bootstrap, expected_bootstrap, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        result, expected_real + expected_model + expected_bootstrap, atol=1e-15
    )
    assert np.all(model_part[1:] > 0.0), "cost critics retain FSRL entropy"


def test_future_speed_cost_appears_only_when_rollout_reaches_limit():
    vehicle = VehicleModel()
    config = SimulationConfig()
    track_context = context(positions=(0.0, 140.0), limits=(100 / 3.6, 60 / 3.6))
    state = ModelState(VehicleState(velocity_m_s=100.0 / 3.6))

    def speed_return(action, horizon):
        batch = pack_states([state])
        total = 0.0
        for step in range(horizon):
            transition = batched_step(
                batch,
                np.asarray([action]),
                contexts=[track_context],
                vehicle=vehicle,
                config=config,
            )
            total += 0.99**step * transition.costs[0, 0]
            batch = transition.state
        return total

    assert speed_return(0.0, 30) == 0.0
    assert speed_return(0.0, 40) == 0.0
    assert speed_return(0.0, 55) > 0.0
    assert speed_return(1.0, 55) > speed_return(0.0, 55)
    assert speed_return(-1.0, 60) == pytest.approx(0.0, abs=1e-12)


def test_constant_limit_completion_deadline_and_existing_violation_cases():
    vehicle = VehicleModel()
    config = SimulationConfig()
    constant = context(positions=(0.0,), limits=(20.0,))
    cases = [
        ModelState(VehicleState(position_m=999.9, velocity_m_s=2.0)),
        ModelState(
            VehicleState(position_m=500.0, velocity_m_s=10.0, elapsed_time_s=130.0)
        ),
        ModelState(VehicleState(position_m=10.0, velocity_m_s=25.0)),
    ]
    result = batched_step(
        pack_states(cases),
        np.asarray([0.0, 0.0, 0.0]),
        contexts=[constant] * 3,
        vehicle=vehicle,
        config=config,
    )
    assert result.terminated[0]
    assert result.costs[1, 1] > 0.0
    assert result.costs[2, 0] > 0.0


class FakePIDPolicy:
    def __init__(self):
        self.calls = 0

    def pre_update_fn(self, **_kwargs):
        self.calls += 1


def test_pid_is_hard_gated_to_completed_real_episodes():
    gate = RealEpisodePIDGate()
    policy = FakePIDPolicy()
    gate.update(
        policy,
        source="real_completed_episode",
        stats={"cost": np.asarray([0.1, 0.2])},
    )
    assert gate.real_episode_updates == policy.calls == 1
    for source in ("model", "model_completed_episode", "synthetic"):
        with pytest.raises(ValueError, match="never update PID"):
            gate.update(policy, source=source, stats={"cost": np.zeros(2)})
    assert policy.calls == 1


def test_m0_adapter_is_bit_identical_to_direct_frozen_update():
    torch = pytest.importorskip("torch")
    pytest.importorskip("fsrl")
    from tianshou.data import ReplayBuffer

    from benchmarks.constrained_rl_v2.adapter import build_agent
    from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
    from benchmarks.scalar_sac.experiment import _base_environment

    configuration = load_configuration()
    environments = [
        DenseDeadlineTaskWrapper(
            _base_environment(configuration.v2),
            task=configuration.v2.task,
            energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
            deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
        )
        for _ in range(2)
    ]
    try:
        policies = [
            build_agent(
                configuration.v2,
                environment,
                training_seed=117,
                device="cpu",
                threads=1,
            )[0].policy
            for environment in environments
        ]
        base = environments[0].unwrapped
        track_context = TrackContext(
            base.track_generator(np.random.default_rng(2000)),
            base.config.sensor_range_m,
            base.config.track_length_m,
            base.energy_factor,
        )
        physics = PhysicsDynamicsModel(base.vehicle, base.config)
        rows = []
        for index in range(512):
            state = ModelState(
                VehicleState(
                    position_m=float(index % 800),
                    velocity_m_s=float(1 + index % 30),
                    acceleration_m_s2=float(index % 5 - 2),
                    elapsed_time_s=float(index % 1200) * 0.1,
                ),
                episode_step=index % 1200,
            )
            rows.append(
                model_step(
                    physics,
                    state,
                    float(-1 + 2 * (index % 101) / 100),
                    context=track_context,
                    vehicle=base.vehicle,
                    config=base.config,
                    energy_scale_kwh=0.25,
                    deadline_s=140.0,
                    max_episode_steps=1800,
                    source_real_transition_id=index,
                    rng=np.random.default_rng(index),
                )
            )
        buffers = [ReplayBuffer(1000), ReplayBuffer(1000)]
        for buffer in buffers:
            _fill_buffer(buffer, rows)
        learned_batches = []
        for policy in policies:
            original_learn = policy.learn

            def capture_learn(self, batch, _learn=original_learn, **kwargs):
                learned_batches.append(
                    {
                        "obs": np.asarray(batch.obs).copy(),
                        "rets": batch.rets.detach().cpu().clone(),
                    }
                )
                return _learn(batch, **kwargs)

            policy.learn = MethodType(capture_learn, policy)
        np.random.seed(991)
        torch.manual_seed(991)
        policies[0].update(256, buffers[0])
        np.random.seed(991)
        torch.manual_seed(991)
        assert (
            real_replay_update(
                policies[1],
                buffers[1],
                RealPhysicsReplay(),
                model_steps=0,
                rollout_rng=np.random.default_rng(42),
                vehicle=base.vehicle,
                simulation_config=base.config,
            )
            is None
        )
        assert len(learned_batches) == 2
        np.testing.assert_array_equal(
            learned_batches[0]["obs"], learned_batches[1]["obs"]
        )
        assert learned_batches[0]["rets"].shape[-1] == 3
        assert torch.equal(learned_batches[0]["rets"], learned_batches[1]["rets"])
        assert snapshots_equal(
            parameter_snapshot(policies[0]), parameter_snapshot(policies[1])
        )
    finally:
        for environment in environments:
            environment.close()


def _checkpoint_payload(rng):
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "condition": "physics_mve",
        "training_seed": 11,
        "attempt_id": "fixture-attempt",
        "configuration_sha256": "a" * 64,
        "source_hashes": {"fixture": "b" * 64},
        "runtime_versions": {"fixture": "1"},
        "policy": {},
        "policy_optimizers": {},
        "lagrangian_states": [],
        "real_replay": [],
        "physics_sidecar": RealPhysicsReplay(),
        "environment_state": {},
        "collector_state": {},
        "track_stream_state": {},
        "counters": {
            "real_transitions": 10_000,
            "synthetic_transitions": 0,
            "model_target_transitions": 12,
            "rl_gradient_updates": 1000,
        },
        "diagnostics": [],
        "target_accumulator": TargetAccumulator(),
        "anticipation_probes": [],
        "rng_states": capture_rng_states(rng),
        "pid_real_episode_updates": 3,
        "previous_completed_episodes": 3,
        "development_checkpoints_completed": [],
    }


def test_checkpoint_roundtrip_restores_dedicated_rollout_rng(tmp_path):
    rng = np.random.default_rng(17)
    payload = _checkpoint_payload(rng)
    path = tmp_path / "checkpoint.pt"
    metadata = write_verified_checkpoint(path, payload)
    expected = rng.normal(size=5)
    restored = load_verified(
        path,
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256="a" * 64,
    )
    resumed_rng = np.random.default_rng(999)
    restore_rng_states(restored["rng_states"], resumed_rng)
    np.testing.assert_array_equal(resumed_rng.normal(size=5), expected)


def test_manifest_launch_preparation_and_split_gates(tmp_path):
    manifest = initial_manifest()
    assert len(manifest["runs"]) == 6
    assert manifest["validation"]["state"] == "SEALED"
    assert manifest["paper_test"]["state"] == "SEALED"
    plist = prepare_launchd_job(
        repository=tmp_path,
        output_root=tmp_path / "runs",
        python="/fixture/python",
    )
    with plist.open("rb") as stream:
        job = plistlib.load(stream)
    assert job["RunAtLoad"] is True
    launcher = json.loads(
        (tmp_path / "runs" / "launcher.json").read_text(encoding="utf-8")
    )
    assert launcher["state"] == "PREPARED_NOT_LAUNCHED"
    assert launcher["main_training_authorized"] is False
    with pytest.raises(PermissionError, match="not authorized"):
        run_policy(condition="physics_mve", training_seed=11, output_root=tmp_path)
    with pytest.raises(PermissionError, match="remain sealed"):
        evaluate_paper_test()
    with pytest.raises(PermissionError, match="remains sealed"):
        evaluate_validation(object(), load_configuration(), manifest)


def test_preregistered_stability_diagnostics():
    collapse = late_collapse([0, 1, 3, 2, 9, 0])
    assert collapse["late_collapse"] is True
    assert collapse["peak_to_final_drop"] == 9
    assert not numerical_instability(
        lambda_speed=1.0,
        lambda_deadline=2.0,
        critic_losses=[3.0, 4.0, 5.0],
        target_magnitudes=[10.0, 11.0, 12.0],
    )
    assert numerical_instability(
        lambda_speed=1.0,
        lambda_deadline=2.0,
        critic_losses=[float("nan")],
        target_magnitudes=[0.0],
    )


def test_no_existing_historical_physics_sources_were_changed():
    from benchmarks.model_based_rl.checkpointing import file_sha256

    frozen = json.loads(
        Path("benchmarks/physics_model_mbrl/freeze.json").read_text(encoding="utf-8")
    )
    expected = {**frozen["parent_files"], **frozen["study_files"]}
    actual = {name: file_sha256(name) for name in expected}
    assert actual == expected


def test_new_scientific_and_readiness_hashes_are_frozen():
    from benchmarks.model_based_rl.checkpointing import file_sha256

    frozen = json.loads(
        Path("benchmarks/multistep_physics_mbrl/freeze.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen["frozen_before_main_training"] is True
    assert frozen["parent_commit"] == "8e5ebf6fb241f434781c0148c122a84c1ab4c35b"
    expected = {**frozen["scientific_files"], **frozen["readiness_files"]}
    assert {name: file_sha256(name) for name in expected} == expected
