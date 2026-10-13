import json
from pathlib import Path

import numpy as np

from benchmarks.multistep_physics_mbrl.adapter import RealEpisodePIDGate
from benchmarks.multistep_physics_mbrl.config import load_configuration
from benchmarks.multistep_physics_mbrl.physics_rollout import braking_horizon
from benchmarks.multistep_physics_mbrl.value_expansion import recursive_soft_target


def test_frozen_preparation_contract():
    root = Path("benchmarks/multistep_physics_mbrl")
    status = json.loads((root / "preparation_status.json").read_text())
    configuration = load_configuration()
    assert status["ready_for_main_training"] is True
    assert status["main_training_authorized"] is True
    assert status["main_training_enabled"] is True
    assert configuration.raw["authorization"]["validation_authorized"] is False
    assert configuration.raw["authorization"]["paper_test_authorized"] is False


def test_horizon_and_three_critic_target_contract():
    horizon = braking_horizon(100.0, 60.0)
    assert (horizon.steps, horizon.time_s) == (60, 6.0)
    target, real, model, bootstrap = recursive_soft_target(
        real_metrics=np.ones((2, 3)),
        model_metrics=np.ones((1, 3)),
        model_log_probabilities=np.zeros(1),
        bootstrap_q=np.ones(3),
        bootstrap_log_probability=0.0,
        alpha=1.0,
    )
    assert target.shape == real.shape == model.shape == bootstrap.shape == (3,)


def test_pid_gate_rejects_model_source():
    gate = RealEpisodePIDGate()
    try:
        gate.update(object(), source="model", stats={"cost": np.zeros(2)})
    except ValueError as error:
        assert "never update PID" in str(error)
    else:
        raise AssertionError("Model source crossed the real-episode PID gate")
