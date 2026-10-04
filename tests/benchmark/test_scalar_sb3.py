import json

import pytest

from benchmarks.scalar_sac.evaluation import EpisodeEvaluation, EvaluationSummary
from benchmarks.scalar_sac.v2_config import load_v2_configuration
from benchmarks.scalar_sb3.config import (
    DEFAULT_CONFIG_PATH,
    configuration_sha256,
    load_configuration,
)
from benchmarks.scalar_sb3.results import (
    SB3BenchmarkRunResult,
    load_result,
    save_result,
)
from gym_longicontrol.domain.metrics import EpisodeMetrics


def test_canonical_sb3_protocol_freezes_reward_splits_budgets_and_defaults():
    configuration = load_configuration()
    v2b = load_v2_configuration(
        DEFAULT_CONFIG_PATH.parent.parent / "scalar_sac" / "canonical_v2b.json"
    )

    assert configuration.reward == v2b.reward_candidates[0]
    assert configuration.training_seeds == (11, 29, 47)
    assert configuration.learning_curve_steps == tuple(range(50_000, 300_001, 50_000))
    assert configuration.track_splits.development_calibration == tuple(
        range(2000, 2009)
    )
    assert configuration.track_splits.validation == tuple(range(3000, 3009))
    assert configuration.track_splits.paper_final_test_reserved == tuple(
        range(4000, 4018)
    )
    assert configuration.sac.policy_network == (256, 256)
    assert configuration.sac.ent_coef == "auto"
    assert configuration.ppo.policy_network == (64, 64)
    assert configuration.ppo.n_steps == 2048
    assert len(configuration_sha256(configuration)) == 64


def test_sb3_protocol_rejects_reserved_seed_overlap(tmp_path):
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["track_splits"]["validation"][0] = 4000
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_configuration(path)


def test_sb3_result_roundtrip_preserves_episode_level_metrics(tmp_path):
    configuration = load_configuration()
    metrics = EpisodeMetrics(True, 100.0, 0.2, 0, 0.0, 0.0)
    episodes = (
        EpisodeEvaluation.from_metrics(3000, metrics, configuration.task),
    )
    result = SB3BenchmarkRunResult(
        benchmark_name=configuration.name,
        configuration_sha256=configuration_sha256(configuration),
        environment_id=configuration.environment_id,
        evaluation_split_id="validation-test",
        algorithm="sac",
        library_version="test",
        training_seed=11,
        training_steps=50_000,
        training_wall_time_s=1.5,
        policy_updates=49_900,
        task=configuration.task,
        reward_parameters=configuration.reward,
        episodes=episodes,
        summary=EvaluationSummary.from_episodes(episodes, configuration.task),
        diagnostics={"train/actor_loss": -0.25},
    )
    path = save_result(tmp_path / "result.json", result)
    assert load_result(path) == result


def test_diagnostic_writer_implements_sb3_logger_protocol():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.logger import KVWriter

    from benchmarks.scalar_sb3.experiment import DiagnosticWriter

    writer = DiagnosticWriter()
    assert isinstance(writer, KVWriter)
    writer.write(
        {"train/actor_loss": 0.25, "non_numeric": object()},
        {},
        step=100,
    )
    assert writer.latest == {"train/actor_loss": 0.25}
    assert writer.history == [{"training_steps": 100, "train/actor_loss": 0.25}]


@pytest.mark.parametrize("algorithm", ["sac", "ppo"])
def test_sb3_model_builder_uses_frozen_standard_configuration(algorithm):
    pytest.importorskip("stable_baselines3")
    from benchmarks.scalar_sb3.experiment import _model, _training_environment

    configuration = load_configuration()
    environment = _training_environment(configuration)
    try:
        model = _model(configuration, algorithm, environment, 11, "cpu")
        assert model.seed == 11
        assert model.gamma == pytest.approx(0.99)
        if algorithm == "sac":
            assert model.batch_size == 256
            assert model.tau == pytest.approx(0.005)
            assert model.ent_coef == "auto"
        else:
            assert model.n_steps == 2048
            assert model.batch_size == 64
            assert model.n_epochs == 10
    finally:
        environment.close()
