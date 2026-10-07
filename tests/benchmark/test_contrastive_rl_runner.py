import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("optax")

from benchmarks.contrastive_rl.checkpointing import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
)
from benchmarks.contrastive_rl.config import ReferenceCoreConfig  # noqa: E402
from benchmarks.contrastive_rl.diagnostics import (  # noqa: E402
    DiagnosticsAccumulator,
    learning_batch_diagnostics,
    sampled_batch_diagnostics,
)
from benchmarks.contrastive_rl.learner import (  # noqa: E402
    ContrastiveBatch,
    ReferenceLearner,
)
from benchmarks.contrastive_rl.replay import (  # noqa: E402
    SampledFutureBatch,
    TransitionReplay,
)
from benchmarks.contrastive_rl.runner import (  # noqa: E402
    PolicyRunner,
    _checkpoint_policy,
    _observed_future_is_canonical,
)


class TinySerializableEnvironment:
    def __init__(self):
        self.position = 0

    def step(self, action):
        self.position += 1
        return self.position, float(np.asarray(action).item())

    def close(self):
        return None


def _small_learner():
    configuration = ReferenceCoreConfig(
        state_dim=3,
        goal_dim=3,
        action_dim=1,
        depth=4,
        width=16,
        embedding_dim=8,
    )
    return ReferenceLearner.create(configuration, seed=7)


def _batch():
    return ContrastiveBatch(
        states=jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 12,
        actions=jnp.linspace(-0.5, 0.5, 4).reshape(4, 1),
        critic_goals=jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 12,
        actor_goals=jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 12,
        historical_reward=jnp.arange(4, dtype=jnp.float32),
    )


def _runtime():
    future_rng = np.random.default_rng(88)
    track_rng = np.random.default_rng(99)
    return {
        "transition_count": 12,
        "update_cycle_count": 3,
        "episode_id": 2,
        "episode_step": 4,
        "current_episode_ended": False,
        "observation": np.arange(8, dtype=np.float64),
        "current_outcome": np.array([4.0, 3.0, 1.2, 0.0]),
        "actor_rng_key": np.asarray(jax.random.PRNGKey(1)),
        "update_rng_key": np.asarray(jax.random.PRNGKey(2)),
        "future_rng_state": future_rng.bit_generator.state,
        "track_rng_state": track_rng.bit_generator.state,
        "track_seeds_used": [123, 456],
        "diagnostics": {"rows": []},
        "training_outcomes": [],
        "development_completed": [],
        "first_canonical_positive_available_transition": None,
    }


def _replay():
    replay = TransitionReplay(8, state_dim=3)
    for step in range(4):
        replay.append(
            state=np.full(3, step, dtype=np.float64),
            action=np.array([step / 10]),
            source_outcome=np.array([step, max(0, step - 1), step, 0.0]),
            outcome=np.array([step + 1, step, step + 1, 0.0]),
            historical_reward=float(step * 100),
            terminated=step == 3,
            truncated=False,
            episode_id=0,
            episode_step=step,
        )
    return replay


def _assert_trees_equal(first, second):
    for left, right in zip(
        jax.tree_util.tree_leaves(first),
        jax.tree_util.tree_leaves(second),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)


def test_atomic_checkpoint_reproduces_next_environment_rng_and_update(tmp_path):
    learner, state = _small_learner()
    environment = TinySerializableEnvironment()
    environment.step(np.array([0.0]))
    runtime = _runtime()
    configuration_hash = "c" * 64
    source_hashes = {"frozen.py": "d" * 64}
    execution_hashes = {"runner.py": "e" * 64}
    path = tmp_path / "exact.ckpt"
    digest = save_checkpoint(
        path,
        learner_state=state,
        replay=_replay(),
        environment=environment,
        runtime=runtime,
        configuration_sha256=configuration_hash,
        scientific_source_sha256=source_hashes,
        execution_source_sha256=execution_hashes,
    )
    assert len(digest) == 64
    assert not tuple(tmp_path.glob("*.tmp"))
    loaded = load_checkpoint(
        path,
        learner_state_template=state,
        expected_configuration_sha256=configuration_hash,
        expected_scientific_source_sha256=source_hashes,
        expected_execution_source_sha256=execution_hashes,
    )
    assert environment.step(np.array([0.25])) == loaded.environment.step(
        np.array([0.25])
    )
    original_rng = np.random.default_rng()
    restored_rng = np.random.default_rng()
    original_rng.bit_generator.state = runtime["future_rng_state"]
    restored_rng.bit_generator.state = loaded.runtime["future_rng_state"]
    np.testing.assert_array_equal(original_rng.random(20), restored_rng.random(20))
    np.testing.assert_array_equal(
        runtime["actor_rng_key"], loaded.runtime["actor_rng_key"]
    )
    np.testing.assert_array_equal(
        runtime["update_rng_key"], loaded.runtime["update_rng_key"]
    )
    expected_action = learner.deterministic_action(
        state.actor.params, _batch().states, _batch().actor_goals
    )
    restored_action = learner.deterministic_action(
        loaded.learner_state.actor.params,
        _batch().states,
        _batch().actor_goals,
    )
    np.testing.assert_array_equal(expected_action, restored_action)
    key = jax.random.PRNGKey(123)
    expected_state, expected_metrics = learner.update(state, _batch(), key)
    actual_state, actual_metrics = learner.update(loaded.learner_state, _batch(), key)
    _assert_trees_equal(expected_state, actual_state)
    _assert_trees_equal(expected_metrics, actual_metrics)


def test_checkpoint_rejects_configuration_source_and_schema_mismatches(tmp_path):
    _learner, state = _small_learner()
    path = tmp_path / "checkpoint.bin"
    save_checkpoint(
        path,
        learner_state=state,
        replay=_replay(),
        environment=TinySerializableEnvironment(),
        runtime=_runtime(),
        configuration_sha256="a" * 64,
        scientific_source_sha256={"science": "b" * 64},
        execution_source_sha256={"runner": "d" * 64},
    )
    with pytest.raises(RuntimeError, match="configuration"):
        load_checkpoint(
            path,
            learner_state_template=state,
            expected_configuration_sha256="c" * 64,
            expected_scientific_source_sha256={"science": "b" * 64},
            expected_execution_source_sha256={"runner": "d" * 64},
        )
    with pytest.raises(RuntimeError, match="source hashes"):
        load_checkpoint(
            path,
            learner_state_template=state,
            expected_configuration_sha256="a" * 64,
            expected_scientific_source_sha256={"science": "z" * 64},
            expected_execution_source_sha256={"runner": "d" * 64},
        )
    with pytest.raises(RuntimeError, match="execution source hashes"):
        load_checkpoint(
            path,
            learner_state_template=state,
            expected_configuration_sha256="a" * 64,
            expected_scientific_source_sha256={"science": "b" * 64},
            expected_execution_source_sha256={"runner": "z" * 64},
        )
    payload = pickle.loads(path.read_bytes())
    payload["checkpoint_schema_version"] += 1
    path.write_bytes(pickle.dumps(payload, protocol=5))
    with pytest.raises(RuntimeError, match="schema version"):
        load_checkpoint(
            path,
            learner_state_template=state,
            expected_configuration_sha256="a" * 64,
            expected_scientific_source_sha256={"science": "b" * 64},
            expected_execution_source_sha256={"runner": "d" * 64},
        )


def test_development_milestone_directory_is_published_atomically(tmp_path, monkeypatch):
    class Diagnostics:
        rows = [{"update_cycle": 1}]

        @staticmethod
        def summary():
            return {"learning_batch_count": 1}

    run_directory = tmp_path / "study" / "depth-4" / "seed-11"
    run_directory.mkdir(parents=True)
    runner = SimpleNamespace(
        transition_count=50_000,
        update_cycle_count=1_000,
        development_completed=[],
        first_canonical_positive_available_transition=None,
        diagnostics=Diagnostics(),
        development_evaluation=lambda: {
            "summary": {"evaluation_simulator_transitions": 123}
        },
        save=lambda path: (path.write_bytes(b"checkpoint"), "a" * 64)[1],
    )
    item = {"development_checkpoints": [], "attempts": [{}]}
    manifest = {
        "actual_resources_across_attempts": {
            "native_simulator_transitions": 0,
            "complete_update_cycles": 0,
            "development_simulator_transitions": 0,
        },
        "policies": {"only": item},
    }
    monkeypatch.setattr(
        "benchmarks.contrastive_rl.runner.update_manifest",
        lambda *_args, **_kwargs: None,
    )
    _checkpoint_policy(
        runner,
        run_directory=run_directory,
        item=item,
        study_root=tmp_path / "study",
        manifest=manifest,
    )
    final = run_directory / "step-050000"
    assert final.is_dir()
    assert (final / "exact-resume.ckpt").read_bytes() == b"checkpoint"
    assert (final / "development-result.json").is_file()
    assert (final / "diagnostics.json").is_file()
    assert not tuple(run_directory.glob(".step-050000.pending-*"))
    assert runner.development_completed == [50_000]
    assert item["development_checkpoints"][0]["checkpoint_sha256"] == "a" * 64


def test_failed_milestone_checkpoint_never_publishes_final_directory(
    tmp_path, monkeypatch
):
    run_directory = tmp_path / "study" / "depth-4" / "seed-11"
    run_directory.mkdir(parents=True)

    def fail_save(_path):
        raise OSError("synthetic write failure")

    runner = SimpleNamespace(
        transition_count=50_000,
        update_cycle_count=1_000,
        development_completed=[],
        development_evaluation=lambda: {
            "summary": {"evaluation_simulator_transitions": 123}
        },
        save=fail_save,
    )
    item = {"development_checkpoints": [], "attempts": [{}]}
    manifest = {
        "actual_resources_across_attempts": {
            "native_simulator_transitions": 0,
            "complete_update_cycles": 0,
            "development_simulator_transitions": 0,
        },
        "policies": {"only": item},
    }
    monkeypatch.setattr(
        "benchmarks.contrastive_rl.runner.update_manifest",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(OSError, match="synthetic write failure"):
        _checkpoint_policy(
            runner,
            run_directory=run_directory,
            item=item,
            study_root=tmp_path / "study",
            manifest=manifest,
        )
    assert not (run_directory / "step-050000").exists()
    assert len(tuple(run_directory.glob(".step-050000.pending-*"))) == 1
    assert runner.development_completed == []
    assert item["development_checkpoints"] == []


def _sampled(goals):
    size = len(goals)
    return SampledFutureBatch(
        states=np.zeros((size, 3)),
        actions=np.zeros((size, 1)),
        source_outcomes=np.column_stack(
            (np.arange(size), np.zeros(size), np.arange(size), np.zeros(size))
        ),
        future_outcomes=np.zeros((size, 4)),
        future_terminated=np.array([False] * (size - 1) + [True]),
        historical_rewards=np.arange(size) * 1000.0,
        source_indices=np.arange(size),
        future_indices=np.arange(size) + 1,
        lags=np.arange(size) + 1,
        episode_ids=np.zeros(size, dtype=np.int64),
    )


def test_exact_duplicate_and_canonical_support_diagnostics_do_not_use_rng():
    goals = np.ones((4, 3), dtype=np.float64)
    sampled = _sampled(goals)
    rng = np.random.default_rng(44)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    result = sampled_batch_diagnostics(sampled, goals, dt_s=0.1)
    after = json.dumps(rng.bit_generator.state, sort_keys=True)
    assert before == after
    support = result["exact_projected_goal_support"]
    assert support["unique_goal_count"] == 1
    assert support["duplicate_column_count"] == 3
    assert support["duplicate_column_fraction"] == pytest.approx(0.75)
    assert support["pair_collision_rate"] == 1.0
    assert support["all_identical"] is True
    assert result["canonical_positive_count"] == 4
    assert result["diagnostic_binning"] is None


def test_diagnostics_accumulator_tracks_actual_canonical_batches_and_sources():
    accumulator = DiagnosticsAccumulator()
    sampled = _sampled(np.ones((3, 3)))
    accumulator.record(
        transition_count=10_040,
        update_cycle=1,
        sampled=sampled,
        projected_goals=np.ones((3, 3)),
        dt_s=0.1,
        learning={"optimization": {"nonfinite_value_count": 0}},
    )
    summary = accumulator.summary()
    assert summary["learning_batch_count"] == 1
    assert summary["unique_source_transition_coverage"] == 3
    assert summary["first_sampled_canonical_positive_transition"] == 10_040
    assert summary["canonical_positive_batch_fraction"] == 1.0
    restored = DiagnosticsAccumulator.from_state(accumulator.to_state())
    assert restored.summary() == summary


def test_learning_diagnostics_are_finite_and_separate_encoder_gradients():
    learner, state = _small_learner()
    result = learning_batch_diagnostics(
        learner, state, _batch(), jax.random.PRNGKey(321)
    )
    optimization = result["optimization"]
    assert optimization["nonfinite_value_count"] == 0
    assert optimization["actor_gradient_norm"] > 0.0
    assert optimization["critic_gradient_norm"] > 0.0
    assert optimization["state_action_encoder_gradient_norm"] > 0.0
    assert optimization["goal_encoder_gradient_norm"] > 0.0
    assert result["contrastive"]["positive_scores"]["mean"] is not None
    assert (
        result["canonical_query"]["actual_action_canonical_goal_score"]["mean"]
        is not None
    )


def test_collection_action_is_conditioned_on_exact_canonical_command():
    captured = {}

    class Actor:
        def apply(self, params, state_goal):
            del params
            captured["state_goal"] = np.asarray(state_goal)
            return jnp.zeros((1, 1)), jnp.zeros((1, 1))

    runner = PolicyRunner.__new__(PolicyRunner)
    runner.learner = SimpleNamespace(actor=Actor())
    runner.learner_state = SimpleNamespace(actor=SimpleNamespace(params={}))
    runner.actor_rng_key = jax.random.PRNGKey(7)
    action = runner._sample_action(np.arange(12, dtype=np.float64))
    np.testing.assert_array_equal(captured["state_goal"][0, -3:], [1.0, 1.0, 1.0])
    assert action.shape == (1,)
    assert np.isfinite(action).all()


def test_canonical_future_availability_uses_exact_frozen_projection():
    assert _observed_future_is_canonical(
        np.array([1000.0, 999.0, 140.0, 0.0]), terminated=True
    )
    assert not _observed_future_is_canonical(
        np.array([1000.0, 999.0, 140.1, 0.0]), terminated=True
    )
    assert not _observed_future_is_canonical(
        np.array([1000.0, 999.0, 140.0, 0.01]), terminated=True
    )
    assert not _observed_future_is_canonical(
        np.array([999.0, 998.0, 139.0, 0.0]), terminated=False
    )


def test_development_evaluation_does_not_advance_training_rng(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.contrastive_rl.runner.evaluate_actor",
        lambda *args, **kwargs: {"separate": True},
    )
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.learner = object()
    runner.learner_state = SimpleNamespace(actor=SimpleNamespace(params={}))
    runner.actor_rng_key = jax.random.PRNGKey(1)
    runner.update_rng_key = jax.random.PRNGKey(2)
    runner.future_rng = np.random.default_rng(3)
    from benchmarks.contrastive_rl.execution import TrainingTrackStream

    runner.track_stream = TrainingTrackStream(11)
    before = (
        np.asarray(runner.actor_rng_key).copy(),
        np.asarray(runner.update_rng_key).copy(),
        json.dumps(runner.future_rng.bit_generator.state, sort_keys=True),
        json.dumps(runner.track_stream.state, sort_keys=True),
    )
    assert runner.development_evaluation() == {"separate": True}
    np.testing.assert_array_equal(before[0], runner.actor_rng_key)
    np.testing.assert_array_equal(before[1], runner.update_rng_key)
    assert before[2] == json.dumps(
        runner.future_rng.bit_generator.state, sort_keys=True
    )
    assert before[3] == json.dumps(runner.track_stream.state, sort_keys=True)


def test_training_outcome_summary_preserves_partial_boundary_episode():
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.training_outcomes = []
    runner.current_episode_ended = False
    runner.episode_id = 9
    runner.episode_step = 17
    runner.current_outcome = np.array([125.0, 124.0, 1.7, 0.0])
    runner.first_physical_canonical_success_transition = None
    partial = runner.outcomes_summary()["partial_episode_at_training_boundary"]
    assert partial == {
        "episode_id": 9,
        "transition_count": 17,
        "position_m": 125.0,
        "elapsed_time_s": 1.7,
        "max_speed_violation_m_s": 0.0,
        "final_route_progress": 0.125,
    }


def test_depths_change_only_frozen_depth_field():
    depth4 = ReferenceCoreConfig(depth=4)
    depth16 = ReferenceCoreConfig(depth=16)
    first = vars(depth4).copy()
    second = vars(depth16).copy()
    assert first.pop("depth") == 4
    assert second.pop("depth") == 16
    assert first == second
    assert PolicyRunner.config_batch_size.fget(None) == 256
