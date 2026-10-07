from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax")
pytest.importorskip("optax")

from benchmarks.contrastive_rl.config import ReferenceCoreConfig  # noqa: E402
from benchmarks.contrastive_rl.learner import (  # noqa: E402
    ContrastiveBatch,
    ReferenceLearner,
    load_state,
    save_state,
    tree_parameter_count,
)
from benchmarks.contrastive_rl.losses import (  # noqa: E402
    actor_objective,
    alpha_objective,
    association_score,
    contrastive_loss,
    infonce_loss_from_logits,
    pairwise_association_scores,
)
from benchmarks.contrastive_rl.projected_adapter import (  # noqa: E402
    contrastive_batch_from_recorded,
)
from gym_longicontrol.domain.task import TaskSpecification  # noqa: E402


def small_core(depth=4):
    config = ReferenceCoreConfig(
        state_dim=3,
        goal_dim=3,
        action_dim=1,
        depth=depth,
        width=16,
        embedding_dim=8,
    )
    return ReferenceLearner.create(config, seed=3)


def batch(reward_value=0.0):
    return ContrastiveBatch(
        states=jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 12,
        actions=jnp.linspace(-0.5, 0.5, 4).reshape(4, 1),
        critic_goals=jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 12,
        actor_goals=jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 12,
        historical_reward=jnp.full((4,), reward_value),
    )


def test_association_score_direction_and_pairwise_manual_parity():
    origin = jnp.array([[0.0, 0.0]])
    near = jnp.array([[1.0, 0.0]])
    far = jnp.array([[2.0, 0.0]])
    assert float(association_score(origin, near)[0]) > float(
        association_score(origin, far)[0]
    )
    logits = pairwise_association_scores(
        jnp.array([[0.0, 0.0], [2.0, 0.0]]),
        jnp.array([[1.0, 0.0], [4.0, 0.0]]),
        epsilon=1e-12,
    )
    np.testing.assert_allclose(logits, [[-1, -4], [-1, -2]], atol=1e-6)


def test_infonce_matches_manual_rows_and_known_duplicate_behavior():
    logits = jnp.array([[2.0, 0.0], [1.0, 3.0]])
    loss, metrics = infonce_loss_from_logits(logits, logsumexp_penalty=0.0)
    expected = np.mean(
        [np.log(np.exp(2) + 1) - 2, np.log(np.exp(1) + np.exp(3)) - 3]
    )
    assert float(loss) == pytest.approx(expected, rel=1e-6)
    assert float(metrics["classification_loss"]) == pytest.approx(expected, rel=1e-6)

    duplicate_logits = jnp.zeros((3, 3))
    _, duplicate_metrics = infonce_loss_from_logits(
        duplicate_logits, logsumexp_penalty=0.0
    )
    assert float(duplicate_metrics["classification_loss"]) == pytest.approx(
        np.log(3), rel=1e-6
    )


def test_coincident_embeddings_have_finite_loss_and_gradients():
    embeddings = jnp.zeros((3, 2))

    def loss_fn(value):
        return contrastive_loss(value, embeddings)[0]

    value, gradients = jax.value_and_grad(loss_fn)(embeddings)
    assert np.isfinite(value)
    assert np.isfinite(np.asarray(gradients)).all()


def test_actor_objective_backpropagates_through_action_score():
    def loss(action):
        score = association_score(action, jnp.ones_like(action))
        return actor_objective(score, jnp.zeros(action.shape[0]), jnp.array(0.0))

    gradients = jax.grad(loss)(jnp.zeros((2, 1)))
    assert np.isfinite(np.asarray(gradients)).all()
    assert not np.allclose(gradients, 0.0)


def test_actor_step_does_not_update_critic_and_alpha_has_intended_direction():
    learner, state = small_core()
    critic_before = jax.tree_util.tree_map(np.asarray, state.critic.params)
    state_after, _ = learner.actor_alpha_step(
        state, batch(), jax.random.PRNGKey(12)
    )
    for before, after in zip(
        jax.tree_util.tree_leaves(critic_before),
        jax.tree_util.tree_leaves(state_after.critic.params),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)

    gradient = jax.grad(alpha_objective)(jnp.array(0.0), jnp.array([-1.0]), -0.5)
    assert float(gradient) > 0.0


def test_depth_convention_shapes_and_exact_parameter_increase():
    learner4, state4 = small_core(depth=4)
    learner16, state16 = small_core(depth=16)
    assert learner4.config.depth == 4
    assert learner16.config.depth == 16
    per_dense = (16 + 1) * 16
    per_layer_norm = 2 * 16
    expected_per_network_increase = 12 * (per_dense + per_layer_norm)
    assert (
        tree_parameter_count(state16.actor.params)
        - tree_parameter_count(state4.actor.params)
        == expected_per_network_increase
    )
    critic_increase = (
        tree_parameter_count(state16.critic.params)
        - tree_parameter_count(state4.critic.params)
    )
    assert critic_increase == 2 * expected_per_network_increase
    assert learner4.deterministic_action(
        state4.actor.params, jnp.ones((2, 3)), jnp.ones((2, 3))
    ).shape == (2, 1)


def test_save_load_reproduces_outputs(tmp_path: Path):
    learner, state = small_core()
    expected = learner.deterministic_action(
        state.actor.params, jnp.ones((2, 3)), jnp.ones((2, 3))
    )
    path = tmp_path / "state.msgpack"
    save_state(path, state)
    restored = load_state(path, state)
    actual = learner.deterministic_action(
        restored.actor.params, jnp.ones((2, 3)), jnp.ones((2, 3))
    )
    np.testing.assert_array_equal(expected, actual)


def test_historical_reward_does_not_affect_losses_or_gradients():
    learner, state = small_core()

    def value_and_grad(candidate_batch):
        def loss(params):
            return learner.critic_loss(params, candidate_batch)[0]

        return jax.value_and_grad(loss)(state.critic.params)

    loss_a, gradients_a = value_and_grad(batch(-1000.0))
    loss_b, gradients_b = value_and_grad(batch(1000.0))
    np.testing.assert_array_equal(loss_a, loss_b)
    for first, second in zip(
        jax.tree_util.tree_leaves(gradients_a),
        jax.tree_util.tree_leaves(gradients_b),
        strict=True,
    ):
        np.testing.assert_array_equal(first, second)


def test_recorded_adapter_projects_futures_without_inserting_success():
    task = TaskSpecification(140.0, 0.0)
    future_outcomes = np.array(
        [
            [500.0, 499.0, 100.0, 0.0],
            [1000.0, 999.0, 141.0, 0.0],
            [1000.0, 999.0, 130.0, 0.1],
        ]
    )
    adapted = contrastive_batch_from_recorded(
        states=np.zeros((3, 3)),
        actions=np.zeros((3, 1)),
        future_outcomes=future_outcomes,
        future_terminated=np.array([False, True, True]),
        task=task,
        historical_reward=np.array([-10.0, 0.0, 10.0]),
    )
    expected = np.array([[0.5, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    np.testing.assert_array_equal(adapted.critic_goals, expected)
    np.testing.assert_array_equal(adapted.actor_goals, expected)
    assert not np.all(np.asarray(adapted.critic_goals) == 1.0, axis=-1).any()
