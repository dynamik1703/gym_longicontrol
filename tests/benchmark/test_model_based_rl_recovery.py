"""Deterministic exact-resume and non-interference integration fixtures."""

from __future__ import annotations

import copy
import random

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tianshou.data import Batch, Collector, ReplayBuffer  # noqa: E402
from tianshou.policy import BasePolicy  # noqa: E402

from benchmarks.model_based_rl.checkpointing import (  # noqa: E402
    atomic_torch_save,
    capture_rng_states,
    load_checkpoint,
    restore_rng_states,
)
from benchmarks.model_based_rl.config import load_configuration  # noqa: E402
from benchmarks.model_based_rl.execution import (  # noqa: E402
    TrainingTrackStream,
    scientific_hashes,
)
from benchmarks.model_based_rl.recovery import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    collector_state,
    execution_hashes,
    heartbeat_payload,
    restore_collector,
    runtime_versions,
    verify_recovery_checkpoint,
    verify_scientific_freeze,
    write_heartbeat,
    write_verified_checkpoint,
)


class ZeroPolicy(BasePolicy):
    def __init__(self, action_space):
        super().__init__(action_space=action_space)

    def forward(self, batch, state=None, **kwargs):
        return Batch(act=np.zeros((len(batch), 1)), state=state)

    def learn(self, batch, **kwargs):
        return {}


class CheckpointableTrackEnvironment(gym.Env):
    """Tiny non-simulator fixture exercising env, collector and track RNG state."""

    observation_space = gym.spaces.Box(-1e12, 1e12, (2,), dtype=np.float64)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def __init__(self, seed):
        self.stream = TrainingTrackStream(seed)
        self.track_seed = 0
        self.position = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.track_seed = self.stream.next_seed()
        self.position = 0.0
        return self._observation(), {}

    def step(self, action):
        self.position += float(np.asarray(action).reshape(-1)[0]) + 0.25
        return self._observation(), self.position, False, False, {}

    def _observation(self):
        return np.asarray([self.position, self.track_seed], dtype=np.float64)


class DeterministicLearnedFixture:
    """Small causal analogue covering every stateful MBRL operation."""

    def __init__(self, seed: int = 19):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.track_stream = TrainingTrackStream(seed)
        self.model_rng = np.random.default_rng([seed, 101])
        self.synthetic_rng = np.random.default_rng([seed, 202])
        self.actor = torch.nn.Linear(2, 1)
        self.critic = torch.nn.Linear(3, 1)
        self.model = torch.nn.Linear(3, 2)
        self.policy_optimizer = torch.optim.Adam(
            [*self.actor.parameters(), *self.critic.parameters()], lr=1e-3
        )
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.position = 0.0
        self.velocity = 0.0
        self.episode_step = 0
        self.completed_episodes = 0
        self.current_track_seed = self.track_stream.next_seed()
        self.replay: list[tuple[list[float], float, list[float]]] = []
        self.synthetic: list[list[float]] = []
        self.pid = np.zeros(2, dtype=np.float64)
        self.counters = {
            "real_transitions": 0,
            "synthetic_transitions": 0,
            "rl_gradient_updates": 0,
            "model_gradient_updates": 0,
            "model_refreshes": 0,
        }

    def operation(self) -> dict:
        observation = torch.tensor(
            [self.position, self.velocity], dtype=torch.float32
        )
        actor_noise = torch.randn(1) * 0.01
        action_tensor = torch.tanh(self.actor(observation) + actor_noise)
        action = float(action_tensor.detach().item())
        next_velocity = self.velocity + 0.1 * action
        next_position = self.position + 0.1 * next_velocity
        next_observation = [next_position, next_velocity]
        self.replay.append(([self.position, self.velocity], action, next_observation))
        self.position, self.velocity = next_position, next_velocity
        self.episode_step += 1
        self.counters["real_transitions"] += 1

        replay_indices: list[int] = []
        loss_value = None
        if len(self.replay) >= 2:
            replay_indices = np.random.randint(0, len(self.replay), size=2).tolist()
            rows = [self.replay[index] for index in replay_indices]
            observations = torch.tensor([row[0] for row in rows])
            actions = torch.tensor([[row[1]] for row in rows])
            targets = torch.tensor([[row[2][0]] for row in rows])
            prediction = self.critic(torch.cat([observations, actions], dim=1))
            loss = torch.nn.functional.mse_loss(prediction, targets)
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.policy_optimizer.step()
            self.counters["rl_gradient_updates"] += 1
            loss_value = float(loss.detach())

        permutation: list[int] = []
        bootstrap: list[int] = []
        if self.counters["real_transitions"] % 5 == 0:
            permutation = self.model_rng.permutation(len(self.replay)).tolist()
            bootstrap = self.model_rng.choice(
                len(self.replay), size=len(self.replay), replace=True
            ).tolist()
            rows = [self.replay[index] for index in bootstrap]
            inputs = torch.tensor(
                [row[0] + [row[1]] for row in rows], dtype=torch.float32
            )
            targets = torch.tensor([row[2] for row in rows], dtype=torch.float32)
            model_loss = torch.nn.functional.mse_loss(self.model(inputs), targets)
            self.model_optimizer.zero_grad(set_to_none=True)
            model_loss.backward()
            self.model_optimizer.step()
            self.counters["model_gradient_updates"] += 1
            self.counters["model_refreshes"] += 1

        selected_member = None
        model_prediction = None
        synthetic_transition = None
        if self.counters["real_transitions"] >= 5:
            selected_member = int(self.synthetic_rng.integers(0, 3))
            model_input = torch.tensor(
                [self.position, self.velocity, action], dtype=torch.float32
            )
            predicted = self.model(model_input).detach().numpy()
            noise = self.synthetic_rng.normal(size=2) * 0.001
            synthetic_transition = (predicted + noise).tolist()
            model_prediction = predicted.tolist()
            self.synthetic.append(synthetic_transition)
            self.counters["synthetic_transitions"] += 1

        alpha_loss = -(self.log_alpha * action_tensor.detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        episode_boundary = self.episode_step == 7
        if episode_boundary:
            self.pid += np.asarray([abs(self.velocity), max(0.0, self.position - 0.1)])
            self.completed_episodes += 1
            self.position = 0.0
            self.velocity = 0.0
            self.episode_step = 0
            self.current_track_seed = self.track_stream.next_seed()
        return {
            "next_environment_transition": next_observation,
            "track_seed": self.current_track_seed,
            "actor_action": action,
            "replay_sampling_indices": replay_indices,
            "model_permutation": permutation,
            "model_bootstrap": bootstrap,
            "selected_ensemble_member": selected_member,
            "model_prediction": model_prediction,
            "synthetic_transition": synthetic_transition,
            "rl_loss": loss_value,
            "alpha": float(self.log_alpha.detach().exp()),
            "pid": self.pid.tolist(),
            "episode_boundary": episode_boundary,
            "python_rng": random.random(),
            "counters": dict(self.counters),
        }

    def state_dict(self) -> dict:
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic": copy.deepcopy(self.critic.state_dict()),
            "model": copy.deepcopy(self.model.state_dict()),
            "policy_optimizer": copy.deepcopy(self.policy_optimizer.state_dict()),
            "model_optimizer": copy.deepcopy(self.model_optimizer.state_dict()),
            "alpha_optimizer": copy.deepcopy(self.alpha_optimizer.state_dict()),
            "log_alpha": self.log_alpha.detach().clone(),
            "position": self.position,
            "velocity": self.velocity,
            "episode_step": self.episode_step,
            "completed_episodes": self.completed_episodes,
            "current_track_seed": self.current_track_seed,
            "track_stream": copy.deepcopy(self.track_stream.state_dict()),
            "replay": copy.deepcopy(self.replay),
            "synthetic": copy.deepcopy(self.synthetic),
            "pid": self.pid.copy(),
            "counters": dict(self.counters),
        }

    def load_state_dict(self, state: dict, rng_states: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.model.load_state_dict(state["model"])
        self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.model_optimizer.load_state_dict(state["model_optimizer"])
        self.log_alpha.data.copy_(state["log_alpha"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        for name in (
            "position",
            "velocity",
            "episode_step",
            "completed_episodes",
            "current_track_seed",
        ):
            setattr(self, name, state[name])
        self.track_stream.load_state_dict(state["track_stream"])
        self.replay = copy.deepcopy(state["replay"])
        self.synthetic = copy.deepcopy(state["synthetic"])
        self.pid = state["pid"].copy()
        self.counters = dict(state["counters"])
        restore_rng_states(
            rng_states, model_rng=self.model_rng, synthetic_rng=self.synthetic_rng
        )


def fixture_payload(fixture: DeterministicLearnedFixture) -> dict:
    configuration = load_configuration()
    state = fixture.state_dict()
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "attempt_id": "fixture-attempt",
        "training_seed": 19,
        "model_condition": "learned",
        "policy": {"actor": state["actor"], "critic": state["critic"]},
        "policy_optimizers": {
            "actor": state["policy_optimizer"],
            "critics": state["policy_optimizer"],
        },
        "lagrangian_states": [fixture.pid.copy()],
        "real_rl_replay": state["replay"],
        "real_model_replay": state["replay"],
        "real_source_replay": state["replay"],
        "synthetic_replay": state["synthetic"],
        "environment_state": {
            "position": state["position"],
            "velocity": state["velocity"],
            "episode_step": state["episode_step"],
        },
        "collector_state": {"observation": [state["position"], state["velocity"]]},
        "track_stream_state": state["track_stream"],
        "counters": state["counters"],
        "diagnostics": [],
        "hashes": {
            "configuration": configuration.configuration_sha256,
            "scientific_sources": scientific_hashes(),
            "execution_sources": execution_hashes(),
        },
        "runtime_versions": runtime_versions(),
        "previous_completed_episodes": state["completed_episodes"],
        "pid_real_episode_updates": state["completed_episodes"],
        "development_checkpoints_completed": [],
        "last_recovery_checkpoint": None,
        "last_scientific_checkpoint": None,
        "rng_states": capture_rng_states(
            model_rng=fixture.model_rng, synthetic_rng=fixture.synthetic_rng
        ),
        "learned_model": state["model"],
        "model_optimizer_states": state["model_optimizer"],
        "normalization": {"input": [0.0], "target": [0.0]},
        "learned_model_full_state": state["model"],
        "fixture_state": state,
    }


@pytest.mark.parametrize("boundary", [3, 4, 5, 6, 7])
def test_exact_resume_parity_across_scientific_boundaries(tmp_path, boundary):
    continuous = DeterministicLearnedFixture()
    for _ in range(boundary):
        continuous.operation()
    payload = fixture_payload(continuous)
    checkpoint = tmp_path / f"boundary-{boundary}" / "checkpoint.pt"
    write_verified_checkpoint(checkpoint, payload, kind="recovery")
    expected = [continuous.operation() for _ in range(3)]
    expected_state = continuous.state_dict()

    restored_payload, _metadata = verify_recovery_checkpoint(
        checkpoint,
        configuration_sha256=load_configuration().configuration_sha256,
        expected_attempt_id="fixture-attempt",
    )
    resumed = DeterministicLearnedFixture(seed=999)
    resumed.load_state_dict(
        restored_payload["fixture_state"], restored_payload["rng_states"]
    )
    actual = [resumed.operation() for _ in range(3)]
    assert actual == expected
    actual_state = resumed.state_dict()
    for component in ("actor", "critic", "model"):
        for name, value in expected_state[component].items():
            assert torch.equal(value, actual_state[component][name])
    assert actual_state["replay"] == expected_state["replay"]
    assert actual_state["synthetic"] == expected_state["synthetic"]
    np.testing.assert_array_equal(actual_state["pid"], expected_state["pid"])
    assert actual_state["counters"] == expected_state["counters"]


def test_heartbeat_and_checkpointing_do_not_change_scientific_outputs(tmp_path):
    control = DeterministicLearnedFixture()
    control_outputs = [control.operation() for _ in range(12)]
    instrumented = DeterministicLearnedFixture()
    instrumented_outputs = []
    recorder = type(
        "Recorder",
        (),
        {"completed_episodes": 0, "episode_step": 0},
    )()
    synthetic = type("Synthetic", (), {"sampled_total": 0})()
    for index in range(12):
        instrumented_outputs.append(instrumented.operation())
        recorder.completed_episodes = instrumented.completed_episodes
        recorder.episode_step = instrumented.episode_step
        write_heartbeat(
            tmp_path,
            heartbeat_payload(
                condition="learned",
                seed=19,
                attempt_id="fixture-attempt",
                counters=instrumented.counters,
                recorder=recorder,
                synthetic=synthetic,
                status="RUNNING",
                last_recovery_checkpoint=None,
                last_scientific_checkpoint=None,
            ),
        )
        if index in (4, 9):
            write_verified_checkpoint(
                tmp_path / f"step-{index + 1}" / "checkpoint.pt",
                fixture_payload(instrumented),
                kind="recovery",
            )
    assert instrumented_outputs == control_outputs


def test_scientific_freeze_is_byte_exact():
    assert verify_scientific_freeze() == scientific_hashes()


def test_environment_and_collector_continue_exactly_after_restore(tmp_path):
    recorder = CheckpointableTrackEnvironment(313)
    collector = Collector(
        ZeroPolicy(recorder.action_space), recorder, ReplayBuffer(32)
    )
    collector.collect(n_step=1)
    payload = {
        "model_condition": "physics",
        "policy": {},
        "policy_optimizers": {},
        "lagrangian_states": [],
        "real_rl_replay": collector.buffer,
        "real_model_replay": [],
        "real_source_replay": [],
        "synthetic_replay": [],
        "environment_state": copy.deepcopy(recorder),
        "collector_state": collector_state(collector),
        "track_stream_state": copy.deepcopy(recorder.stream.state_dict()),
        "counters": {"real_transitions": 1},
        "diagnostics": [],
        "hashes": {},
        "rng_states": capture_rng_states(
            model_rng=np.random.default_rng(1),
            synthetic_rng=np.random.default_rng(2),
        ),
    }
    checkpoint = tmp_path / "actual-environment.pt"
    digest = atomic_torch_save(checkpoint, payload)
    expected_stats = collector.collect(n_step=1)
    expected = copy.deepcopy(collector.data.obs)
    expected_next_track = copy.deepcopy(recorder.stream).next_seed()

    loaded = load_checkpoint(checkpoint, expected_sha256=digest)
    placeholder_recorder = CheckpointableTrackEnvironment(999)
    restored_collector = Collector(
        ZeroPolicy(placeholder_recorder.action_space),
        placeholder_recorder,
        ReplayBuffer(32),
    )
    restored_recorder = loaded["environment_state"]
    restore_collector(
        restored_collector,
        state=loaded["collector_state"],
        recorder=restored_recorder,
        replay_buffer=loaded["real_rl_replay"],
    )
    actual_stats = restored_collector.collect(n_step=1)
    actual = copy.deepcopy(restored_collector.data.obs)
    restored_stream = restored_recorder.stream
    actual_next_track = copy.deepcopy(restored_stream).next_seed()

    assert int(actual_stats["n/st"]) == int(expected_stats["n/st"]) == 1
    assert int(actual_stats["n/ep"]) == int(expected_stats["n/ep"]) == 0
    np.testing.assert_array_equal(actual, expected)
    assert actual_next_track == expected_next_track
    assert len(restored_collector.buffer) == len(collector.buffer)
    recorder.close()
    restored_recorder.close()
