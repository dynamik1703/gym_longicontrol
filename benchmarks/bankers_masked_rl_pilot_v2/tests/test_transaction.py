"""Crash recovery across real warm-up, updates, action and checkpoint stages."""

import copy
import shutil

import numpy as np
import pytest
import torch

from benchmarks.bankers_masked_rl_pilot_v2.integrity import canonical_bytes
from benchmarks.bankers_masked_rl_pilot_v2.journal import Journal
from benchmarks.bankers_masked_rl_pilot_v2.transaction import Engine
from benchmarks.bankers_masked_rl_pilot_v2.training import Runner
from benchmarks.bankers_masked_rl_pilot_v2.verification import Auditor
from test_verifier import context
from benchmarks.bankers_masked_rl_pilot_v2.shield_treatments import METHODS


@pytest.fixture(scope='module', params=METHODS)
def recovery_prefix(request, tmp_path_factory):
    root = tmp_path_factory.mktemp('v2-recovery-' + request.param)
    runner, _ = context(request.param, [0], [20], max_steps=1800)
    # Larger replay is required for the unchanged 10k warm-up rule.
    from benchmarks.bankers_masked_rl_pilot_v2.replay import Replay
    runner.replay = Replay(12000)
    engine = Engine(root, runner)
    for _ in range(10000):
        engine.step()
    assert engine.runner.native == 10000 and engine.runner.agent.updates == 0
    return root, request.param, runner.attempt


def assert_same(a, b):
    assert a.native == b.native and a.agent.updates == b.agent.updates
    assert canonical_bytes(a.memory.snapshot()) == canonical_bytes(b.memory.snapshot())
    assert a.cursor == b.cursor and a.pending_reset == b.pending_reset
    assert canonical_bytes(a.episodes) == canonical_bytes(b.episodes)
    assert canonical_bytes(a.diagnostics) == canonical_bytes(b.diagnostics)
    for name, module in a.agent.modules().items():
        for key, value in module.state_dict().items():
            assert torch.equal(value, b.agent.modules()[name].state_dict()[key])
    assert torch.equal(a.agent.log_alpha, b.agent.log_alpha)
    assert torch.equal(a.agent.multipliers, b.agent.multipliers)
    assert torch.equal(a.policy_rng.get_state(), b.policy_rng.get_state())
    assert a.replay_rng.bit_generator.state == b.replay_rng.bit_generator.state
    for key, array in a.replay.arrays.items():
        assert np.array_equal(array, b.replay.arrays[key])
    # Optimizer moments, step counts and parameter groups are part of exact state.
    def recursive(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x:
                recursive(x[k], y[k])
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for v, w in zip(x, y):
                recursive(v, w)
        else:
            assert x == y
    for key, optimizer in a.agent.optimizers.items():
        recursive(optimizer.state_dict(), b.agent.optimizers[key].state_dict())


@pytest.mark.parametrize('stage', ['after_intent', 'after_action', 'before_replay',
    'after_replay', 'before_update', 'after_update', 'after_commit',
    'before_checkpoint', 'after_checkpoint'])
def test_same_attempt_recovery(recovery_prefix, stage, tmp_path):
    root, method, attempt = recovery_prefix
    reference, interrupted = tmp_path / 'reference', tmp_path / 'interrupted'
    shutil.copytree(root, reference)
    shutil.copytree(root, interrupted)
    expected = Engine.recover(reference, attempt, checkpoint_interval=8)
    for _ in range(8):
        expected.step()
    actual = Engine.recover(interrupted, attempt, checkpoint_interval=8)
    for _ in range(7):
        actual.step()

    def crash(point):
        if point == stage:
            raise RuntimeError('V2_INJECTED_PROCESS_INTERRUPTION')

    with pytest.raises(RuntimeError, match='V2_INJECTED'):
        actual.step(hook=crash)
    recovered = Engine.recover(interrupted, attempt, checkpoint_interval=8)
    assert recovered.runner.native == 10008 and recovered.runner.agent.updates == 1
    assert_same(expected.runner, recovered.runner)
    assert expected.head == recovered.head
    assert len(list(recovered.journal.records())) == 10008
    if stage == 'after_action':
        # Full independent native transition replay after recovery, with the
        # declared training grid audit sample, for every treatment.
        audit = Auditor(synthetic=True)
        from benchmarks.bankers_viability_shield_v1.fixtures import make_track
        track = make_track([0], [20])
        for frame in recovered.journal.records():
            audit.verify(frame['row'], track, method)
        assert audit.rows == 10008


@pytest.mark.parametrize('method', METHODS)
def test_recovery_between_terminal_and_reset(method, tmp_path):
    runner, _ = context(method, [0], [100], max_steps=2,
                       more_tracks=[([0], [20])])
    engine = Engine(tmp_path / 'run', runner, checkpoint_interval=1)
    engine.step(learn=False)
    engine.step(learn=False)
    assert engine.runner.pending_reset
    expected_runner = copy.deepcopy(engine.runner)
    expected = expected_runner.advance(learn=False)
    recovered = Engine.recover(tmp_path / 'run', runner.attempt,
                               checkpoint_interval=1, learn=False)
    actual = recovered.step(learn=False)
    assert canonical_bytes(actual) == canonical_bytes(expected)
    assert recovered.audit.task_cursor == 1


def test_duplicate_attempt_and_unverified_recovery_stop(tmp_path):
    runner, _ = context(METHODS[0])
    root = tmp_path / 'run'
    engine = Engine(root, runner)
    with pytest.raises(Exception):
        Engine(root, Runner(METHODS[0], runner.attempt, synthetic=True))
    engine.step(learn=False)
    with (root / 'journal/live.jsonl').open('ab') as out:
        out.write(b'{"torn":')
    with pytest.raises(Exception, match='torn'):
        Engine.recover(root, runner.attempt, learn=False)
