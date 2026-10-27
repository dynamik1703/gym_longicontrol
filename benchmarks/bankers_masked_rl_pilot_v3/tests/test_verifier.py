"""Independent memory, lifecycle, preview and adversarial evidence fixtures."""

import copy

import gymnasium as gym
import numpy as np
import pytest

from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from benchmarks.bankers_masked_rl_pilot_v3.shield_treatments import METHODS
from benchmarks.bankers_masked_rl_pilot_v3.training import Runner
from benchmarks.bankers_masked_rl_pilot_v3.verification import Auditor


class Tracks:
    def __init__(self, specifications):
        self.specifications = specifications
        self.cursor = 0

    def __call__(self, _):
        positions, limits = self.specifications[self.cursor % len(self.specifications)]
        self.cursor += 1
        return make_track(positions, limits)


def context(method, positions=(0,), limits=(100,), state=None, max_steps=1800,
            more_tracks=()):
    generator = Tracks([(positions, limits), *more_tracks])
    env = gym.wrappers.TimeLimit(LongiControlEnv(track_generator=generator), max_steps)
    runner = Runner(method, 'V3_SYNTHETIC_VERIFIER_' + method, env, True, 4096,
                    fixture_initial_state=state)
    auditor = Auditor(synthetic=True, initial_state=state, max_episode_steps=max_steps)
    return runner, auditor


def drive(runner, full=False):
    prepared = runner.prepare()
    prepared['proposal'] = (max(i for i, b in enumerate(runner.physical_mask) if b)
                            if runner.method == METHODS[2] else 20)
    return runner.advance(full_evidence=full, learn=False, prepared=prepared)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('scenario', [
    'reset', 'two_steps', 'new_lower', 'remember_lower', 'new_higher',
    'two_preview', 'cross_sign', 'preview_below', 'preview_exact', 'preview_above',
])
def test_independent_memory_scenarios(method, scenario, monkeypatch):
    state = None
    positions, limits = [0, 300], [60, 20]
    if scenario in ('new_lower', 'remember_lower', 'new_higher'):
        state = VehicleState(position_m=149.8, velocity_m_s=5.0)
        if scenario == 'new_higher':
            limits = [20, 60]
    if scenario == 'two_preview':
        positions, limits = [0, 100, 200], [60, 20, 60]
        state = VehicleState(position_m=50.5, velocity_m_s=5.0)
    if scenario == 'cross_sign':
        positions, limits = [0, 100], [60, 20]
        state = VehicleState(position_m=99.9, velocity_m_s=4.0)
    if scenario.startswith('preview_'):
        x = {'preview_below': np.nextafter(150.0, -np.inf),
             'preview_exact': 150.0, 'preview_above': np.nextafter(150.0, np.inf)}[scenario]
        state = VehicleState(position_m=float(x), velocity_m_s=4.0)
    r, audit = context(method, positions, limits, state)
    import benchmarks.bankers_masked_rl_pilot_v3.verification as module
    frozen = module.verify_envelope
    calls = []

    def checked(current, previous, track):
        assert current is not previous
        assert previous.last_observation is None
        calls.append((copy.deepcopy(previous), copy.deepcopy(current)))
        return frozen(current, previous, track)

    monkeypatch.setattr(module, 'verify_envelope', checked)
    for _ in range(1 if scenario == 'reset' else 4):
        row = drive(r, full=True)
        audit.verify(row, r.environment.unwrapped.track, method, full=True)
        assert audit.memory.state == r.environment.unwrapped.state
    assert calls
    if scenario == 'remember_lower':
        assert audit.memory.remembered_signs


@pytest.mark.parametrize('method', METHODS)
def test_multi_episode_timelimit_and_different_track(method):
    r, audit = context(method, [0], [100], max_steps=3,
                       more_tracks=[([0], [20]), ([0, 100], [60, 20])])
    for _ in range(9):
        row = drive(r, full=True)
        audit.verify(row, r.environment.unwrapped.track, method, full=True)
    assert audit.rows == 9 and len(audit.episodes) == 3
    assert audit.task_cursor == 2 and audit.episode_step == 3


@pytest.mark.parametrize('method', METHODS)
def test_full_route_termination(method):
    # A trusted synthetic initial state isolates actual physical termination,
    # rather than relabeling a budget cutoff as completion.
    state = VehicleState(position_m=998.5, velocity_m_s=5.0)
    r, audit = context(method, [0], [100], state)
    for _ in range(20):
        row = drive(r, full=True)
        audit.verify(row, r.environment.unwrapped.track, method, full=True)
        if row['terminated']:
            break
    assert row['completed'] and row['terminated'] and not row['truncated']
    assert audit.ended


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('corruption', [
    'envelope', 'predecessor_state', 'missing', 'duplicate', 'wrong_track',
    'successor_binding', 'executed_action', 'energy', 'forgot_sign',
    'task_boundary', 'terminal', 'reward', 'mask_bit',
])
def test_corrupted_native_evidence_fails(method, corruption):
    r, audit = context(method, [0, 100, 200], [60, 20, 60])
    first = drive(r, full=True)
    track = r.environment.unwrapped.track
    audit.verify(first, track, method, full=True)
    row = copy.deepcopy(drive(r, full=True))
    if corruption == 'envelope':
        row['memory']['envelope'][0] = (0.0, 37.0)
    elif corruption == 'predecessor_state':
        row['memory']['state']['position_m'] += 0.01
    elif corruption == 'missing':
        row['native'] += 1
    elif corruption == 'duplicate':
        row = first
    elif corruption == 'wrong_track':
        track = make_track([0], [20])
        # The observation or exact physical route identity must reject this.
        fresh = Auditor(synthetic=True)
        with pytest.raises(Exception):
            fresh.verify(first, track, method, full=True)
        return
    elif corruption == 'successor_binding':
        row['next_binding'] = '0' * 64
    elif corruption == 'executed_action':
        row['executed'] = (row['executed'] + 1) % 21
    elif corruption == 'energy':
        row['signed_energy'] += 1e-5
    elif corruption == 'forgot_sign':
        row['memory']['remembered_signs'] = []
    elif corruption == 'task_boundary':
        row['task_cursor'] += 1
        row['episode_step'] = 0
    elif corruption == 'terminal':
        row['terminated'] = True
    elif corruption == 'reward':
        row['reward'] += 1
    elif corruption == 'mask_bit':
        row['next_policy_mask'][0] = False
    with pytest.raises(Exception):
        audit.verify(row, track, method, full=True)
