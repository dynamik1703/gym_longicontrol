"""Stateful independent causal reconstruction with genuine predecessor memory."""

import copy
from dataclasses import asdict

import numpy as np

from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG, CausalMemory
from benchmarks.bankers_viability_shield_v1.certificate import identities
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
from benchmarks.bankers_viability_shield_v2.independent_verification import verify_envelope
from benchmarks.bankers_viability_shield_v4.independent_verification import (
    physical_certificate, verify_compact,
)
from benchmarks.bankers_viability_shield_v4.serialization import canonical_bytes, digest, require
from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import _EpisodeMetricsAccumulator
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .sac_agent import GAMMA_NATIVE
from .shield_treatments import METHODS, causal_binding
from .training import signals, stream_rows, update_target


def equal(a, b, reason):
    require(canonical_bytes(a) == canonical_bytes(b), reason)


def check_task_identity(track, task):
    fp = fingerprint_track(track, v2_semantics())
    equal(fp.route_sha256, task['route_sha256'], 'actual route identity')
    equal(fp.full_task_sha256, task['full_task_sha256'], 'actual full-task identity')


class Auditor:
    def __init__(self, *, synthetic=False, initial_state=None, max_episode_steps=1800,
                 expected_identity=None):
        self.synthetic = synthetic
        self.initial_state = initial_state or VehicleState()
        require(synthetic or self.initial_state == VehicleState(), 'noncanonical initial state')
        self.max_episode_steps = max_episode_steps
        self.expected_identity = expected_identity
        self.vehicle = VehicleModel()
        self.original_identity = identities()
        self.rows = self.grid_states = self.candidates = self.selected = 0
        self.task_cursor = -1
        self.episode_step = 0
        self.ended = True
        self.memory = None
        self.metrics = None
        self.last_next_binding = None
        self.last_next_mask = None
        self.last_next_certificates = None
        self.episodes = []

    def _begin(self, row, track):
        require(self.ended and row['episode_step'] == 0, 'reset before actual task end')
        require(row['task_cursor'] == self.task_cursor + 1, 'missing/duplicate task cursor')
        task = row['task']
        check_task_identity(track, task)
        if not self.synthetic:
            equal(task, stream_rows()[row['task_cursor']], 'canonical task-stream identity')
        self.task_cursor = row['task_cursor']
        self.episode_step = 0
        self.ended = False
        self.memory = CausalMemory(state=self.initial_state)
        self.track = copy.deepcopy(track)
        self.metrics = _EpisodeMetricsAccumulator()
        self.last_next_binding = self.last_next_mask = self.last_next_certificates = None

    def _grid(self, row, memory, ordinary, mask_key, cert_key, full):
        bits = []
        for i, action in enumerate(ACTION_GRID):
            if full:
                rebuilt = verify_compact(row[cert_key][i], ordinary, memory,
                    self.vehicle, self.original_identity, row['identity'], action_index=i)
            else:
                rebuilt, _ = physical_certificate(ordinary, memory, action,
                    self.vehicle, self.original_identity)
            bits.append(rebuilt['accepted'])
            self.candidates += 1
        equal(bits, row[mask_key], 'independently rebuilt full mask')
        self.grid_states += 1

    def verify(self, row, track, method, full=False):
        require(row['contract'] == 'bankers-training-evidence-v2', 'evidence contract')
        require(type(row['native']) is int and row['native'] == self.rows,
                'missing/duplicate native record')
        require(row['replay_insertion'] == self.rows, 'replay insertion identity')
        if self.expected_identity is not None:
            equal(row['identity'], self.expected_identity, 'source/config identity')
        if row['task_cursor'] != self.task_cursor:
            self._begin(row, track)
        else:
            require(not self.ended, 'evidence after episode end')
        require(np.array_equal(track.positions_m, self.track.positions_m) and
                np.array_equal(track.limits_m_s, self.track.limits_m_s),
                'track changed inside episode')
        track = self.track
        require(row['episode_step'] == self.episode_step, 'episode-local discontinuity')
        # The authority is auditor-owned state propagated from the trusted reset,
        # never a restore of the producer's snapshot. Preserve the real predecessor
        # before incorporating any current observation.
        previous = copy.deepcopy(self.memory)
        memory = self.memory
        ordinary = observation(memory.state, track.sense(memory.state.position_m, 150),
                               self.vehicle.specs, CONFIG, 1.0)
        memory.ingest(ordinary)
        verify_envelope(memory, previous, track)
        equal(memory.snapshot(), row['memory'], 'recorded causal memory/envelope')
        equal(ordinary.tolist(), row['observation'], 'actual current observation')
        equal(causal_binding(ordinary, memory), row['binding'], 'current mask-state binding')
        if self.last_next_binding is not None:
            equal(row['binding'], self.last_next_binding, 'predecessor/successor alignment')
            equal(row['physical_mask'], self.last_next_mask, 'cached successor mask')
            if full:
                equal(row['certificates'], self.last_next_certificates,
                      'complete cached successor certificate identity')
        proposal, selected = row['proposal'], row['executed']
        require(type(proposal) is int and type(selected) is int and
                0 <= proposal < 21 and 0 <= selected < 21, 'native grid indices')
        action = ACTION_GRID[selected]
        if method == METHODS[0]:
            require(selected == proposal and row['selected_certificate'] is None, 'RAW action changed')
            equal(row['physical_mask'], [True] * 21, 'RAW admission')
        else:
            rebuilt, _ = physical_certificate(ordinary, memory, action,
                self.vehicle, self.original_identity)
            equal(rebuilt, row['selected_certificate'], 'independent selected production certificate')
            require(rebuilt['accepted'] and row['physical_mask'][selected], 'unsafe selected action')
            equal(digest(rebuilt), row['selected_certificate_sha256'], 'selected certificate hash')
            self.selected += 1
            if method == METHODS[2]:
                require(proposal == selected and row['physical_mask'][proposal], 'PRE replacement')
            else:
                nearest = min((i for i, valid in enumerate(row['physical_mask']) if valid),
                    key=lambda i: (abs(i - proposal), i))
                require(selected == nearest, 'POST nearest/lower tie semantics')
        actual, power, energy = advance(memory.state, action, self.vehicle, CONFIG)
        equal(asdict(actual), row['actual_successor'], 'actual native physics')
        equal(energy, row['signed_energy'], 'signed native energy')
        limit = track.sense(actual.position_m, 150).current_limit_m_s
        excess = max(0.0, actual.velocity_m_s - limit)
        equal(excess, row['speed_excess'], 'exact positive speed excess')
        equal(limit, row['actual_limit'], 'actual physical speed limit')
        if method != METHODS[0]:
            require(excess == 0.0, 'BANKERS_MASKED_RL_PILOT_V3_SAFETY_COUNTEREXAMPLE')
        completed = bool(actual.position_m >= 1000)
        terminated = completed
        truncated = self.episode_step + 1 >= self.max_episode_steps
        require(row['max_episode_steps'] == self.max_episode_steps, 'TimeLimit contract')
        equal(row['completed'], completed, 'completion')
        equal(row['terminated'], terminated, 'true termination')
        equal(row['truncated'], truncated, 'TimeLimit truncation')
        equal(row['discount'], 0.0 if terminated or truncated else GAMMA_NATIVE,
              'terminal native bootstrap')
        reward, costs = signals(dict(step_energy_kwh=energy, speed_excess_m_s=excess,
            elapsed_time_s=actual.elapsed_time_s), completed, terminated, truncated,
            proposal, selected)
        equal(row['reward'], reward, 'signed energy reward')
        equal(row['costs'], costs, 'independent cost arithmetic')
        self.metrics.update(velocity_m_s=actual.velocity_m_s, speed_limit_m_s=limit,
                            dt_s=CONFIG.dt_s)
        equal(asdict(self.metrics.snapshot(actual, completed=completed)),
              row['episode_metrics'], 'independent physical episode metrics')
        # Validate the recorded successor, including its new observation, without
        # advancing the authoritative current memory's observation history early.
        successor_before = copy.deepcopy(memory)
        successor_before.execute(action, self.vehicle)
        successor = copy.deepcopy(successor_before)
        next_o = observation(actual, track.sense(actual.position_m, 150),
                             self.vehicle.specs, CONFIG, 1.0)
        successor.ingest(next_o)
        verify_envelope(successor, successor_before, track)
        equal(successor.snapshot(), row['successor_memory'], 'successor causal reconstruction')
        equal(next_o.tolist(), row['successor_observation'], 'successor observation')
        equal(causal_binding(next_o, successor), row['next_binding'], 'successor mask binding')
        if method != METHODS[0] and (full or self.rows % 100 == 0):
            if not full or self.last_next_binding is None:
                self._grid(row, memory, ordinary, 'physical_mask', 'certificates', full)
            self._grid(row, successor, next_o, 'next_physical_mask', 'next_certificates', full)
        support = row['physical_mask'] if method == METHODS[2] else [True] * 21
        next_support = row['next_physical_mask'] if method == METHODS[2] else [True] * 21
        equal(row['policy_mask'], support, 'current categorical support')
        equal(row['next_policy_mask'], next_support, 'successor categorical support')
        if not self.synthetic:
            equal(row['updates_before'], update_target(self.rows), 'update counter before')
            equal(row['updates_after'], update_target(self.rows + 1), 'update counter after')
            equal(row['budget_truncated'], self.rows + 1 == 50000 and not (terminated or truncated),
                  'training-budget cutoff semantics')
        self.last_next_binding = row['next_binding']
        self.last_next_mask = row['next_physical_mask']
        self.last_next_certificates = row['next_certificates'] if full else None
        self.memory = successor_before
        self.episode_step += 1
        self.ended = bool(terminated or truncated)
        self.rows += 1
        if self.ended:
            self.episodes.append(dict(task_cursor=self.task_cursor, native_end=self.rows,
                metrics=asdict(self.metrics.snapshot(actual, completed=completed)),
                final_position_m=actual.position_m))
        return power

    def report(self):
        return dict(native_steps=self.rows, selected_exhaustive=self.selected,
            full_grid_states=self.grid_states, reconstructed_candidates=self.candidates,
            episodes=len(self.episodes), training_grid_exhaustive=False,
            training_grid_scope='native insertion % 100 == 0, current and successor; Diagnostic all states')


def audit_training_disk(directory, runner):
    """Read the immutable evidence stream afresh and check every replay slot."""
    from benchmarks.content_split_audit.streams import track_from_seed
    from .journal import Journal
    from .shield_treatments import features
    audit = Auditor(expected_identity=runner.identity)
    journal = Journal(directory / 'journal', runner.attempt, runner.identity)
    cursor, track, head = -1, None, journal.genesis
    for frame in journal.records():
        row = frame['row']
        if cursor != row['task_cursor']:
            cursor = row['task_cursor']
            seed = stream_rows()[cursor]['reset_seed']
            from .integrity import guard_seed
            guard_seed(seed)
            track = track_from_seed(seed)
        audit.verify(row, track, runner.method)
        i = row['replay_insertion']
        expected = dict(observation=features(row['observation'], CausalMemory.restore(row['memory'])),
            next_observation=features(row['successor_observation'], CausalMemory.restore(row['successor_memory'])),
            action=[row['proposal']], executed=[row['executed']], reward=[row['reward']],
            costs=row['costs'], discount=[row['discount']], mask=row['policy_mask'],
            next_mask=row['next_policy_mask'], binding=row['binding'], next_binding=row['next_binding'],
            insertion=i)
        for name, value in expected.items():
            require(np.array_equal(runner.replay.arrays[name][i],
                  np.asarray(value, dtype=runner.replay.arrays[name].dtype)),
                  'final replay slot ' + name)
        head = frame['sha256']
    require(audit.rows == runner.native == runner.replay.size == runner.replay.cursor == 50000,
            'final native/replay accounting')
    require(runner.agent.updates == 5000, 'final update accounting')
    return dict(**audit.report(), journal_head=head, replay_slots_verified=runner.replay.size)
