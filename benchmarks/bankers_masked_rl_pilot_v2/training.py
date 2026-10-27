"""One native step, one replay insertion; no policy-time action chunks."""

import copy
import json
from dataclasses import asdict
from functools import lru_cache

import gymnasium as gym
import numpy as np
import torch

from benchmarks.bankers_viability_shield_v1.backup_policy import CausalMemory
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.hybrid_rl_design.split import canonical_sha256
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .integrity import ROOT, guard_seed, identity
from .replay import Replay
from .sac_agent import GAMMA_NATIVE, DiscreteSAC
from .shield_treatments import METHODS, Treatment, features

STREAM_SHA = '08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c'


def update_target(native):
    return max(0, min(5000, (native - 10000) * 5000 // 40000))


@lru_cache(maxsize=1)
def stream_rows():
    value = json.loads((ROOT / 'benchmarks/constrained_v2_collector_replication/stream_ledger.json').read_text())['11']
    rows = value['accepted_tasks']
    if value['ordered_sha256'] != STREAM_SHA or len(rows) != 865:
        raise RuntimeError('canonical stream stored identity mismatch')
    if canonical_sha256(rows) != STREAM_SHA:
        raise RuntimeError('canonical stream actual ordered identity mismatch')
    return rows


def signals(info, completed, terminated, truncated, proposal, executed):
    ended = terminated or truncated
    return (-1000.0 * float(info['step_energy_kwh']),
            [float(info['speed_excess_m_s']),
             float(ended and (not completed or info['elapsed_time_s'] > 140.0)),
             abs(ACTION_GRID[proposal] - ACTION_GRID[executed])])


class Runner:
    def __init__(self, method, attempt, environment=None, synthetic=False, capacity=300000,
                 fixture_initial_state=None):
        self.method, self.attempt = method, attempt
        self.synthetic = synthetic
        self.identity = identity()
        self.agent = DiscreteSAC(11)
        self.replay = Replay(capacity)
        self.treatment = Treatment(method, self.identity)
        self.environment = environment if environment is not None else gym.wrappers.TimeLimit(StochasticTrack(), 1800)
        self.policy_rng = torch.Generator().manual_seed(0x504F4C59 + 11)
        self.replay_rng = np.random.default_rng(np.random.SeedSequence([11, 0x52504C59]))
        self.native, self.cursor, self.episode_step = 0, 0, 0
        self.pending_reset = True
        self.task = None
        self.memory = None
        self.ordinary = None
        self.physical_mask = self.certificates = self.binding = None
        self.episodes, self.diagnostics = [], []
        self.fixture_initial_state = fixture_initial_state

    def reset(self):
        if self.synthetic:
            self.ordinary, _ = self.environment.reset(seed=11)
            fp = fingerprint_track(self.environment.unwrapped.track, v2_semantics())
            self.task = dict(reset_seed=None, full_task_sha256=fp.full_task_sha256,
                             route_sha256=fp.route_sha256, fixture=True)
        else:
            self.task = copy.deepcopy(stream_rows()[self.cursor])
            seed = self.task['reset_seed']
            guard_seed(seed)
            self.ordinary, _ = self.environment.reset(seed=seed)
            # Actual route identity is checked in the independent audit layer;
            # the learner/collector never reads hidden route contents here.
        self.cursor += 1
        if self.synthetic and self.fixture_initial_state is not None:
            self.environment.unwrapped.state = self.fixture_initial_state
            self.ordinary = self.environment.unwrapped._observation()
        self.memory = CausalMemory(state=self.environment.unwrapped.state)
        self.memory.ingest(self.ordinary)
        self.physical_mask, self.certificates, self.binding = self.treatment.mask(self.ordinary, self.memory)
        self.episode_step = 0
        self.pending_reset = False

    def prepare(self, deterministic=False):
        if self.pending_reset:
            self.reset()
        support = self.treatment.policy_support(self.physical_mask)
        proposal = self.agent.act(features(self.ordinary, self.memory), support,
                                  self.policy_rng, deterministic=deterministic)
        return dict(native=self.native, task_cursor=self.cursor - 1, task=self.task,
                    binding=self.binding, proposal=proposal,
                    actor_update=self.agent.updates,
                    policy_rng_sha256=__import__('hashlib').sha256(
                        self.policy_rng.get_state().numpy().tobytes()).hexdigest())

    def advance(self, full_evidence=False, learn=True, prepared=None, hook=None,
                deterministic=False):
        if self.native >= 50000:
            raise RuntimeError('native budget exhausted')
        prepared = self.prepare(deterministic) if prepared is None else prepared
        if prepared['native'] != self.native or prepared['binding'] != self.binding:
            raise RuntimeError('stale prepared transaction')
        hook = hook or (lambda _: None)
        m = self.memory
        ordinary = self.ordinary.copy()
        pre = m.snapshot()
        current_mask, certificates, binding = self.physical_mask, self.certificates, self.binding
        support = self.treatment.policy_support(current_mask)
        x = features(ordinary, m)
        proposal = prepared['proposal']
        updates_before = self.agent.updates
        selected, final, reason = self.treatment.execute(proposal, ordinary, m, current_mask, certificates, binding)
        action = ACTION_GRID[selected]
        next_ordinary, _, terminated, truncated, info = self.environment.step([action])
        hook('after_action')
        m.execute(action, self.treatment.shield.vehicle)
        if asdict(m.state) != asdict(self.environment.unwrapped.state):
            raise RuntimeError('native causal dynamics mismatch')
        completed = bool(m.state.position_m >= 1000)
        reward, costs = signals(info, completed, terminated, truncated, proposal, selected)
        if self.method != METHODS[0] and costs[0] > 0:
            raise RuntimeError('BANKERS_MASKED_RL_PILOT_V2_SAFETY_COUNTEREXAMPLE')
        m.ingest(next_ordinary)
        next_mask, next_certs, next_binding = self.treatment.mask(next_ordinary, m)
        next_support = self.treatment.policy_support(next_mask)
        discount = 0.0 if terminated or truncated else GAMMA_NATIVE
        insertion = self.native
        hook('before_replay')
        self.replay.append(dict(observation=x, next_observation=features(next_ordinary, m),
            action=[proposal], executed=[selected], reward=[reward], costs=costs,
            discount=[discount], mask=support, next_mask=next_support,
            binding=binding, next_binding=next_binding, insertion=insertion))
        hook('after_replay')
        row = dict(contract='bankers-training-evidence-v2', attempt=self.attempt,
            identity=self.identity, native=insertion, replay_insertion=insertion,
            task_cursor=self.cursor - 1, task=self.task, episode_step=self.episode_step,
            observation=ordinary.tolist(), memory=pre, binding=binding,
            physical_mask=current_mask, policy_mask=support,
            proposal=proposal, executed=selected, replacement_reason=reason,
            selected_certificate=final, selected_certificate_sha256=None,
            reward=reward, costs=costs, discount=discount,
            successor_observation=next_ordinary.tolist(), successor_memory=m.snapshot(),
            next_binding=next_binding, next_physical_mask=next_mask,
            next_policy_mask=next_support, actual_successor=asdict(self.environment.unwrapped.state),
            speed_excess=costs[0], signed_energy=info['step_energy_kwh'],
            actual_limit=info['speed_limit_m_s'], completed=completed,
            terminated=bool(terminated), truncated=bool(truncated), budget_truncated=False,
            certificates=certificates if full_evidence else None,
            next_certificates=next_certs if full_evidence else None)
        row['positive_acceleration_choices'] = sum(
            c['accepted'] and c['predicted_next_state']['velocity_m_s'] > pre['state']['velocity_m_s']
            for c in certificates) if certificates else None
        row['episode_metrics'] = info['episode_metrics']
        row['max_episode_steps'] = self.environment._max_episode_steps
        row['updates_before'] = updates_before
        if final:
            from benchmarks.bankers_viability_shield_v1.certificate import digest
            row['selected_certificate_sha256'] = digest(final)
        self.native += 1
        self.episode_step += 1
        self.ordinary, self.physical_mask, self.certificates, self.binding = next_ordinary, next_mask, next_certs, next_binding
        self.pending_reset = bool(terminated or truncated)
        if self.pending_reset:
            self.episodes.append(dict(task_cursor=self.cursor - 1, native=self.native,
                completed=completed, deadline=m.state.elapsed_time_s <= 140,
                metrics=info['episode_metrics'], position=m.state.position_m))
        if self.native == 50000 and not self.pending_reset:
            row['budget_truncated'] = True
        if learn:
            while self.agent.updates < update_target(self.native):
                hook('before_update')
                self.diagnostics.append(self.agent.update(self.replay.sample(self.replay_rng)))
                hook('after_update')
        row['updates_after'] = self.agent.updates
        from benchmarks.bankers_viability_shield_v1.certificate import digest
        row['prepared_identity'] = digest(prepared)
        return row
