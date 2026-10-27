"""Exactly-once full-evidence Diagnostic with unchanged physical references."""

import copy
import json
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch

from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG, CausalMemory
from benchmarks.bankers_viability_shield_v2.independent_verification import verify_envelope
from benchmarks.bankers_viability_shield_v4.evaluation import run as v4_run
from benchmarks.bankers_viability_shield_v4.independent_verification import replay as v4_replay
from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.causal_mpc_v2.controller import PriceMPCController
from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import _EpisodeMetricsAccumulator
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel
from gym_longicontrol.envs.longicontrol import StochasticTrack

from .integrity import HERE, PANEL, atomic_json, canonical_bytes, digest, file_hash, guard_panel, guard_seed, identity, require
from .journal import Journal
from .masked_distribution import distribution
from .shield_treatments import METHODS, features
from .training import Runner
from .verification import Auditor, equal, check_task_identity

REFERENCES = ('CAUSAL60', 'BANKERS_MASKED_CAUSAL60', 'ADAPTIVE_PRICE_MPC_V2')
ARMS = (*METHODS, *REFERENCES)


class EvaluationRunner(Runner):
    def __init__(self, method, attempt, agent, task, environment=None):
        super().__init__(method, attempt, environment, True, 1800)
        self.agent = agent
        self.evaluation_task = task

    def reset(self):
        seed = self.evaluation_task['numeric_seed']
        guard_seed(seed)
        self.ordinary, _ = self.environment.reset(seed=seed)
        check_task_identity(self.environment.unwrapped.track, self.evaluation_task)
        self.task = self.evaluation_task
        self.cursor += 1
        self.memory = CausalMemory()
        self.memory.ingest(self.ordinary)
        self.physical_mask, self.certificates, self.binding = self.treatment.mask(self.ordinary, self.memory)
        self.episode_step = 0
        self.pending_reset = False


def actor_entropy(agent, ordinary, memory, support):
    with torch.no_grad():
        logits = agent.actor(torch.as_tensor(features(ordinary, memory)))
        return distribution(logits, torch.as_tensor(support, dtype=torch.bool))[2].item()


def mpc_run(environment, *, seed, attempt, record_sink, initial_state=None, max_steps=1800):
    controller = PriceMPCController()
    selector = AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
    controller.reset()
    selector.reset()
    ordinary, _ = environment.reset(seed=seed)
    initial = initial_state or VehicleState()
    if initial_state is not None:
        environment.unwrapped.state = initial
        ordinary = environment.unwrapped._observation()
        controller.reference.state = initial
    memory = CausalMemory(state=initial)
    price = 1.0
    rows = []
    for step in range(max_steps):
        memory.ingest(ordinary)
        pre = memory.snapshot()
        if step % 10 == 0:
            price = selector.action(controller.prepare_strategy_step(ordinary))
        action = controller.action(ordinary, price)
        decision = asdict(controller.last_decision)
        successor, _, terminated, truncated, info = environment.step([action])
        row = dict(step=step, attempt=attempt, identity=identity(), observation=ordinary.tolist(),
            memory=pre, price=price, decision=decision, action=action,
            actual_successor=asdict(environment.unwrapped.state), signed_energy=info['step_energy_kwh'],
            actual_limit=info['speed_limit_m_s'], speed_excess=info['speed_excess_m_s'],
            episode_metrics=info['episode_metrics'], terminated=bool(terminated), truncated=bool(truncated))
        record_sink(row)
        rows.append(row)
        controller.observe_executed_action(action)
        memory.execute(action, controller.vehicle)
        ordinary = successor
        if terminated or truncated:
            break
    return rows, dict(steps=len(rows), metrics=info['episode_metrics'],
                     actual_progress_m=memory.state.position_m)


def mpc_verify(rows, track, *, initial_state=None, complete=True, max_steps=1800):
    vehicle = VehicleModel()
    memory = CausalMemory(state=initial_state or VehicleState())
    controller = PriceMPCController()
    controller.reset()
    controller.reference.state = memory.state
    selector = AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
    metrics = _EpisodeMetricsAccumulator()
    price = 1.0
    for i, row in enumerate(rows):
        require(row['step'] == i and row['identity'] == identity(), 'MPC row identity')
        previous = copy.deepcopy(memory)
        ordinary = observation(memory.state, track.sense(memory.state.position_m, 150),
                               vehicle.specs, CONFIG, 1.0)
        memory.ingest(ordinary)
        verify_envelope(memory, previous, track)
        equal(memory.snapshot(), row['memory'], 'MPC independent causal memory')
        equal(ordinary.tolist(), row['observation'], 'MPC actual observation')
        if i % 10 == 0:
            price = selector.action(controller.prepare_strategy_step(ordinary))
        action = controller.action(ordinary, price)
        equal(action, row['action'], 'frozen AdaptivePrice MPC law')
        equal(price, row['price'], 'frozen adaptive price')
        equal(asdict(controller.last_decision), row['decision'], 'MPC full physical/deadline decision')
        actual, _, energy = advance(memory.state, action, vehicle, CONFIG)
        equal(asdict(actual), row['actual_successor'], 'MPC actual native dynamics')
        equal(energy, row['signed_energy'], 'MPC signed energy')
        limit = track.sense(actual.position_m, 150).current_limit_m_s
        equal(limit, row['actual_limit'], 'MPC actual limit')
        equal(max(0.0, actual.velocity_m_s - limit), row['speed_excess'], 'MPC exact speed')
        metrics.update(velocity_m_s=actual.velocity_m_s, speed_limit_m_s=limit, dt_s=0.1)
        terminated, truncated = actual.position_m >= 1000, i + 1 == max_steps
        equal(terminated, row['terminated'], 'MPC termination')
        equal(truncated, row['truncated'], 'MPC TimeLimit')
        equal(asdict(metrics.snapshot(actual, completed=bool(terminated))), row['episode_metrics'], 'MPC metrics')
        controller.observe_executed_action(action)
        memory.execute(action, vehicle)
        if terminated or truncated:
            require(i == len(rows) - 1, 'MPC evidence after end')
    require(not complete or row['terminated'] or row['truncated'], 'incomplete MPC episode')
    return dict(native_steps=len(rows), complete_native=True, controller_recomputed=True)


def summary(metrics, progress, stats):
    completed = metrics['completed']
    deadline = completed and metrics['travel_time_s'] <= 140.0
    speed = metrics['max_speed_violation_m_s'] == 0.0
    failures = ([] if completed else ['INCOMPLETE']) + ([] if deadline else ['DEADLINE']) + ([] if speed else ['SPEED'])
    return dict(metrics=metrics, progress_m=progress, completed=completed, deadline=deadline,
        speed=speed, feasible=completed and deadline and speed, failure_reasons=failures, **stats)


def pair(method, task, panel_hash, *, agent=None, checkpoint_sha256=None, budget=None):
    guard_panel(PANEL)
    guard_seed(task['numeric_seed'])
    from .execution import ready
    ready()
    frozen_panel = json.loads((HERE / 'split_ledger.json').read_text())
    require(frozen_panel['panel'] == PANEL and frozen_panel['ordered_sha256'] == panel_hash
            and task in frozen_panel['ordered_tasks'], 'task outside frozen V2 Diagnostic')
    attempt = f'{PANEL}__{method}__{task["numeric_seed"]}__ATTEMPT_01'
    directory = HERE / 'diagnostic' / attempt
    directory.mkdir(parents=True, exist_ok=True)
    require(not (directory / 'STARTED.json').exists(), 'Diagnostic pair rerun/ambiguous partial forbidden')
    atomic_json(directory / 'STARTED.json', dict(attempt=attempt, method=method,
        task=task, panel_sha256=panel_hash, identity=identity(), checkpoint_sha256=checkpoint_sha256), immutable=True)
    journal = Journal(directory / 'journal', attempt, identity())
    head = journal.genesis

    def wrapped_sink(row, kind):
        nonlocal head
        if budget:
            budget()
        wrapper = dict(native=row['step'], attempt=attempt, identity=identity(), kind=kind, evidence=row)
        head = journal.append(wrapper, head)
        if (row['step'] + 1) % 100 == 0:
            journal.rotate()

    environment = gym.wrappers.TimeLimit(StochasticTrack(), 1800)
    if method in METHODS:
        require(agent is not None, 'missing final 50k policy')
        runner = EvaluationRunner(method, attempt, agent, task, environment)
        audit = Auditor(synthetic=True, expected_identity=identity())
        mismatch = positive = mask_size = entropy_sum = excess_sum = 0.0
        action_counts = [0] * 21
        while True:
            if budget:
                budget()
            prepared = runner.prepare(deterministic=True)
            entropy = actor_entropy(agent, runner.ordinary, runner.memory,
                                    runner.treatment.policy_support(runner.physical_mask))
            journal.intent(prepared, head)
            row = runner.advance(prepared=prepared, full_evidence=True, learn=False)
            row['actor_entropy'] = entropy
            audit.verify(row, environment.unwrapped.track, method, full=False)
            head = journal.append(row, head)
            journal.clear_intent()
            mismatch += row['proposal'] != row['executed']
            mask_size += sum(row['physical_mask'])
            positive += row['positive_acceleration_choices'] or 0
            entropy_sum += entropy
            excess_sum += row['speed_excess']
            action_counts[row['executed']] += 1
            if runner.native % 100 == 0:
                journal.rotate()
            if row['terminated'] or row['truncated']:
                break
        journal.rotate()
        # Independent disk read-back, including exact frozen actor proposals and
        # every unique current/successor candidate grid, before pair completion.
        disk_audit = Auditor(synthetic=True, expected_identity=identity())
        for frame in journal.records():
            r = frame['row']
            disk_audit.verify(r, environment.unwrapped.track, method, full=True)
            m = CausalMemory.restore(r['memory'])
            equal(agent.act(features(r['observation'], m), r['policy_mask'],
                            runner.policy_rng, deterministic=True), r['proposal'], 'frozen Diagnostic actor law')
            equal(actor_entropy(agent, r['observation'], m, r['policy_mask']), r['actor_entropy'], 'actor entropy')
        n = runner.native
        result = summary(row['episode_metrics'], runner.memory.state.position_m,
            dict(native_steps=n, mismatches=int(mismatch), correction_fraction=mismatch/n,
                mask_size_mean=mask_size/n, positive_acceleration_choices_mean=positive/n,
                actor_entropy_mean=entropy_sum/n, mean_speed_excess=excess_sum/n,
                action_counts=action_counts, verification=disk_audit.report()))
    elif method in REFERENCES[:2]:
        rows, result0 = v4_run(environment, method, seed=task['numeric_seed'],
            evidence_identity=identity(), attempt=attempt,
            record_sink=lambda row: wrapped_sink(row, 'V4_REFERENCE'), budget=budget)
        check_task_identity(environment.unwrapped.track, task)
        journal.rotate()
        disk = [f['row']['evidence'] for f in journal.records()]
        v4_replay(disk, environment.unwrapped.track, method, result0,
                  evidence_identity=identity(), attempt=attempt)
        n = len(rows)
        mismatch = sum(r['proposal_action'] != r['executed_action'] for r in rows)
        result = summary(result0['metrics'], result0['actual_progress_m'], dict(native_steps=n,
            mismatches=mismatch, correction_fraction=mismatch/n,
            mask_size_mean=float(np.mean([sum(r['safe_action_mask']) for r in rows])) if method == REFERENCES[1] else None,
            positive_acceleration_choices_mean=float(np.mean([r['authority']['positive_acceleration_choices'] for r in rows])) if method == REFERENCES[1] else None,
            actor_entropy_mean=None, mean_speed_excess=float(np.mean([r['speed_excess_m_s'] for r in rows])),
            action_counts=None, verification=dict(native_steps=n, grid_exhaustive=method == REFERENCES[1], complete_native=True)))
    elif method == REFERENCES[2]:
        rows, result0 = mpc_run(environment, seed=task['numeric_seed'], attempt=attempt,
            record_sink=lambda row: wrapped_sink(row, 'MPC_REFERENCE'))
        check_task_identity(environment.unwrapped.track, task)
        journal.rotate()
        disk = [f['row']['evidence'] for f in journal.records()]
        verification = mpc_verify(disk, environment.unwrapped.track)
        n = len(rows)
        mismatch = sum(r['decision']['overridden'] for r in rows)
        result = summary(result0['metrics'], result0['actual_progress_m'], dict(native_steps=n,
            mismatches=mismatch, correction_fraction=mismatch/n, mask_size_mean=None,
            positive_acceleration_choices_mean=None, actor_entropy_mean=None,
            mean_speed_excess=float(np.mean([r['speed_excess'] for r in rows])),
            action_counts=None, verification=verification))
    else:
        raise ValueError('unknown Diagnostic arm')
    environment.close()
    result.update(attempt=attempt, method=method, task=task, panel_sha256=panel_hash,
                  identity=identity(), checkpoint_sha256=checkpoint_sha256, journal_head=head)
    result['evidence_files'] = {p.name: file_hash(p) for p in sorted((directory / 'journal').glob('*.gz'))}
    atomic_json(directory / 'COMPLETED.json', result, immutable=True)
    return result
