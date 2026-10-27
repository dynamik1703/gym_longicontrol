"""Scientific entry points: frozen gates, immutable Attempts and final seals."""

import argparse
import json
import time
import traceback

from .budget import Budget
from .checkpointing import load_state, save
from .evaluation import ARMS, pair
from .integrity import HERE, actor_hash, atomic_json, file_hash, fsync_directory, identity, owner, require, verify_sources
from .shield_treatments import METHODS
from .training import Runner
from .transaction import Engine
from .verification import audit_training_disk


def ready():
    freeze = json.loads((HERE / 'freeze_manifest.json').read_text())
    verify_sources(freeze['identity'], committed=True)
    gate = json.loads((HERE / 'sealed_preflight.json').read_text())
    resource = json.loads((HERE / 'resource_measurements.json').read_text())
    require(gate['all_gates_passed'] and gate['identity'] == identity(), 'sealed preflight closed')
    require(resource['gate_passed'], 'resource gate closed')
    require(not (HERE / 'scientific_stop.json').exists(), 'scientific stop is permanent')
    return freeze


def train_all(budget, resume=False):
    ready()
    results = {}
    for method in METHODS:
        budget.check(force=True)
        directory = HERE / 'training_runs' / method
        completed = directory / 'COMPLETED.json'
        if completed.exists():
            value = json.loads(completed.read_text())
            require(file_hash(directory / 'final_50000.pkl') == value['checkpoint']['sha256'],
                    'completed final checkpoint changed')
            results[method] = value
            continue
        attempt = f'BANKERS_MASKED_RL_PILOT_V2__{method}__SEED_11__ATTEMPT_01'
        if (directory / 'STARTED.json').exists():
            require(resume, 'partial Attempt requires verified same-Attempt recovery')
            engine = Engine.recover(directory, attempt)
        else:
            engine = Engine(directory, Runner(method, attempt))
        start = time.process_time()
        while engine.runner.native < 50000:
            budget.check()
            try:
                engine.step()
            except BaseException as error:
                # Preserve exact partially advanced state for forensics; this
                # emergency record is never advertised as a resumable checkpoint.
                emergency=save(directory/'FAILED_EVENT_STATE.pkl',engine.runner,
                    immutable=True,audit=engine.audit,journal_head=engine.head)
                atomic_json(directory/'FAILED_EVENT.json',dict(error=repr(error),
                    native_counter=engine.runner.native,replay_cursor=engine.runner.replay.cursor,
                    updates=engine.runner.agent.updates,emergency=emergency,
                    recovery_allowed=False),immutable=True)
                raise
            if engine.runner.native % 1000 == 0:
                verify_sources(identity())
                budget.save()
                print(json.dumps(dict(method=method, native=engine.runner.native,
                    updates=engine.runner.agent.updates, cpu_s=time.process_time()-start)), flush=True)
        require(engine.runner.agent.updates == 5000, 'final scheduled update count')
        engine.journal.rotate()
        final_path=directory/'final_50000.pkl'
        if final_path.exists():
            existing=load_state(final_path,None,attempt)
            require(existing['runner'].native==50000 and existing['runner'].agent.updates==5000
                and existing['journal_head']==engine.head
                and actor_hash(existing['runner'].agent)==actor_hash(engine.runner.agent),
                'ambiguous interrupted final checkpoint seal')
            checkpoint=dict(sha256=file_hash(final_path),bytes=final_path.stat().st_size,
                native=50000,updates=5000,attempt=attempt)
        else:
            checkpoint = save(final_path, engine.runner, immutable=True,
                              audit=engine.audit, journal_head=engine.head)
        restored = load_state(directory / 'final_50000.pkl', checkpoint['sha256'], attempt)
        verification = audit_training_disk(directory, restored['runner'])
        require(verification['journal_head'] == engine.head, 'final evidence head')
        value = dict(method=method, attempt=attempt, identity=identity(), native=50000, updates=5000,
            checkpoint=checkpoint, actor_sha256=actor_hash(restored['runner'].agent),
            verification=verification, tasks_exposed=engine.runner.cursor,
            completed_episodes=engine.runner.episodes, optimizer_diagnostics=engine.runner.diagnostics,
            final_partial_episode=dict(pending_reset=engine.runner.pending_reset,
                task_cursor=engine.runner.cursor-1, episode_step=engine.runner.episode_step,
                memory=engine.runner.memory.snapshot()),
            evidence_files={p.name:file_hash(p) for p in sorted((directory/'journal').glob('*.gz'))})
        atomic_json(completed, value, immutable=True)
        (directory / 'technical.pkl').unlink()
        fsync_directory(directory)
        results[method] = value
        print(json.dumps(dict(method=method, state='COMPLETED', native=50000, updates=5000)), flush=True)
    return results


def diagnose_all(budget):
    ready()
    from .split import build
    split_path = HERE / 'split_ledger.json'
    panel = json.loads(split_path.read_text()) if split_path.exists() else build()
    results = []
    for method in ARMS:
        agent = checkpoint_sha = None
        if method in METHODS:
            directory = HERE / 'training_runs' / method
            record = json.loads((directory / 'COMPLETED.json').read_text())
            checkpoint_sha = record['checkpoint']['sha256']
            saved = load_state(directory / 'final_50000.pkl', checkpoint_sha, record['attempt'])
            agent = saved['runner'].agent
            require(actor_hash(agent) == record['actor_sha256'], 'final actor identity')
        for task in panel['ordered_tasks']:
            budget.check(force=True)
            attempt = f'{panel["panel"]}__{method}__{task["numeric_seed"]}__ATTEMPT_01'
            completed = HERE / 'diagnostic' / attempt / 'COMPLETED.json'
            if completed.exists():
                row = json.loads(completed.read_text())
                require(row['identity'] == identity() and row['panel_sha256'] == panel['ordered_sha256'],
                        'sealed Diagnostic identity')
            else:
                row = pair(method, task, panel['ordered_sha256'], agent=agent,
                           checkpoint_sha256=checkpoint_sha, budget=budget.check)
            results.append(row)
            budget.save()
            print(json.dumps(dict(method=method, seed=task['numeric_seed'],
                state='COMPLETED', feasible=row['feasible'], completed_pairs=len(results))), flush=True)
    require(len(results) == 72 and len({r['attempt'] for r in results}) == 72,
            'unique Diagnostic denominator')
    atomic_json(HERE / 'diagnostic_results.json', dict(panel=panel['panel'],
        ordered_sha256=panel['ordered_sha256'], results=results), immutable=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'diagnostic', 'all'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with owner(recover_stale=args.resume):
        budget = Budget()
        try:
            ready()
            if args.phase in ('train', 'all'):
                train_all(budget, resume=args.resume)
            if args.phase in ('diagnostic', 'all'):
                diagnose_all(budget)
        except BaseException as error:
            label = ('BANKERS_MASKED_RL_PILOT_V2_RESOURCE_STOP' if 'RESOURCE_STOP' in str(error)
                else 'BANKERS_MASKED_RL_PILOT_V2_SAFETY_COUNTEREXAMPLE' if 'SAFETY_COUNTEREXAMPLE' in str(error)
                else 'BANKERS_MASKED_RL_PILOT_V2_INTEGRITY_STOP')
            atomic_json(HERE / 'scientific_stop.json', dict(classification=label,
                error=repr(error), traceback=traceback.format_exc(), identity=identity()), immutable=True)
            raise
        finally:
            budget.close()


if __name__ == '__main__':
    main()
