import ast
import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.bankers_masked_rl_pilot_v2.integrity import HERE, ROOT, guard_panel, guard_seed
from benchmarks.bankers_masked_rl_pilot_v2.shield_treatments import METHODS
from benchmarks.bankers_masked_rl_pilot_v2.transaction import Engine
from benchmarks.bankers_masked_rl_pilot_v2.evaluation import mpc_run, mpc_verify
from test_verifier import context


def test_learner_and_treatment_bytes_unchanged():
    old=ROOT/'benchmarks/bankers_masked_rl_pilot_v1'
    for name in ('sac_agent.py','masked_distribution.py','replay.py'):
        assert (HERE/name).read_bytes()==(old/name).read_bytes()


def test_historical_content_guards():
    for n in range(4000,4018):
        with pytest.raises(Exception): guard_seed(n)
    for panel in ('HYBRID_RL_VALIDATION_V1','VALIDATION','PAPER','BANKERS_MASKED_RL_PILOT_DIAGNOSTIC_V1'):
        with pytest.raises(Exception): guard_panel(panel)


@pytest.mark.parametrize('method',METHODS)
def test_real_process_crash_after_action(method,tmp_path):
    runner,_=context(method,[0],[100],max_steps=2,more_tracks=[([0],[20])])
    directory=tmp_path/'run'
    engine=Engine(directory,runner,checkpoint_interval=2)
    engine.step(learn=False)
    expected=copy.deepcopy(engine.runner)
    expected_row=expected.advance(learn=False)
    code='''
import os,sys
from benchmarks.bankers_masked_rl_pilot_v2.transaction import Engine
engine=Engine.recover(sys.argv[1],sys.argv[2],checkpoint_interval=2,learn=False)
def crash(stage):
    if stage=='after_action': os._exit(73)
engine.step(hook=crash,learn=False)
'''
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT/'src'),str(ROOT),str(HERE/'tests')]))
    result=subprocess.run([sys.executable,'-c',code,str(directory),runner.attempt],env=env)
    assert result.returncode==73
    restored=Engine.recover(directory,runner.attempt,checkpoint_interval=2,learn=False)
    assert restored.runner.native==2 and restored.runner.pending_reset
    from benchmarks.bankers_masked_rl_pilot_v2.integrity import canonical_bytes
    row=list(restored.journal.records())[-1]['row']
    assert canonical_bytes(row)==canonical_bytes(expected_row)
    # Recovery cannot duplicate the action/replay IDs.
    assert restored.runner.replay.cursor==2 and restored.audit.rows==2


def test_frozen_mpc_native_independent_replay():
    runner,_=context(METHODS[0],[0],[100])
    rows,result=mpc_run(runner.environment,seed=11,attempt='V2_SYNTHETIC_MPC',record_sink=lambda _:None,max_steps=3)
    verified=mpc_verify(rows,runner.environment.unwrapped.track,complete=False)
    assert verified['native_steps']==3
    corrupted=copy.deepcopy(rows)
    corrupted[0]['signed_energy']+=0.1
    with pytest.raises(Exception):
        mpc_verify(corrupted,runner.environment.unwrapped.track,complete=False)


def test_no_privileged_actor_interface():
    # The unchanged feature function has only the observation and causal-memory
    # arguments; actor invocation never receives route/task/seed objects.
    from benchmarks.bankers_masked_rl_pilot_v2.shield_treatments import features
    import inspect
    assert list(inspect.signature(features).parameters)==['ordinary','memory']
    tree=ast.parse((HERE/'shield_treatments.py').read_text())
    assert not any(isinstance(n,ast.Name) and n.id in ('track','route_seed') for n in ast.walk(tree))
