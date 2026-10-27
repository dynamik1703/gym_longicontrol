"""Complete synthetic resource measurements and a fixed conservative model."""

import copy
import gzip
import hashlib
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v4.evaluation import run as v4_run
from benchmarks.bankers_viability_shield_v4.independent_verification import replay as v4_replay
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.domain.dynamics import advance
from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from .checkpointing import load_state, save
from .evaluation import REFERENCES, actor_entropy, mpc_run, mpc_verify
from .integrity import HERE, atomic_json, canonical_bytes, digest, file_hash, identity, read_config, verify_sources
from .journal import Journal
from .shield_treatments import METHODS
from .training import Runner
from .transaction import Engine
from .verification import Auditor


def constant_track(_):
    return make_track([0], [100])


def fixture():
    return gym.wrappers.TimeLimit(LongiControlEnv(track_generator=constant_track),1800)


def timed(call):
    wall,cpu=time.perf_counter(),time.process_time()
    value=call()
    return value,dict(wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu)


def measure(*, prior_active_upper_s=0.0, sealed_preflight_upper_s=0.0):
    config=read_config()
    _,source_check_cost=timed(lambda:verify_sources())
    start_cpu=time.process_time()
    native,diagnostic,updates,checkpoints,component_rows={},{},{},{},[]
    with tempfile.TemporaryDirectory(prefix='bankers-v2-resource-',dir='/private/tmp') as temp:
        root=Path(temp)
        for method in METHODS:
            samples=[]
            diag_samples=[]
            for speed in config['resource_probe_speeds']:
                state=VehicleState(velocity_m_s=float(speed))
                directory=root/(method+'_'+float(speed).hex())
                runner=Runner(method,'V2_RESOURCE_'+method,fixture(),True,300000,
                              fixture_initial_state=state)
                engine,init_cost=timed(lambda:Engine(directory,runner))
                rows=[]
                step_cost=[]
                for _ in range(5):
                    row,cost=timed(lambda:engine.step(learn=False))
                    rows.append(row);step_cost.append(cost)
                audit=Auditor(synthetic=True,initial_state=state)
                offline=[]
                for row in rows:
                    _,cost=timed(lambda:audit.verify(row,runner.environment.unwrapped.track,method))
                    offline.append(cost)
                engine.journal.rotate()
                _,read_cost=timed(lambda:list(engine.journal.records()))
                compressed=sum(p.stat().st_size for p in (directory/'journal').glob('*.gz'))/5
                raw=max(len(canonical_bytes(row)) for row in rows)
                sample=dict(speed=speed,engine_step=step_cost,offline_audit=offline,
                    journal_read_per_step_s=read_cost['wall_s']/5,
                    compressed_bytes_per_step=compressed,raw_row_max_bytes=raw,
                    initial_checkpoint_cost=init_cost)
                samples.append(sample)
                # Same learner and evidence implementation, deterministic policy,
                # full stored grids; complete independent Diagnostic read-back.
                dr=Runner(method,'V2_RESOURCE_DIAG_'+method,fixture(),True,300000,
                          fixture_initial_state=state)
                da=Auditor(synthetic=True,initial_state=state)
                dj=Journal(directory/'diagnostic',dr.attempt,identity())
                head=dj.genesis
                wall=time.perf_counter();cpu=time.process_time()
                diag_rows=[]
                for _ in range(5):
                    prepared=dr.prepare(deterministic=True)
                    entropy=actor_entropy(dr.agent,dr.ordinary,dr.memory,dr.treatment.policy_support(dr.physical_mask))
                    dj.intent(prepared,head)
                    row=dr.advance(prepared=prepared,full_evidence=True,learn=False)
                    row['actor_entropy']=entropy
                    da.verify(row,dr.environment.unwrapped.track,method)
                    head=dj.append(row,head);dj.clear_intent()
                    diag_rows.append(row)
                dj.rotate()
                disk=Auditor(synthetic=True,initial_state=state)
                for frame in dj.records():
                    disk.verify(frame['row'],dr.environment.unwrapped.track,method,full=True)
                diag_samples.append(dict(speed=speed,total_wall_s=time.perf_counter()-wall,
                    total_cpu_s=time.process_time()-cpu,native_steps=5,
                    compressed_bytes_per_step=sum(p.stat().st_size for p in (directory/'diagnostic').glob('*.gz'))/5,
                    raw_row_max_bytes=max(len(canonical_bytes(r)) for r in diag_rows)))
                # Component timings without changing scientific algorithms.
                _,inference=timed(lambda:runner.agent.act(
                    runner.replay.arrays['observation'][0],
                    runner.treatment.policy_support(runner.physical_mask),runner.policy_rng))
                _,mask=timed(lambda:runner.treatment.mask(runner.ordinary,runner.memory))
                proposed=runner.prepare(deterministic=True)['proposal']
                selected,selected_cost=timed(lambda:runner.treatment.execute(proposed,
                    runner.ordinary,runner.memory,runner.physical_mask,runner.certificates,runner.binding))
                from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
                action=ACTION_GRID[selected[0]]
                env_copy=copy.deepcopy(runner.environment)
                _,environment_cost=timed(lambda:env_copy.step([action]))
                _,successor_mask=timed(lambda:runner.treatment.mask(runner.ordinary,runner.memory))
                evidence_probe=Journal(directory/'evidence_io',runner.attempt,runner.identity)
                _,evidence_cost=timed(lambda:evidence_probe.append(rows[0],evidence_probe.genesis))
                _,physics_cost=timed(lambda:advance(runner.memory.state,action,
                    runner.treatment.shield.vehicle,CONFIG))
                component_rows.append(dict(method=method,speed=speed,inference=inference,
                    mask=mask,selected_final_certificate=selected_cost,native_environment=environment_cost,
                    actual_physics=physics_cost,successor_mask=successor_mask,evidence_write=evidence_cost,
                    complete_native_wall_s=max(x['wall_s'] for x in step_cost[1:]),
                    sample_grid_extra_wall_s=max(0,step_cost[0]['wall_s']-max(x['wall_s'] for x in step_cost[1:]))))
                if speed==0:
                    # Fill an explicitly synthetic tensor table to time the exact
                    # batch size and all optimizers. No policy performance is used.
                    source={k:v[0].copy() for k,v in runner.replay.arrays.items()}
                    while runner.replay.size<256:
                        item=copy.deepcopy(source)
                        for key in ('binding','next_binding'):
                            item[key]=bytes(item[key]).decode()
                        item['insertion']=runner.replay.size
                        runner.replay.append(item)
                    costs=[]
                    for _ in range(20):
                        _,cost=timed(lambda:runner.agent.update(runner.replay.sample(runner.replay_rng)))
                        costs.append(cost)
                    updates[method]=costs
                    _,sample_cost=timed(lambda:runner.replay.sample(runner.replay_rng))
                    component_rows[-1]['replay_sampling']=sample_cost
                    meta,cost=timed(lambda:save(directory/'full_state.pkl',runner,
                        audit=engine.audit,journal_head=engine.head))
                    _,restore=timed(lambda:load_state(directory/'full_state.pkl',meta['sha256'],runner.attempt))
                    checkpoints[method]=dict(metadata=meta,write=cost,restore=restore)
                shutil.rmtree(directory)
                print('RESOURCE '+method+' '+str(speed),flush=True)
            native[method]=samples;diagnostic[method]=diag_samples
        for method in REFERENCES:
            samples=[]
            for speed in config['resource_probe_speeds']:
                state=VehicleState(velocity_m_s=float(speed))
                env=fixture();path=root/(method+'_'+float(speed).hex())
                journal=Journal(path, 'V2_RESOURCE_'+method, identity());head=journal.genesis
                def sink(row):
                    nonlocal head
                    head=journal.append(dict(native=row['step'],attempt=journal.attempt,
                        identity=identity(),evidence=row),head)
                wall=time.perf_counter();cpu=time.process_time()
                if method in REFERENCES[:2]:
                    rows,result=v4_run(env,method,max_steps=5,initial_state=state,
                        attempt=journal.attempt,evidence_identity=identity(),record_sink=sink)
                    journal.rotate()
                    disk=[f['row']['evidence'] for f in journal.records()]
                    v4_replay(disk,env.unwrapped.track,method,result,complete=False,
                        initial_state=state,attempt=journal.attempt,evidence_identity=identity())
                else:
                    rows,result=mpc_run(env,seed=11,attempt=journal.attempt,record_sink=sink,
                        initial_state=state,max_steps=5)
                    journal.rotate()
                    disk=[f['row']['evidence'] for f in journal.records()]
                    mpc_verify(disk,env.unwrapped.track,initial_state=state,complete=False)
                samples.append(dict(speed=speed,total_wall_s=time.perf_counter()-wall,
                    total_cpu_s=time.process_time()-cpu,native_steps=5,
                    compressed_bytes_per_step=sum(p.stat().st_size for p in path.glob('*.gz'))/5,
                    raw_row_max_bytes=max(len(canonical_bytes(r)) for r in rows)))
                env.close();shutil.rmtree(path)
            diagnostic[method]=samples
        # Measure the actual hashing/sealing primitive on deterministic synthetic bytes.
        buffer=np.random.default_rng(11).bytes(8*1024*1024)
        path=root/'sealing.bin';path.write_bytes(buffer)
        _,hash_cost=timed(lambda:file_hash(path))
        sealing_bytes_per_s=len(buffer)/hash_cost['wall_s']
    # Fixed formula: per-arm maxima across all seven speed bands, exact matrix,
    # additive sampled-grid work, optimizer/checkpoint/recovery/evaluation/sealing.
    pieces={};storage=0;live_peak=0
    for method,samples in native.items():
        step=max(x['wall_s'] for s in samples for x in s['engine_step'][1:])
        audit=max(x['wall_s'] for s in samples for x in s['offline_audit'][1:])
        grid=max(max(0,s['engine_step'][0]['wall_s']-step) for s in samples)
        offline_grid=max(max(0,s['offline_audit'][0]['wall_s']-audit) for s in samples)
        read=max(s['journal_read_per_step_s'] for s in samples)
        update=max(x['wall_s'] for x in updates[method])
        cp=checkpoints[method]
        pieces[method]=dict(native=50000*step,offline=50000*(audit+read),
            sampled_grids=500*(grid+offline_grid),optimizer=5000*update,
            checkpoints=52*cp['write']['wall_s'],
            one_recovery=cp['restore']['wall_s']+50000*read+1000*(step+update/8))
        storage+=50000*max(s['compressed_bytes_per_step'] for s in samples)
        live_peak=max(live_peak,1000*max(s['raw_row_max_bytes'] for s in samples))
    diagnostic_work={m:12*1800*max(s['total_wall_s']/s['native_steps'] for s in ss)
                     for m,ss in diagnostic.items()}
    storage+=sum(12*1800*max(s['compressed_bytes_per_step'] for s in ss) for ss in diagnostic.values())
    live_peak=max(live_peak,100*max(s['raw_row_max_bytes'] for ss in diagnostic.values() for s in ss))
    # Three final replay-bearing checkpoints, one live technical, one atomic
    # temporary. Uncompressed sizes bound all compressed representations.
    cp_peak=5*max(c['metadata']['uncompressed_bytes'] for c in checkpoints.values())
    base_storage=storage+live_peak+cp_peak+16*1024*1024
    sealing=3*base_storage/sealing_bytes_per_s
    measured_probe_cpu=time.process_time()-start_cpu
    source_checks=250*source_check_cost['wall_s']
    base=sum(sum(p.values()) for p in pieces.values())+sum(diagnostic_work.values())+sealing+source_checks+600
    projected=config['resource_overhead_factor']*base+prior_active_upper_s+measured_probe_cpu+sealed_preflight_upper_s
    projected_storage=config['resource_overhead_factor']*base_storage
    scale_one=3*(sum(pieces[METHODS[2]].values())*6+diagnostic_work[METHODS[2]])*config['resource_overhead_factor']+sealing+source_checks*6+600
    scale_all=3*(sum(sum(p.values()) for p in pieces.values())*6+sum(diagnostic_work.values()))*config['resource_overhead_factor']+sealing+source_checks*18+600
    result=dict(identity=identity(),gate_passed=projected<=28800 and projected_storage<=2147483648,
        projected_active_compute_upper_s=projected,projected_retained_bytes=projected_storage,
        active_compute_limit_s=28800,artifact_limit_bytes=2147483648,measured_probe_cpu_s=measured_probe_cpu,
        prior_active_upper_s=prior_active_upper_s,sealed_preflight_upper_s=sealed_preflight_upper_s,training_components_s=pieces,diagnostic_s=diagnostic_work,
        sealing_s=sealing,report_allowance_s=600,checkpoint_peak_bound_bytes=cp_peak,
        source_checks_s=source_checks,source_check_measurement=source_check_cost,
        factor=config['resource_overhead_factor'],native_samples=native,diagnostic_samples=diagnostic,
        update_samples=updates,checkpoint_samples=checkpoints,component_samples=component_rows,
        scaleup_one_method='BANKER_PRE_SAC',scaleup_one_method_three_seeds_300k_s=scale_one,
        scaleup_three_methods_three_seeds_300k_s=scale_all,
        scaleup_forecast_scope='same 30k updates per 300k arm; one 12-task Diagnostic per final arm plus references; prospective only',
        formula='2*(exact per-arm maxima native+independent audit+sampled grids+15k optimizer+52 CP/arm+1 recovery/arm+72x1800 Diagnostic+sealing+600s reporting)+prior/probe/sealed-preflight cost',
        all_workloads='explicit unreserved constant100 synthetic fixtures; no held-out policy outcome')
    atomic_json(HERE/'resource_measurements.json',result,immutable=True)
    print('RESOURCE_GATE '+str(result['gate_passed'])+' '+str(projected/3600)+'h '+str(projected_storage/2**30)+'GiB',flush=True)
    return result
