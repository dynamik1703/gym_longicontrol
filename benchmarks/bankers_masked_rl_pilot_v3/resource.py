"""Prospective repeated-block profiling. Admission is a separate post-freeze action."""
import argparse
import copy
import json
import resource
import shutil
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v1.backup_policy import CausalMemory
from benchmarks.bankers_viability_shield_v4.evaluation import run as v4_run
from benchmarks.bankers_viability_shield_v4.independent_verification import replay as v4_replay
from gym_longicontrol.domain.state import VehicleState
from gym_longicontrol.envs.longicontrol import LongiControlEnv
from .checkpointing import load_state, save
from .evaluation import REFERENCES, actor_entropy, mpc_run, mpc_verify
from .integrity import HERE, ROOT, atomic_json, canonical_bytes, file_hash, identity, read_config, require, verify_sources
from .journal import Journal
from .shield_treatments import METHODS, features
from .training import Runner
from .transaction import Engine
from .verification import Auditor, equal


def constant_track(_):
    return make_track([0], [100])


def fixture():
    return gym.wrappers.TimeLimit(LongiControlEnv(track_generator=constant_track),1800)


def timed(call):
    w,c=time.perf_counter(),time.process_time()
    value=call()
    return value,dict(wall_s=time.perf_counter()-w,cpu_s=time.process_time()-c)


def stats(samples):
    return {unit:dict(n=len(samples),median=float(np.median([x[unit] for x in samples])),
        p95=float(np.percentile([x[unit] for x in samples],95)),
        p99=float(np.percentile([x[unit] for x in samples],99)),
        maximum=max(x[unit] for x in samples)) for unit in ('wall_s','cpu_s')}


def block_cost(start, n):
    return dict(wall_s=time.perf_counter()-start[0],cpu_s=time.process_time()-start[1],native_steps=n)


def instrument(obj, name, samples):
    original=getattr(obj,name)
    def wrapped(*args,**kwargs):
        value,cost=timed(lambda:original(*args,**kwargs))
        samples.append(cost)
        return value
    setattr(obj,name,wrapped)
    return lambda:setattr(obj,name,original)


def replay_equal(row,runner):
    i=row['replay_insertion']
    values=dict(observation=features(row['observation'],CausalMemory.restore(row['memory'])),
        next_observation=features(row['successor_observation'],CausalMemory.restore(row['successor_memory'])),
        action=[row['proposal']],executed=[row['executed']],reward=[row['reward']],costs=row['costs'],
        discount=[row['discount']],mask=row['policy_mask'],next_mask=row['next_policy_mask'],
        binding=row['binding'],next_binding=row['next_binding'],insertion=i)
    for k,v in values.items():
        require(np.array_equal(runner.replay.arrays[k][i],np.asarray(v,dtype=runner.replay.arrays[k].dtype)),
            'synthetic full replay verification '+k)


def measure():
    require(not (HERE/'profile_raw.json').exists(),'profiling already sealed')
    config=read_config(); n=config['resource_probe_steps']
    begin=time.perf_counter(),time.process_time()
    child_begin=resource.getrusage(resource.RUSAGE_CHILDREN)
    result=dict(identity=identity(),protocol_sha256=file_hash(HERE/'RESOURCE_PROTOCOL.md'),
        formula_commit=verify_sources(),native={},diagnostic={},components={},checkpoints={},updates={},resets={},source_checks=[])
    for _ in range(3):
        _,cost=timed(verify_sources);result['source_checks'].append(cost)
    with tempfile.TemporaryDirectory(prefix='bankers-masked-v3-profile-',dir='/private/tmp') as temp:
        temp=Path(temp)
        for method in METHODS:
            result['native'][method]=[];result['diagnostic'][method]=[]
            components={key:[] for key in ('initialization','engine_startup','cold_inference','steady_inference','banker_mask',
                'selected_certificate','environment_advance','replay_insertion','serialization','native_reconstruction')}
            result['components'][method]=components;result['resets'][method]=[]
            for repeat in range(config['resource_probe_repeats']):
                for speed in config['resource_probe_speeds']:
                    state=VehicleState(velocity_m_s=float(speed)); path=temp/(method+f'_{repeat}_{speed}')
                    runner,cost=timed(lambda:Runner(method,f'V3_PROFILE_{method}_{repeat}_{speed}',fixture(),True,300000,fixture_initial_state=state))
                    components['initialization'].append(cost)
                    engine,cost=timed(lambda:Engine(path,runner));components['engine_startup'].append(cost)
                    # A reset-only clone measures startup independently, never a numbered task.
                    clone=copy.deepcopy(runner)
                    _,cost=timed(clone.reset);result['resets'][method].append(cost)
                    _,cost=timed(lambda:clone.agent.act(features(clone.ordinary,clone.memory),clone.treatment.policy_support(clone.physical_mask),clone.policy_rng))
                    components['cold_inference'].append(cost);clone.environment.close();del clone
                    restores=[]
                    for obj,name,key in ((runner.agent,'act','steady_inference'),(runner.treatment,'mask','banker_mask'),
                        (runner.treatment,'execute','selected_certificate'),(runner.environment,'step','environment_advance'),
                        (runner.replay,'append','replay_insertion'),(engine.journal,'append','serialization'),
                        (engine.audit,'verify','native_reconstruction')):
                        restores.append(instrument(obj,name,components[key]))
                    costs=[]; rows=[];start=time.perf_counter(),time.process_time()
                    for _ in range(n):
                        row,cost=timed(lambda:engine.step(learn=False));rows.append(row);costs.append(cost)
                    engine.journal.rotate()
                    total=block_cost(start,n)
                    for undo in reversed(restores):undo()
                    raw=max(len(canonical_bytes(r)) for r in rows)
                    compressed=sum(p.stat().st_size for p in (path/'journal').glob('*.gz'))/n
                    disk=Auditor(synthetic=True,initial_state=state);offline=[]
                    start=time.perf_counter(),time.process_time()
                    for frame in engine.journal.records():
                        def verify():
                            disk.verify(frame['row'],runner.environment.unwrapped.track,method)
                            replay_equal(frame['row'],runner)
                        _,cost=timed(verify);offline.append(cost)
                    audit_total=block_cost(start,n)
                    _,read=timed(lambda:list(engine.journal.records()))
                    result['native'][method].append(dict(repeat=repeat,speed=speed,execution=total,verification=audit_total,
                        steps=costs,audits=offline,read=read,compressed_bytes_per_step=compressed,raw_row_max_bytes=raw))
                    if repeat==0 and speed==0:
                        source={k:v[0].copy() for k,v in runner.replay.arrays.items()}
                        while runner.replay.size<256:
                            item=copy.deepcopy(source)
                            for key in ('binding','next_binding'):item[key]=bytes(item[key]).decode()
                            item['insertion']=runner.replay.size;runner.replay.append(item)
                        updates=[]
                        for _ in range(64):
                            _,cost=timed(lambda:runner.agent.update(runner.replay.sample(runner.replay_rng)));updates.append(cost)
                        result['updates'][method]=updates
                        # Synthetic incompressible full-capacity arrays conservatively
                        # exercise checkpoint compression/I/O without any task outcome.
                        noise=np.random.default_rng(1100)
                        for array in runner.replay.arrays.values():
                            array.view(np.uint8).reshape(-1)[:]=np.frombuffer(noise.bytes(array.nbytes),dtype=np.uint8)
                        cp=[]
                        for _ in range(3):
                            meta,write=timed(lambda:save(path/'full.pkl',runner,audit=engine.audit,journal_head=engine.head))
                            _,restore=timed(lambda:load_state(path/'full.pkl',meta['sha256'],runner.attempt))
                            cp.append(dict(metadata=meta,write=write,restore=restore))
                        result['checkpoints'][method]=cp
                    runner.environment.close();del runner,engine,rows
                    # Exact learned Diagnostic path, with all-state stored certificates.
                    dr=Runner(method,f'V3_PROFILE_DIAG_{method}_{repeat}_{speed}',fixture(),True,1800,fixture_initial_state=state)
                    audit=Auditor(synthetic=True,initial_state=state)
                    journal=Journal(path/'diag',dr.attempt,identity());head=journal.genesis
                    diag_rows=[];step_costs=[];start=time.perf_counter(),time.process_time()
                    for _ in range(n):
                        w,c=time.perf_counter(),time.process_time()
                        prepared=dr.prepare(deterministic=True)
                        entropy=actor_entropy(dr.agent,dr.ordinary,dr.memory,dr.treatment.policy_support(dr.physical_mask))
                        journal.intent(prepared,head)
                        row=dr.advance(prepared=prepared,full_evidence=True,learn=False);row['actor_entropy']=entropy
                        audit.verify(row,dr.environment.unwrapped.track,method)
                        head=journal.append(row,head);journal.clear_intent();diag_rows.append(row)
                        if dr.native%100==0:journal.rotate()
                        step_costs.append(dict(wall_s=time.perf_counter()-w,cpu_s=time.process_time()-c))
                    journal.rotate();execution=block_cost(start,n)
                    full=Auditor(synthetic=True,initial_state=state);costs=[]
                    start=time.perf_counter(),time.process_time()
                    for frame in journal.records():
                        r=frame['row']
                        def verify_diag():
                            full.verify(r,dr.environment.unwrapped.track,method,full=True)
                            m=CausalMemory.restore(r['memory'])
                            equal(dr.agent.act(features(r['observation'],m),r['policy_mask'],dr.policy_rng,deterministic=True),r['proposal'],'synthetic actor law')
                            equal(actor_entropy(dr.agent,r['observation'],m,r['policy_mask']),r['actor_entropy'],'synthetic entropy')
                        _,cost=timed(verify_diag);costs.append(cost)
                    verification=block_cost(start,n)
                    result['diagnostic'][method].append(dict(repeat=repeat,speed=speed,execution=execution,verification=verification,
                        steps=step_costs,audits=costs,compressed_bytes_per_step=sum(p.stat().st_size for p in (path/'diag').glob('*.gz'))/n,
                        raw_row_max_bytes=max(len(canonical_bytes(r)) for r in diag_rows)))
                    dr.environment.close();del dr,diag_rows
                    shutil.rmtree(path)
                    print(json.dumps(dict(profile=method,repeat=repeat,speed=speed,steps=n)),flush=True)
        for method in REFERENCES:
            result['diagnostic'][method]=[]
            for repeat in range(config['resource_probe_repeats']):
                for speed in config['resource_probe_speeds']:
                    path=temp/(method+f'_{repeat}_{speed}');env=fixture();state=VehicleState(velocity_m_s=float(speed))
                    journal=Journal(path,f'V3_PROFILE_{method}_{repeat}_{speed}',identity());head=journal.genesis
                    step_costs=[];last=[time.perf_counter(),time.process_time()]
                    def sink(row):
                        nonlocal head
                        head=journal.append(dict(native=row['step'],attempt=journal.attempt,identity=identity(),evidence=row),head)
                        if (row['step']+1)%100==0:journal.rotate()
                        step_costs.append(dict(wall_s=time.perf_counter()-last[0],cpu_s=time.process_time()-last[1]))
                        last[:]=time.perf_counter(),time.process_time()
                    start=time.perf_counter(),time.process_time();last[:]=start
                    if method in REFERENCES[:2]:
                        rows,out=v4_run(env,method,max_steps=n,initial_state=state,attempt=journal.attempt,evidence_identity=identity(),record_sink=sink)
                    else:
                        rows,out=mpc_run(env,seed=11,attempt=journal.attempt,record_sink=sink,initial_state=state,max_steps=n)
                    journal.rotate();execution=block_cost(start,n)
                    start=time.perf_counter(),time.process_time()
                    disk=[f['row']['evidence'] for f in journal.records()]
                    if method in REFERENCES[:2]:
                        v4_replay(disk,env.unwrapped.track,method,out,complete=False,initial_state=state,attempt=journal.attempt,evidence_identity=identity())
                    else:mpc_verify(disk,env.unwrapped.track,initial_state=state,complete=False,max_steps=n)
                    verification=block_cost(start,n)
                    result['diagnostic'][method].append(dict(repeat=repeat,speed=speed,execution=execution,verification=verification,
                        steps=step_costs,audits=[],compressed_bytes_per_step=sum(p.stat().st_size for p in path.glob('*.gz'))/n,
                        raw_row_max_bytes=max(len(canonical_bytes(r)) for r in rows)))
                    env.close();shutil.rmtree(path)
                    print(json.dumps(dict(profile=method,repeat=repeat,speed=speed,steps=n)),flush=True)
        data=np.random.default_rng(11).bytes(8*1024*1024);p=temp/'hash.bin';p.write_bytes(data)
        result['hash_samples']=[]
        for _ in range(3):
            _,cost=timed(lambda:file_hash(p));result['hash_samples'].append(cost)
    result['profile_wall_s']=time.perf_counter()-begin[0]
    child_end=resource.getrusage(resource.RUSAGE_CHILDREN)
    result['profile_self_cpu_s']=time.process_time()-begin[1]
    result['profile_child_cpu_s']=child_end.ru_utime+child_end.ru_stime-child_begin.ru_utime-child_begin.ru_stime
    result['profile_cpu_s']=result['profile_self_cpu_s']+result['profile_child_cpu_s']
    result['component_statistics']={m:{k:stats(v) for k,v in cc.items()} for m,cc in result['components'].items()}
    result['band_statistics']={kind:{m:[dict(repeat=s['repeat'],speed=s['speed'],steps=stats(s['steps']),
        audits=stats(s['audits']) if s['audits'] else None) for s in samples] for m,samples in result[kind].items()} for kind in ('native','diagnostic')}
    atomic_json(HERE/'profile_raw.json',result,immutable=True)
    print('PROFILE_SEALED',flush=True)
    return result


def projection(raw, engineering, current_bytes):
    pieces={};storage=0;peak=0;stress=0
    def rate(samples,key):return max(s[key]['wall_s']/s[key]['native_steps'] for s in samples)
    for method,samples in raw['native'].items():
        native=rate(samples,'execution');audit=rate(samples,'verification')
        update=max(s['wall_s'] for s in raw['updates'][method]);cp=raw['checkpoints'][method]
        reset=max(s['wall_s'] for s in raw['resets'][method])
        init=max(s['wall_s'] for s in raw['components'][method]['initialization'])
        startup=max(s['wall_s'] for s in raw['components'][method]['engine_startup'])
        read=max(s['read']['wall_s']/s['execution']['native_steps'] for s in samples)
        pieces[method]=dict(native=50000*native,independent_audit=50000*audit,optimizer=5000*update,
            checkpoints=52*max(s['write']['wall_s'] for s in cp),startup=init+startup,
            resets=1000*reset,recovery=max(s['restore']['wall_s'] for s in cp)+50000*read+1000*(native+update/8))
        stress+=50000*(max(x['wall_s'] for s in samples for x in s['steps'])+max(x['wall_s'] for s in samples for x in s['audits'])+read)+sum(v for k,v in pieces[method].items() if k not in ('native','independent_audit'))
        storage+=50000*max(s['compressed_bytes_per_step'] for s in samples)
        peak=max(peak,1000*max(s['raw_row_max_bytes'] for s in samples))
    diagnostic={}
    for method,samples in raw['diagnostic'].items():
        diagnostic[method]=21600*(rate(samples,'execution')+rate(samples,'verification'))
        stress+=21600*(max(x['wall_s'] for s in samples for x in s['steps'])+rate(samples,'verification'))
        storage+=21600*max(s['compressed_bytes_per_step'] for s in samples)
        peak=max(peak,100*max(s['raw_row_max_bytes'] for s in samples))
    cp_peak=5*max(s['metadata']['uncompressed_bytes'] for ss in raw['checkpoints'].values() for s in ss)
    # gzip publication retains the live journal and its compressed segment concurrently.
    base_storage=storage+2*peak+cp_peak+32*1024*1024
    hash_rate=8*1024*1024/max(s['wall_s'] for s in raw['hash_samples'])
    overhead=250*max(s['wall_s'] for s in raw['source_checks'])+3*base_storage/hash_rate+600
    prior=120+engineering['active_cpu_s']+raw['profile_cpu_s']
    preflight=2*engineering['wall_s']
    upper=2*(sum(sum(p.values()) for p in pieces.values())+sum(diagnostic.values())+overhead)+prior+preflight
    retained=2*base_storage+current_bytes
    return dict(gate_passed=upper<=86400 and retained<=3*2**30,projected_active_compute_upper_s=upper,
        projected_retained_bytes=retained,active_compute_limit_s=86400,artifact_limit_bytes=3*2**30,
        training_components_s=pieces,diagnostic_s=diagnostic,prior_active_upper_s=120+engineering['active_cpu_s'],
        measured_probe_cpu_s=raw['profile_cpu_s'],sealed_preflight_upper_s=preflight,
        checkpoint_peak_bound_bytes=cp_peak,metadata_and_sealing_s=overhead,individual_maximum_stress_s=2*(stress+overhead)+prior+preflight,
        factor=2,formula='max repeated complete block wall rate; exact matrix; 2x future work; prior+probe+2x sealed preflight wall',
        scaleup_one_method_three_seeds_300k_s=2*(18*sum(pieces[METHODS[2]].values())+3*diagnostic[METHODS[2]]+sum(diagnostic[r] for r in REFERENCES)+overhead)+prior,
        scaleup_three_methods_three_seeds_300k_s=2*(18*sum(sum(v.values()) for v in pieces.values())+3*sum(diagnostic[m] for m in METHODS)+sum(diagnostic[r] for r in REFERENCES)+overhead)+prior)


def admit():
    freeze=json.loads((HERE/'freeze_manifest.json').read_text());verify_sources(freeze['identity'],committed=True)
    raw=json.loads((HERE/'profile_raw.json').read_text());require(raw['identity']==identity(),'profile scientific identity changed')
    engineering=json.loads((HERE/'engineering_results.json').read_text())
    require(engineering['exit_code']==0 and engineering['failures']==0 and engineering['errors']==0,'engineering closed')
    result=projection(raw,engineering,sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
    result.update(identity=identity(),freeze_commit=freeze['freeze_commit'],profile_sha256=file_hash(HERE/'profile_raw.json'),
        protocol_sha256=file_hash(HERE/'RESOURCE_PROTOCOL.md'))
    atomic_json(HERE/'resource_measurements.json',result,immutable=True)
    if not result['gate_passed']:
        atomic_json(HERE/'scientific_stop.json',dict(classification='BANKERS_MASKED_RL_PILOT_V3_RESOURCE_STOP',
            phase='SEALED_RESOURCE_GATE',identity=identity()),immutable=True)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('measure','admit'));args=parser.parse_args()
    measure() if args.phase=='measure' else admit()
