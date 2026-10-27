import copy
import pytest
from benchmarks.bankers_masked_rl_pilot_v3 import integrity
from benchmarks.bankers_masked_rl_pilot_v3.resource import projection, stats
from benchmarks.bankers_masked_rl_pilot_v3.shield_treatments import METHODS
from benchmarks.bankers_masked_rl_pilot_v3.evaluation import ARMS

def raw_fixture():
    cost=dict(wall_s=.001,cpu_s=.0008)
    sample=dict(execution=dict(wall_s=.128,native_steps=128),verification=dict(wall_s=.128,native_steps=128),
        steps=[cost],audits=[cost],read=dict(wall_s=.01),compressed_bytes_per_step=100,raw_row_max_bytes=1000)
    return dict(native={m:[copy.deepcopy(sample)] for m in METHODS},diagnostic={m:[copy.deepcopy(sample)] for m in ARMS},
        updates={m:[cost] for m in METHODS},resets={m:[cost] for m in METHODS},
        components={m:dict(initialization=[cost],engine_startup=[cost]) for m in METHODS},
        checkpoints={m:[dict(write=cost,restore=cost,metadata=dict(uncompressed_bytes=100000000))] for m in METHODS},
        source_checks=[cost],hash_samples=[cost],profile_cpu_s=10)

def test_resource_model_exact_matrix_and_all_mandatory_costs():
    raw=raw_fixture();result=projection(raw,dict(active_cpu_s=20,wall_s=30),1000)
    assert result['gate_passed']
    for p in result['training_components_s'].values():
        assert p['native']==50 and p['independent_audit']==50 and p['optimizer']==5
        assert p['checkpoints']>0 and p['recovery']>0 and p['resets']>0 and p['startup']>0
    assert all(x==43.2 for x in result['diagnostic_s'].values())
    assert result['checkpoint_peak_bound_bytes']==500000000
    assert result['sealed_preflight_upper_s']==60
    assert result['prior_active_upper_s']==140

def test_recurring_slow_block_cannot_be_dropped():
    raw=raw_fixture();raw['native'][METHODS[0]].append(copy.deepcopy(raw['native'][METHODS[0]][0]))
    raw['native'][METHODS[0]][1]['execution']['wall_s']=128*2
    result=projection(raw,dict(active_cpu_s=20,wall_s=30),1000)
    assert not result['gate_passed'] and result['training_components_s'][METHODS[0]]['native']==100000

def test_storage_limit_and_publication_peak():
    raw=raw_fixture()
    for ss in raw['diagnostic'].values():ss[0]['compressed_bytes_per_step']=100000
    assert not projection(raw,dict(active_cpu_s=0,wall_s=0),0)['gate_passed']

def test_observed_cpu_wall_and_quantiles_separate():
    result=stats([dict(wall_s=x,cpu_s=x/100) for x in (1,2,3,4)])
    assert result['wall_s']['median']==2.5 and result['wall_s']['maximum']==4
    assert result['cpu_s']['maximum']==.04
    assert result['wall_s']['p99']>result['wall_s']['p95']

def test_source_guard_rejects_frozen_hash_mutation(monkeypatch):
    expected=integrity.identity()
    original=integrity.source_hashes()
    original['invented_corruption']='f'*64
    monkeypatch.setattr(integrity,'source_hashes',lambda:original)
    with pytest.raises(RuntimeError,match='frozen scientific source'):
        integrity.verify_sources(expected)

def test_v3_resource_and_namespace_frozen_before_science():
    c=integrity.read_config()
    assert c['active_compute_limit_s']==86400 and c['retained_artifact_limit_bytes']==3*2**30
    assert c['native_transitions_per_method']==50000 and c['updates_per_method']==5000
    assert c['resource_probe_repeats']==3 and c['resource_probe_steps']==128
    assert c['diagnostic_start']==12100000
    assert not c['validation_authorized'] and not c['paper_authorized']
