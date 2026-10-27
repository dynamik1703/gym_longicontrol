"""Frozen descriptive matched comparisons and pilot-only decision rules."""

import json
import statistics

from .evaluation import ARMS
from .integrity import HERE, atomic_json, require
from .shield_treatments import METHODS


def summarize(rows):
    require(len(rows) == 72, 'Diagnostic denominator must be 72')
    output = {}
    for method in ARMS:
        subset = [r for r in rows if r['method'] == method]
        require(len(subset) == 12 and len({r['task']['numeric_seed'] for r in subset}) == 12,
                'method task denominator must be 12 unique tasks')
        feasible_energy = [r['metrics']['energy_kwh'] for r in subset if r['feasible']]
        output[method] = dict(rsr=sum(r['feasible'] for r in subset), denominator=12,
            completion=sum(r['completed'] for r in subset),
            deadline=sum(r['deadline'] for r in subset), speed=sum(r['speed'] for r in subset),
            mean_travel_time_s=statistics.mean(r['metrics']['travel_time_s'] for r in subset),
            mean_progress_m=statistics.mean(r['progress_m'] for r in subset),
            median_progress_m=statistics.median(r['progress_m'] for r in subset),
            feasible_energy_mean_kwh=statistics.mean(feasible_energy) if feasible_energy else None,
            feasible_energy_n=len(feasible_energy),
            maximum_speed_excess=max(r['metrics']['max_speed_violation_m_s'] for r in subset),
            mean_native_speed_excess=sum(r['mean_speed_excess']*r['native_steps'] for r in subset)/sum(r['native_steps'] for r in subset),
            correction_fraction=sum(r['mismatches'] for r in subset)/sum(r['native_steps'] for r in subset),
            mask_size_mean=statistics.mean(r['mask_size_mean'] for r in subset) if subset[0]['mask_size_mean'] is not None else None,
            positive_acceleration_choices_mean=statistics.mean(r['positive_acceleration_choices_mean'] for r in subset) if subset[0]['positive_acceleration_choices_mean'] is not None else None,
            actor_entropy_mean=statistics.mean(r['actor_entropy_mean'] for r in subset) if subset[0]['actor_entropy_mean'] is not None else None,
            failure_identities=[dict(numeric_seed=r['task']['numeric_seed'], reasons=r['failure_reasons']) for r in subset if not r['feasible']])
    pairs = {}
    for a, b in [(METHODS[2],METHODS[0]),(METHODS[2],METHODS[1]),(METHODS[1],METHODS[0]),
                 *[(m,r) for m in METHODS for r in ARMS[3:]]]:
        aa = {r['task']['numeric_seed']:r for r in rows if r['method']==a}
        bb = {r['task']['numeric_seed']:r for r in rows if r['method']==b}
        require(aa.keys()==bb.keys(), 'matched task identities')
        co = [n for n in aa if aa[n]['feasible'] and bb[n]['feasible']]
        pairs[a+'__VS__'+b] = dict(a_only_success=sum(aa[n]['feasible'] and not bb[n]['feasible'] for n in aa),
            b_only_success=sum(bb[n]['feasible'] and not aa[n]['feasible'] for n in aa),
            both_success=len(co), both_failed=sum(not aa[n]['feasible'] and not bb[n]['feasible'] for n in aa),
            jointly_feasible_n=len(co),
            jointly_feasible_energy_a_minus_b_kwh=statistics.mean(aa[n]['metrics']['energy_kwh']-bb[n]['metrics']['energy_kwh'] for n in co) if co else None)
    pre = output[METHODS[2]]
    if pre['speed']==12 and pre['completion']>=1 and pre['median_progress_m']>=500 and pre['rsr']>max(output[METHODS[0]]['rsr'],output[METHODS[1]]['rsr']):
        label='BANKERS_MASKED_RL_PILOT_V3_PROMISING'
    elif all(output[m]['completion']==0 for m in METHODS) or (pre['completion']==0 and pre['median_progress_m']<100):
        label='BANKERS_MASKED_RL_PILOT_V3_NO_LEARNING_SIGNAL'
    else:
        label='BANKERS_MASKED_RL_PILOT_V3_INCONCLUSIVE'
    return dict(classification=label, methods=output, matched_comparisons=pairs,
                exploratory_only=True, training_seeds=1, diagnostic_tasks=12)


def finalize():
    data=json.loads((HERE/'diagnostic_results.json').read_text())
    result=summarize(data['results'])
    result.update(diagnostic_ordered_sha256=data['ordered_sha256'],
                  training_native_transitions=150000,training_updates=15000)
    resource=json.loads((HERE/'resource_measurements.json').read_text())
    result['scaleup_justified_exploratorily']=(result['classification'].endswith('PROMISING')
        and resource['scaleup_three_methods_three_seeds_300k_s']<=86400)
    result['full_scale_training_authorized']=False
    atomic_json(HERE/'results.json',result,immutable=True)
    return result
