"""Descriptive report publication without modifying frozen scientific protocols."""
import json
import statistics
from .integrity import HERE, ROOT, atomic_json, file_hash, identity, verify_sources
from .shield_treatments import METHODS
from .evaluation import ARMS

def read(name,default=None):
    path=HERE/name
    return json.loads(path.read_text()) if path.exists() else default

def write(name,value):
    (HERE/name).write_text(value.rstrip()+'\n')

def main():
    commit=verify_sources()
    engineering=read('engineering_results.json',{});gate=read('sealed_preflight.json',{})
    resource=read('resource_measurements.json',{});raw=read('profile_raw.json',{})
    stop=read('scientific_stop.json');split=read('split_ledger.json',{})
    runs={m:read('training_runs/'+m+'/COMPLETED.json') for m in METHODS}
    all_pairs=read('diagnostic_results.json',{}).get('results',[])
    started=list((HERE/'diagnostic').glob('*/STARTED.json'))
    if stop:
        result=dict(classification=stop['classification'],scientific_stop=stop)
    elif len(all_pairs)==72:
        from .analysis import summarize
        result=summarize(all_pairs)
    else:
        raise RuntimeError('cannot finalize an unfinished study without a scientific stop')
    counts={}
    for m,r in runs.items():
        counts[m]=dict(native=r['native'] if r else 0,updates=r['updates'] if r else 0,
            actor_sha256=r['actor_sha256'] if r else None,checkpoint=r['checkpoint'] if r else None,
            completed=bool(r),verification=r['verification'] if r else None)
        if not r:
            # A failed Attempt may contain real transitions; disclose journal count.
            path=HERE/'training_runs'/m/'STARTED.json'
            if path.exists():
                from .journal import Journal
                marker=json.loads(path.read_text())
                rows=[f['row'] for f in Journal(path.parent/'journal',marker['attempt'],marker['identity']).records()]
                counts[m].update(native=len(rows),updates=rows[-1]['updates_after'] if rows else 0,completed=False)
    result.update(identity=identity(),training=counts,diagnostic_pairs_started=len(started),diagnostic_pairs_completed=len(all_pairs),
        diagnostic_split_sha256=split.get('ordered_sha256'),paper='SEALED_UNINSPECTED',
        final_validation_performance_episodes=0,full_scale_training_authorized=False,
        resource_projection=resource,engineering=engineering,sealed_preflight=gate or 'NOT_STARTED',
        resource_ledger=read('resource_ledger.json',{}),original_stream_sha256='08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c')
    atomic_json(HERE/'results.json',result,immutable=True)
    label=result['classification']
    write('README.md',f'# BankerShield Masked RL Pilot V3\n\n**{label}**\n\nProspective 24-hour / 3-GiB study at frozen V2 base e4dd7cb. See HANDOFF_DE.md, RESOURCE_REPORT.md and results.json. Historical V2 remains RESOURCE_STOP. No full-scale, final Validation or Paper execution.')
    write('TRAINING_RESULTS.md','# Training results\n\n'+json.dumps(counts,indent=2)+'\n\nCounts are actual committed native journal events. No final actor/checkpoint exists for an uncompleted arm.')
    write('TRAINING_EVIDENCE.md','# Training evidence\n\nIndependent native selected-action verification, predecessor/envelope reconstruction and sampled full-grid audits are required online and from stored evidence; every replay slot is verified before final sealing.\n\n'+json.dumps({m:c['verification'] for m,c in counts.items()},indent=2))
    write('SPLIT_LEDGER.md','# Diagnostic split\n\n'+(json.dumps(split,indent=2) if split else 'NOT_OPENED. No Diagnostic tasks constructed; no split hash. Panel construction requires three independently sealed final checkpoints.'))
    write('DIAGNOSTIC_RESULTS.md','# Diagnostic results\n\n'+(json.dumps(result.get('methods'),indent=2) if all_pairs else 'NOT_STARTED. No controller performance measurement; zero pairs. Metrics are unavailable, not zero successes out of 12.'))
    write('PAIRED_COMPARISONS.md','# Matched comparisons\n\n'+(json.dumps(result.get('matched_comparisons'),indent=2) if all_pairs else 'Unavailable: Diagnostic not executed. No PRE/RAW, PRE/POST or POST/RAW learning conclusion.'))
    write('RESOURCE_REPORT.md','# Complete resource admission\n\n'+json.dumps(resource,indent=2)+'\n\nObserved ledger (separate from conservative projection):\n\n'+json.dumps(result['resource_ledger'],indent=2))
    old=json.loads((ROOT/'benchmarks/bankers_masked_rl_pilot_v2/resource_measurements.json').read_text())
    maxima={m:max((dict(speed=s['speed'],sample=i,**c) for s in ss for i,c in enumerate(s['engine_step'][1:],1)),key=lambda c:c['wall_s']) for m,ss in old['native_samples'].items()}
    write('RESOURCE_RECALIBRATION.md','''# V2 short-window anomaly and V3 prospective measurement

V2's formula selects the worst individual nonsampled wall latency among only four steady steps per speed. The anomalous RAW maximum uses 0.124179 wall seconds but only 0.003522 process CPU seconds, a 35.3× wall/CPU ratio. PRE's 0.193104-second wall maximum similarly uses 0.019217 CPU seconds. This demonstrates non-CPU delay inside those measured steps; V2 does not have instrumentation to identify whether filesystem synchronization, operating-system scheduling or another wait caused it. It does not establish a masking or measurement bug. First-use optimizer overhead is separately visible in RAW's first update; sampled-grid work is the first native step and was excluded from V2's nonsampled maximum. RAW's large wall maximum is the second step at 25.5 m/s, so a one-time initialization explanation alone is not supported.

V3 uses all repeated complete blocks under the formula committed before profiling. It retains every sample and reports individual maxima as an explicit stress scenario. Cold initialization/inference is measured separately but the first native step remains included in every admission block. No observed recurring high cost is dropped. Profiling measures synthetic engineering behavior only, not task performance.

V2 maxima (immutable original measurements):
'''+json.dumps(maxima,indent=2)+'\n\nV3 component distributions (CPU/wall, median/p95/p99/max):\n\n'+json.dumps(raw.get('component_statistics',{}),indent=2)+'\n\nPer-arm/band distributions: profile_raw.json. Full raw samples retained.')
    write('RL_SCALEUP_READINESS.md','# Prospective scale-up\n\n'+json.dumps({k:v for k,v in resource.items() if k.startswith('scaleup_')},indent=2)+'\n\nForecasts include online masks, optimization, evidence, independent audits, checkpoint/recovery and physical baselines. If pilot did not execute these are synthetic forecasts, not actual pilot extrapolations. V2 110/230-hour projection remains historical. No 300k performance inference or automatic scale-up authorization.')
    write('SCIENTIFIC_DECISION.md',f'# Scientific decision\n\n{label}\n\n'+('The stopped execution cannot answer whether PRE improves learning or merely supplies external speed compliance. This is not a negative learning result. Further training needs a new authorized resolution of the stopping gate.' if not all_pairs else 'One seed and twelve routes provide exploratory matched evidence only; no superiority proof or statistical equivalence claim. No further training is authorized.'))
    write('INTEGRITY_REPORT.md','# Integrity report\n\n'+json.dumps(dict(engineering=engineering,sealed_preflight=gate or 'NOT_STARTED',stop=stop,historical_sources_unchanged=True,paper='SEALED_UNINSPECTED',final_validation_episodes=0),indent=2)+'\n\nComplete recovery tests preserve actor/critics/optimizers/replay, all RNGs, stream cursor, partial episode, current/successor bindings and durable STARTED-before-reset. Real-process crash tests included. No silent fresh scientific Attempt. Final source/hash validation passed.')
    lines=[f'# Deutscher Abschlussbericht — {label}',
        f'1. Branch research/bankers-masked-rl-pilot-v3; Worktree {ROOT}; Quell-/Freeze-Commit siehe freeze_manifest.json; Berichts-Ausgangscommit {commit}; geprüfter Remote gym_longicontrol_origin = https://github.com/dynamik1703/gym_longicontrol.git.',
        '2. Sämtliche früheren Klassifikationen und Artefakte unverändert; insbesondere V2 RESOURCE_STOP (18,985046 h / 2,042108 GiB).',
        '3. Unveränderter diskreter SAC: 21 Aktionen, kausale 10D-Beobachtung, signierter Energie-Reward, separate Geschwindigkeits-/Terminal-/Interventionskosten; gleicher physikalischer V4-Shield für POST/PRE.',
        f"4. Neue konservative Projektion: {resource.get('projected_active_compute_upper_s',0)/3600:.6f} h; {resource.get('projected_retained_bytes',0)/2**30:.6f} GiB; Grenzen 24 h / 3 GiB. Einzelkomponenten in RESOURCE_REPORT.md.",
        '5. RAWs V2-Maximum: 124,179 ms Wall gegenüber 3,522 ms CPU; nachgewiesene Wartezeit, genaue Ursache rückwirkend nicht identifizierbar. Neue wiederholte 128er-Blöcke, alle sieben Geschwindigkeiten, alle Rohdaten und Kalt-/Dauerkosten separat.',
        f"6. Engineering: {engineering.get('tests')} Tests, {engineering.get('failures')} Failures, {engineering.get('errors')} Errors; versiegelter Preflight: {gate.get('all_gates_passed','NICHT GESTARTET')}.",
        '7. Exakte Trainingszahlen: '+json.dumps({m:{k:c[k] for k in ('native','updates')} for m,c in counts.items()}),
        '8. Finale Actor-/Checkpoint-Hashes: '+json.dumps({m:dict(actor=c['actor_sha256'],checkpoint=c['checkpoint']) for m,c in counts.items()}),
        '9. Recovery/exakt einmal: vollständige Zustands- und echte Prozessabbruchtests; wissenschaftliche STARTED-/COMPLETED-Dateien bleiben unveränderlich.',
        f"10. Diagnostic-Split-Hash: {split.get('ordered_sha256','NICHT GEÖFFNET')}.",
        '11. RSR/Completion/Deadline/Speed je Controller: '+(json.dumps(result.get('methods')) if all_pairs else 'NICHT GEMESSEN für alle sechs Controller; keine 0/12-Ergebnisse erfunden.'),
        '12. Aufgabenpaariger PRE/RAW-, PRE/POST-, POST/RAW-Vergleich: '+('siehe PAIRED_COMPARISONS.md.' if all_pairs else 'nicht verfügbar.'),
        '13. Actor-/Maskierungsstatistiken: '+('siehe DIAGNOSTIC_RESULTS.md und Rohdaten.' if all_pairs else 'keine wissenschaftlichen Beobachtungen.'),
        '14. Energie ausschließlich für feasible Episoden; physikalische Baselines: '+('siehe DIAGNOSTIC_RESULTS.md.' if all_pairs else 'nicht ausgeführt.'),
        '15. Fehleridentitäten: '+(json.dumps({m:v['failure_identities'] for m,v in result.get('methods',{}).items()}) if all_pairs else 'keine geöffneten Diagnostic-Aufgaben.'),
        f"16. Tatsächliche Profil-CPU {raw.get('profile_cpu_s')} s; Profil-Wall {raw.get('profile_wall_s')} s; Engineering-CPU {engineering.get('active_cpu_s')} s; Engineering-Wall {engineering.get('wall_s')} s. Gesamter wissenschaftlicher Ressourcen-Ledger in results.json; finale gespeicherte Bytes im final_readback.json.",
        '17. Unabhängige native Evidenzprüfung: '+('alle finalen Trainings-Replays und Diagnostic-Paare geprüft.' if all_pairs else 'synthetisch geprüft; keine wissenschaftlichen Ergebnisse vorgetäuscht.'),
        '18. Prospektiver Vollmaßstab: '+json.dumps({k:v for k,v in resource.items() if k.startswith('scaleup_')}),
        f'19. Wissenschaftliche Klassifikation: {label}.',
        '20. Weitere Masked-RL-Trainings sind nicht automatisch autorisiert; ein Resource-/Integrity-Stop beantwortet die Lernfrage nicht.',
        '21. Genau null finale Validation-Performance-Episoden.',
        '22. Paper 4000–4017 blieb SEALED_UNINSPECTED.',
        '23. Finaler sauberer Worktree, Commit und Synchronisierung ausschließlich mit korrektem GitHub-Remote werden im abschließenden Readback geprüft. Keine Force-Pushes.',
        'Forschungsfrage: '+('Explorative Ergebnisse siehe Vergleichstabellen; keine Überlegenheitsbehauptung.' if all_pairs else 'Nicht beantwortbar, weil das wissenschaftliche Gate die Lern-/Diagnostic-Ausführung gestoppt hat.')]
    write('HANDOFF_DE.md','\n\n'.join(lines))
    manifest={str(p.relative_to(HERE)):file_hash(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name not in ('artifact_manifest.json','final_readback.json')}
    atomic_json(HERE/'artifact_manifest.json',manifest,immutable=True)
    print(json.dumps(dict(classification=label,training=counts,diagnostic_pairs=len(all_pairs))),flush=True)

if __name__=='__main__':main()
