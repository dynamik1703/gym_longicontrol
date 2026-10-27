"""Read sealed records only; write descriptive reports and the final manifest."""

import json
from pathlib import Path

from .analysis import finalize
from .evaluation import ARMS
from .integrity import BASE, HERE, ROOT, REMOTE, atomic_json, file_hash, identity, require, verify_sources
from .shield_treatments import METHODS


def write(name,text):
    (HERE/name).write_text(text.strip()+'\n')


def main():
    verify_sources()
    resource=json.loads((HERE/'resource_measurements.json').read_text())
    stop_path=HERE/'scientific_stop.json'
    if stop_path.exists():
        stop=json.loads(stop_path.read_text())
        classification=stop['classification']
    elif not resource['gate_passed']:
        classification='BANKERS_MASKED_RL_PILOT_V2_RESOURCE_STOP'
        stop=dict(phase='COMPLETE_RESOURCE_GATE',classification=classification)
    else:
        stop=None
        if not (HERE/'results.json').exists():
            finalize()
        classification=json.loads((HERE/'results.json').read_text())['classification']
    training={}
    for method in METHODS:
        path=HERE/'training_runs'/method/'COMPLETED.json'
        training[method]=json.loads(path.read_text()) if path.exists() else None
        if training[method] is None and (path.parent/'STARTED.json').exists():
            from .journal import Journal
            marker=json.loads((path.parent/'STARTED.json').read_text())
            journal=Journal(path.parent/'journal',marker['attempt'],marker['identity'])
            committed=updates=0
            try:
                for frame in journal.records():
                    committed+=1
                    updates=frame['row']['updates_after']
            except Exception:
                pass  # Report a lower bound; preserve the failed journal verbatim.
            training[method]=dict(native=committed,updates=updates,partial=True,
                native_count_is_lower_bound=True,pending_intent=(path.parent/'journal/pending.json').exists(),
                actor_sha256=None,checkpoint=dict(sha256=None))
    scientific=json.loads((HERE/'sealed_preflight.json').read_text()) if (HERE/'sealed_preflight.json').exists() else None
    engineering=json.loads((HERE/'engineering_results.json').read_text())
    result_path=HERE/'results.json'
    if not result_path.exists():
        atomic_json(result_path,dict(classification=classification,identity=identity(),
            failed_phase=stop['phase'] if stop else None,
            training_native_transitions=sum(r['native'] for r in training.values() if r),
            training_updates=sum(r['updates'] for r in training.values() if r),
            training={m:dict(status=('STOPPED_PARTIAL' if r.get('partial') else 'COMPLETED') if r else 'NOT_STARTED',native=r['native'] if r else 0,
                updates=r['updates'] if r else 0,actor_sha256=r['actor_sha256'] if r else None,
                checkpoint_sha256=r['checkpoint']['sha256'] if r else None,
                native_count_is_lower_bound=bool(r and r.get('partial'))) for m,r in training.items()},
            diagnostic_ordered_sha256=None,diagnostic_pairs=0,performance_question_answered=False,
            resource=resource,full_scale_authorized=False,validation_authorized=False,paper='SEALED_UNINSPECTED'),immutable=True)
    result=json.loads(result_path.read_text())
    write('README.md',f'''# BankerShield masked RL pilot V2

**{classification}**

V2 independently reconstructs causal predecessor/current memory and preserves
Pilot V1's permanent integrity stop. Learner components are reused byte-for-byte; native lower ties are corrected
by the documented pre-training GRID_TIE_INDEX_EXACT_V2 amendment; execution/evidence/recovery are new V2 mechanisms.

Engineering: {engineering['tests']} tests, {engineering['failures']} failures,
{engineering['errors']} errors. Sealed preflight: {scientific['all_gates_passed'] if scientific else 'NOT_STARTED'}.
Projected active-compute upper bound: {resource['projected_active_compute_upper_s']/3600:.3f} h;
projected retained artifact bound: {resource['projected_retained_bytes']/2**30:.3f} GiB.

See [HANDOFF_DE.md](HANDOFF_DE.md), [SCIENTIFIC_DECISION.md](SCIENTIFIC_DECISION.md),
[RESOURCE_REPORT.md](RESOURCE_REPORT.md), [INTEGRITY_REPORT.md](INTEGRITY_REPORT.md),
and [results.json](results.json). No full-scale, final Validation or Paper evaluation.
''')
    write('RESOURCE_REPORT.md',f'''# Complete measured resource model

The fixed conservative formula gives **{resource['projected_active_compute_upper_s']/3600:.6f} h**
and **{resource['projected_retained_bytes']/2**30:.6f} GiB** for the complete
150,000-transition/15,000-update/72-pair pilot. Limits: 8 h and 2 GiB.
Gate: **{'PASS' if resource['gate_passed'] else 'FAIL'}**.

The model uses maxima across seven synthetic speed bands separately for each
arm. It includes online and successor masks, full selected certification,
actual transition audits, sampled training grids, compact durable journals,
replay, all optimizers, checkpoints and one verified recovery per arm,
complete Diagnostic native evidence and independent grid/controller verification,
source checks and final sealing/reporting. A factor {resource['factor']} applies
to the complete workload; measured/prior engineering costs are additive.
Raw samples, formulas and checkpoint byte bounds are in resource_measurements.json.

Prospective one-method x3 seeds x300k: {resource['scaleup_one_method_three_seeds_300k_s']/3600:.3f} h.
Prospective three-method x3 seeds x300k: {resource['scaleup_three_methods_three_seeds_300k_s']/3600:.3f} h.
The workload keeps 30k updates per 300k run and a single final 12-task panel;
it is not authorization or a guarantee for a future policy-state distribution.
No <=24h claim is made if the corresponding measured model exceeds 24h.

Actual process CPU/wall and retained artifact accounting, when execution occurs,
are in resource_ledger.json. Temporary deterministic engineering fixture directories
are excluded from retained scientific evidence and may be removed after their
XML/source identities are retained. Historical and V1 dependency environments are reused.
''')
    table=['|Method|Native transitions|Updates|Status|','|---|---:|---:|---|']
    for m,r in training.items():
        table.append(f'|{m}|{r["native"] if r else 0}/50000|{r["updates"] if r else 0}/5000|'+(('STOPPED_PARTIAL_LOWER_BOUND' if r.get('partial') else 'COMPLETED') if r else 'NOT_STARTED')+'|')
    write('TRAINING_RESULTS.md','# Training results\n\n'+'\n'.join(table)+'\n\n'+
        'Completed-run records retain actor/checkpoint hashes, all optimizer diagnostics,\n'
        'partial episode state and actual task exposure. Different episode lengths may\n'
        'expose different prefixes of the identical canonical task order.\n')
    if 'methods' in result:
        lines=['|Arm|RSR/12|Completion|Deadline|Exact speed|Feasible energy kWh|',
               '|---|---:|---:|---:|---:|---:|']
        for m in ARMS:
            r=result['methods'][m]
            lines.append(f'|{m}|{r["rsr"]}/12|{r["completion"]}|{r["deadline"]}|{r["speed"]}|{r["feasible_energy_mean_kwh"]}|')
        write('DIAGNOSTIC_RESULTS.md','# Exploratory Diagnostic\n\n'+'\n'.join(lines)+'\n\n'+
            'Full failure identities, progress, speed excess, corrections, masks and entropy\n'
            'are in results.json and each immutable completed pair. Energy is feasible-only.\n'
            'One training seed and twelve tasks cannot establish superiority or equivalence.\n')
        write('PAIRED_COMPARISONS.md','# Matched comparisons\n\n```json\n'+json.dumps(result['matched_comparisons'],indent=2)+'\n```\n\n'+
            'All denominators are the same twelve tasks. Energy differences use only\n'
            'jointly feasible pairs and cannot compensate for any failed requirement.\n')
    else:
        write('DIAGNOSTIC_RESULTS.md','# Diagnostic not executed\n\nNo held-out panel was constructed or opened. All six arms have unmeasured performance; do not report them as 0/12.\n')
        write('PAIRED_COMPARISONS.md','# No performance comparison\n\nPRE–RAW, PRE–POST, POST–RAW and physical-reference comparisons are unavailable. No feasible-energy comparison exists.\n')
    write('SCIENTIFIC_DECISION.md',f'''# Scientific decision

**{classification}**

{('The complete resource gate failed before scientific training. No matrix reduction, safety omission or held-out task opening occurred.' if classification.endswith('RESOURCE_STOP') and not any(training.values()) else 'See the sealed preflight, completed run/pair records and results.json for the exact gate and outcome evidence.')}

Pilot V1 remains BANKERS_MASKED_RL_PILOT_INTEGRITY_STOP with its original 9/3
test outcome. The repaired V2 engineering results do not reclassify V1.
Speed safety and learned deadline success are separate. A pilot-only descriptive
signal cannot prove method superiority. No subsequent reward tuning, final
Validation, Paper evaluation or automatic full-scale training is authorized.
''')
    write('INTEGRITY_REPORT.md',f'''# Integrity report

V2 engineering: {engineering['tests']} tests; failures={engineering['failures']}; errors={engineering['errors']}.
Post-freeze sealed preflight: {json.dumps(scientific,indent=2) if scientific else 'NOT_STARTED: complete resource gate did not authorize scientific freeze/training.'}

Auditor state is reconstructed from trusted reset, actual actions and independently
reconstructed observations. The predecessor is copied before observation ingest;
the inherited three-argument verifier checks physical bounds and information
monotonicity. Recorded memory is compared against this reconstructed authority.
Every selected Banker execution receives the original final certificate.
Training audit mask coverage is sampled at insertion %100==0, current and successor;
Diagnostic coverage is complete. No sampled grid audit is called exhaustive.

{json.dumps(stop,indent=2) if stop else 'All 72 completed Diagnostic pairs and all three 50k endpoints were independently verified before sealing.'}

V1–V4 BankerShield and Pilot V1 remain byte-for-byte unchanged. Historical
Validation content was neither constructed nor evaluated. Paper 4000–4017
remains SEALED_UNINSPECTED. No historical sealed preflight was rerun.
''')
    write('RL_SCALEUP_READINESS.md',f'''# Scale-up readiness

Pilot classification: {classification}.
One-method, three-seed 300k prospective model: {resource['scaleup_one_method_three_seeds_300k_s']/3600:.3f} h.
Three-method, three-seed 300k prospective model: {resource['scaleup_three_methods_three_seeds_300k_s']/3600:.3f} h.
Resource/pass status and actual learned performance must both justify a future
study. This package authorizes no full-scale training. A stopped or ineffective
pilot cannot be presented as evidence of useful masked learning.
''')
    panel_path=HERE/'split_ledger.json'
    panel=json.loads(panel_path.read_text()) if panel_path.exists() else None
    write('SPLIT_LEDGER.md','# Split ledger\n\n'+(f"Frozen Diagnostic hash: `{panel['ordered_sha256']}`. See split_ledger.json for all Policy-R exclusions." if panel else 'No Diagnostic panel was constructed. The proposed 12,000,000 namespace was not opened.')+'\n\nHistorical Validation remains untouched; Paper 4000–4017 remains SEALED_UNINSPECTED.\n')
    hashes={m:dict(actor=r['actor_sha256'],checkpoint=r['checkpoint']['sha256']) if r else None for m,r in training.items()}
    write('HANDOFF_DE.md',f'''# Deutscher Abschlussbericht — BankerShield Masked RL Pilot V2

1. Branch `research/bankers-masked-rl-pilot-v2`; Worktree `{ROOT}`; Basis `{BASE}`.
   Der finale Berichtscommit ist HEAD; der Chat bestätigt den tatsächlichen Commit und Remote-Gleichstand.
2. Ausschließlich verifiziertes Remote `gym_longicontrol_origin`: `{REMOTE}`. Kein Force-Push.
3. Reparatur: `verify_envelope(current_memory, previous_memory, track)` mit echtem Vorgänger.
4. Der Auditor rekonstruiert seinen Speicher vom vertrauenswürdigen Reset aus; vor jedem Ingest kopiert er den Vorgänger. Evidenzsnapshots sind Vergleichsdaten, keine Autorität.
5. Pilot V1 bleibt unverändert `BANKERS_MASKED_RL_PILOT_INTEGRITY_STOP`, 9 bestandene/3 fehlgeschlagene Tests, null wissenschaftliche Transitionen.
6. V2-Engineering: {engineering['tests']} Tests, {engineering['failures']} Fehler, {engineering['errors']} Errors. Versiegelter Preflight: {scientific['all_gates_passed'] if scientific else 'NICHT GESTARTET'}.
7. Learner, Verteilungen und Replay wurden gegenüber V1 bytegleich übernommen. Kein neuer Reward, keine geänderten Lernraten oder Aktionsgitter. Die vorab dokumentierte Korrektur GRID_TIE_INDEX_EXACT_V2 setzt die festgelegte kleinere Aktion bei Gitter-Gleichstand exakt um.
8. Vollständige konservative Projektion: {resource['projected_active_compute_upper_s']/3600:.6f} Stunden, {resource['projected_retained_bytes']/2**30:.6f} GiB; Gate {'BESTANDEN' if resource['gate_passed'] else 'NICHT BESTANDEN'}.
9. Trainingszähler je Behandlung: {json.dumps({m:dict(native=r['native'] if r else 0,updates=r['updates'] if r else 0) for m,r in training.items()})}.
10. Finale Actor-/Checkpoint-Hashes: {json.dumps(hashes)}.
11. Same-Attempt-Recovery wurde synthetisch um Aktion, Replay, Updates, Checkpoints und Taskwechsel sowie mit echtem Prozessabbruch geprüft. Bei unvollständig verifizierbarem Journal erfolgt kein Neustart.
12. Unabhängige native Evidenz: vollständig implementiert und synthetisch geprüft; tatsächliche wissenschaftliche Abdeckung steht in den versiegelten Run-/Pair-Verifikationen. Trainings-Gridaudit ist eine deklarierte Stichprobe, kein Vollnachweis aller Masken.
13. Diagnostic-Panelhash: {panel['ordered_sha256'] if panel else 'nicht vorhanden; Panel nicht konstruiert'}.
14. RSR, Completion, Deadline und Speed der sechs Arme: siehe DIAGNOSTIC_RESULTS.md; bei nicht ausgeführtem Diagnostic ausdrücklich nicht gemessen.
15. PRE/POST/RAW-Vergleiche: siehe PAIRED_COMPARISONS.md; keine Aussage ohne ausgeführte Paare.
16. Actor-Verhalten, Korrekturen, Masken, Entropie und Multiplikatoren: nur tatsächlich ausgeführte Trainings-/Diagnostic-Datensätze auswertbar; keine Ersatzwerte aus synthetischen Tests.
17. Feasible-only-Energie und physische Referenzen: nur gemeinsam zulässige abgeschlossene Aufgaben; Energie kompensiert keinen harten Fehler.
18. Exakte Fehler-/Stoppidentität: {json.dumps(stop) if stop else 'siehe Failure-Identitäten je Arm in results.json; kein Integritätsstopp'}.
19. Gemessener Ressourcenprobe-CPU-Aufwand: {resource['measured_probe_cpu_s']:.6f} s. Studienartefakte und weitere Ausführungskosten werden in resource_ledger.json/final_readback dokumentiert; reine Projektionen sind keine gemessenen Trainingskosten.
20. Konservative Hochskalierung: eine Methode ×3 Seeds ×300k = {resource['scaleup_one_method_three_seeds_300k_s']/3600:.3f} h; drei Methoden ×3 Seeds ×300k = {resource['scaleup_three_methods_three_seeds_300k_s']/3600:.3f} h. Keine automatische Freigabe oder 24h-Zusage bei überschrittenem Modell.
21. Klassifikation: **{classification}**.
22. Eine weitere Vollstudie benötigt nachgewiesene Integrität, einen tragfähigen Ressourcenplan und ein begründetes Lernsignal; dieser Auftrag autorisiert sie nicht.
23. BankerShield V1–V4 und Pilot V1 bleiben unverändert und ihre Incidents erhalten.
24. Historische Validation bleibt unangetastet.
25. Paper 4000–4017 bleibt **SEALED_UNINSPECTED**.
26. Sauberer Worktree und Remote-Synchronisierung werden nach finalem Commit im Chat mit SHA bestätigt.

Die Forschungsfrage ist bei einem Ressourcen-/Integritätsstopp unbeantwortet. Bei
vollständiger Ausführung sind alle Unterschiede explorativ: ein Seed, zwölf Aufgaben.
Keine finale Validation, Paper-Ausführung oder automatische 300k-Vollstudie.
''')
    verify_sources()
    files={str(p.relative_to(HERE)):file_hash(p) for p in sorted(HERE.rglob('*')) if p.is_file()
        and p.name not in ('artifact_manifest.json','seal_manifest.json','.exclusive-owner')}
    atomic_json(HERE/'artifact_manifest.json',files,immutable=True)
    atomic_json(HERE/'seal_manifest.json',dict(classification=classification,
        identity=identity(),artifact_manifest_sha256=file_hash(HERE/'artifact_manifest.json'),
        results_sha256=file_hash(result_path)),immutable=True)
    print(json.dumps(dict(classification=classification,sealed_files=len(files),
        retained_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))),flush=True)


if __name__=='__main__':
    main()
