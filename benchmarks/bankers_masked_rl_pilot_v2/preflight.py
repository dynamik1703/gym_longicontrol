"""Exactly-once sealed post-commit preflight, separate from engineering runs."""

import json
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .integrity import HERE, ROOT, atomic_json, file_hash, identity, require, verify_sources


def main():
    freeze=json.loads((HERE/'freeze_manifest.json').read_text())
    verify_sources(freeze['identity'],committed=True)
    resource_model=json.loads((HERE/'resource_measurements.json').read_text())
    require(resource_model['gate_passed'],'resource gate closed')
    atomic_json(HERE/'SEALED_PREFLIGHT_STARTED.json',dict(identity=identity(),
        freeze_commit=freeze['freeze_commit'],started_unix=time.time()),immutable=True)
    command=[sys.executable,'-m','pytest',str(HERE/'tests'),'-q','-p','no:cacheprovider',
             '--junitxml='+str(HERE/'sealed_preflight.xml'),
             '--basetemp=/private/tmp/bankers_v2_sealed_preflight_tmp']
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    started=time.perf_counter()
    with (HERE/'sealed_preflight.log').open('xb') as out:
        completed=subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    suite=next(ET.parse(HERE/'sealed_preflight.xml').iter('testsuite'))
    passed=completed.returncode==0 and int(suite.attrib['tests'])>0
    record=dict(identity=identity(),freeze_commit=freeze['freeze_commit'],
        all_gates_passed=passed,exit_code=completed.returncode,
        tests=int(suite.attrib['tests']),failures=int(suite.attrib['failures']),
        errors=int(suite.attrib['errors']),wall_s=time.perf_counter()-started,
        active_cpu_s=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
        xml_sha256=file_hash(HERE/'sealed_preflight.xml'),log_sha256=file_hash(HERE/'sealed_preflight.log'),
        scientific_training_transitions=0,scientific_training_updates=0,
        gates={c:passed for c in 'ABCDEFGHIJKLMNOP'})
    verify_sources(freeze['identity'],committed=True)
    atomic_json(HERE/'sealed_preflight.json',record,immutable=True)
    if not passed:
        atomic_json(HERE/'scientific_stop.json',dict(
            classification='BANKERS_MASKED_RL_PILOT_V2_INTEGRITY_STOP',
            phase='SEALED_PREFLIGHT',identity=identity(),failures=record),immutable=True)
    print(json.dumps(record),flush=True)
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':
    main()
