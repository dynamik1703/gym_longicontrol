"""Capture real child CPU, source identities and complete engineering test results."""
import json
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from .integrity import HERE, ROOT, atomic_json, file_hash, identity, source_hashes

def main():
    directory=HERE/'engineering';directory.mkdir(exist_ok=True)
    index=len(list(directory.glob('run_*.json')))+1
    stem=f'run_{index:02d}'
    hashes=source_hashes();start=time.perf_counter(),time.process_time()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    with (directory/(stem+'.log')).open('xb') as out:
        result=subprocess.run([sys.executable,'-m','pytest',str(HERE/'tests'),'-q','-p','no:cacheprovider',
            '--junitxml='+str(directory/(stem+'.xml')),'--basetemp=/private/tmp/bankers_masked_v3_'+stem],
            cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    suite=next(ET.parse(directory/(stem+'.xml')).iter('testsuite'))
    record=dict(identity=identity(),source_sha256=hashes,exit_code=result.returncode,
        tests=int(suite.attrib['tests']),failures=int(suite.attrib['failures']),errors=int(suite.attrib['errors']),
        wall_s=time.perf_counter()-start[0],active_cpu_s=time.process_time()-start[1]+
            after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        xml_sha256=file_hash(directory/(stem+'.xml')),log_sha256=file_hash(directory/(stem+'.log')),
        phase='ENGINEERING_BEFORE_SCIENTIFIC_FREEZE')
    atomic_json(directory/(stem+'.json'),record,immutable=True)
    record['active_cpu_s']=sum(json.loads(p.read_text())['active_cpu_s'] for p in directory.glob('run_*.json'))
    atomic_json(HERE/'engineering_results.json',record)
    print(json.dumps(record),flush=True)
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
