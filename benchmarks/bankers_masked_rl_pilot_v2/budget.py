"""Active CPU accounting and retained-study artifact hard limits."""

import json
import resource
import time

from .integrity import HERE, atomic_json, require


class Budget:
    def __init__(self):
        self.path = HERE / 'resource_ledger.json'
        if self.path.exists():
            self.value = json.loads(self.path.read_text())
        else:
            model_path=HERE/'resource_measurements.json'
            model=json.loads(model_path.read_text()) if model_path.exists() else {}
            gate_path=HERE/'sealed_preflight.json'
            gate=json.loads(gate_path.read_text()) if gate_path.exists() else {}
            prior=model.get('prior_active_upper_s',0)+model.get('measured_probe_cpu_s',0)+gate.get('active_cpu_s',0)
            self.value=dict(closed_jobs=[dict(kind='PRIOR_ENGINEERING_PROBE_PREFLIGHT_UPPER_BOUND',
                actual_cpu_s=prior, prior_is_upper_bound=True)],
                recovered_unclosed_wall_upper_bound_s=0.0)
        if self.value.get('open_job'):
            job = self.value['open_job']
            self.value['recovered_unclosed_wall_upper_bound_s'] += max(0.0, time.time() - job['started_unix'])
            self.value['closed_jobs'].append(dict(kind='INTERRUPTED_WALL_UPPER_BOUND',
                actual_cpu_s=None, started_unix=job['started_unix']))
        self.start_cpu = time.process_time()
        self.start_wall = time.perf_counter()
        self.value['open_job'] = dict(started_unix=time.time())
        self.count = 0
        self.save()

    def cpu(self):
        return time.process_time() - self.start_cpu

    def total(self):
        return (self.value['recovered_unclosed_wall_upper_bound_s'] + self.cpu() +
            sum(x.get('actual_cpu_s') or 0.0 for x in self.value['closed_jobs']))

    def check(self, force=False):
        self.count += 1
        require(self.total() < 28800, 'BANKERS_MASKED_RL_PILOT_V2_RESOURCE_STOP: active compute')
        if force or self.count % 500 == 0:
            retained = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
            require(retained < 2147483648, 'BANKERS_MASKED_RL_PILOT_V2_RESOURCE_STOP: artifacts')
            self.value['retained_study_bytes'] = retained
            self.save()

    def save(self):
        self.value['live_cpu_s'] = self.cpu()
        self.value['charged_total_active_upper_bound_s'] = self.total()
        atomic_json(self.path, self.value)

    def close(self):
        self.value['closed_jobs'].append(dict(actual_cpu_s=self.cpu(),
            wall_s=time.perf_counter() - self.start_wall,
            max_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            started_unix=self.value['open_job']['started_unix']))
        self.value['open_job'] = None
        self.start_cpu = time.process_time()
        self.save()
