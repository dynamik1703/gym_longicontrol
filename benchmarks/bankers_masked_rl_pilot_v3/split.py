"""Policy-R selection from metadata; no historical reserved route construction."""

import json

from benchmarks.content_split_audit.fingerprints import fingerprint_track
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import track_from_seed

from .integrity import HERE, ROOT, PANEL, atomic_json, digest, file_hash, guard_seed, require
from .shield_treatments import METHODS


def build():
    require(not (HERE / 'split_ledger.json').exists(), 'Diagnostic panel already frozen')
    for method in METHODS:
        run = HERE / 'training_runs' / method / 'COMPLETED.json'
        require(run.exists(), 'all three final checkpoints must be sealed before split construction')
        record = json.loads(run.read_text())
        require(record['native'] == 50000 and record['updates'] == 5000,
                'training endpoint incomplete')
        require(file_hash(run.parent / 'final_50000.pkl') == record['checkpoint']['sha256'],
                'final checkpoint mutation')
    numbers, routes, fulls, reserved_starts, sources = set(range(4000, 4018)), set(), set(), set(), []

    def metadata(value):
        if isinstance(value, dict):
            for key in ('numeric_seed', 'reset_seed'):
                if type(value.get(key)) is int:
                    numbers.add(value[key])
            if isinstance(value.get('route_sha256'), str):
                routes.add(value['route_sha256'])
            if isinstance(value.get('full_task_sha256'), str):
                fulls.add(value['full_task_sha256'])
            for key, child in value.items():
                if key in ('candidate_start', 'numeric_scan_start', 'scan_start', 'selected_start') and type(child) is int:
                    reserved_starts.add(child)
                # Never parse or inspect serialized task/route contents from a
                # reserved ledger. Only existing numeric and hash metadata is used.
                if key not in ('canonical_task_serialization', 'canonical_route_serialization'):
                    metadata(child)
        elif isinstance(value, list):
            for child in value:
                metadata(child)

    paths = set(ROOT.glob('benchmarks/**/*ledger*.json')) | set(ROOT.glob('benchmarks/**/*reservation*.json'))
    paths.add(ROOT / 'benchmarks/content_split_audit/results.json')
    for path in sorted(paths):
        if HERE in path.parents:
            continue
        metadata(json.loads(path.read_text()))
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=file_hash(path)))
    start = 12100000
    require(not any(start <= n < start + 10000 for n in reserved_starts),
            'requested Diagnostic numeric namespace already reserved')
    atomic_json(HERE / 'namespace_reservation.json', dict(panel=PANEL, scan_start=start,
        protected_metadata_sources=sources, paper='SEALED_UNINSPECTED'), immutable=True)
    accepted, scanned = [], []
    seed = start
    while len(accepted) < 12:
        if seed in numbers:
            scanned.append(dict(numeric_seed=seed, accepted=False, reasons=['NUMERIC_PROTECTED']))
            seed += 1
            continue
        guard_seed(seed)
        track = track_from_seed(seed)
        fp = fingerprint_track(track, v2_semantics())
        reasons = []
        if fp.route_sha256 in routes:
            reasons.append('ROUTE_PROTECTED')
        if fp.full_task_sha256 in fulls:
            reasons.append('FULL_TASK_PROTECTED')
        row = dict(numeric_seed=seed, route_sha256=fp.route_sha256,
                   full_task_sha256=fp.full_task_sha256)
        scanned.append(dict(**row, accepted=not reasons, reasons=reasons))
        if not reasons:
            accepted.append(row)
            numbers.add(seed)
            routes.add(fp.route_sha256)
            fulls.add(fp.full_task_sha256)
        seed += 1
    result = dict(panel=PANEL, policy='R', selection='FIRST_12_NO_OUTCOME_FILTER',
        ordered_tasks=accepted, ordered_sha256=digest(accepted), scanned=scanned,
        protected_metadata_sources=sources, paper='SEALED_UNINSPECTED')
    atomic_json(HERE / 'split_ledger.json', result, immutable=True)
    return result
