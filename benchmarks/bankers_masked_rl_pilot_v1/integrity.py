"""Frozen source, execution ownership and explicit held-out access guards."""

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from benchmarks.bankers_viability_shield_v4.serialization import (
    atomic_json, canonical_bytes, digest, file_hash, require,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = 'f1490b9558ee1ea70b9f615b7add2f89b43ecffb'
REMOTE = 'https://github.com/dynamik1703/gym_longicontrol.git'


def read_config():
    return json.loads((HERE / 'canonical.json').read_text())


def source_hashes():
    paths = [*HERE.glob('*.py'), *HERE.glob('*PROTOCOL.md'),
             *HERE.glob('*CONTRACT*.md'), *HERE.glob('*TARGETS.md'),
             HERE / 'canonical.json', *(HERE / 'tests').glob('*.py')]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


def identity():
    return dict(base=BASE, sources_sha256=digest(source_hashes()),
                configuration_sha256=file_hash(HERE / 'canonical.json'),
                runtime=runtime())


def runtime():
    import gymnasium
    import numpy
    import torch
    return dict(python=sys.version, numpy=numpy.__version__,
                gymnasium=gymnasium.__version__, torch=torch.__version__)


def verify_historical():
    require(subprocess.check_output(
        ['git', 'remote', 'get-url', '--push', 'gym_longicontrol_origin'],
        cwd=ROOT, text=True).strip() == REMOTE, 'incorrect push remote')
    require(subprocess.run(['git', 'diff', '--quiet', BASE, '--', '.',
        ':(exclude)benchmarks/bankers_masked_rl_pilot_v1'], cwd=ROOT).returncode == 0,
        'historical mutation')
    v4 = ROOT / 'benchmarks/bankers_viability_shield_v4'
    manifest = json.loads((v4 / 'artifact_manifest.json').read_text())
    require(all(file_hash(v4 / p) == h for p, h in manifest.items()),
            'V4 sealed artifact mutation')
    audit = json.loads((v4 / 'source_audit.json').read_text())
    require(all(file_hash(ROOT / p) == h
                for p, h in audit['historical_actual_sha256'].items()),
            'V4 inherited source mutation')
    return dict(v4_files=len(manifest), inherited_files=len(audit['historical_actual_sha256']))


def guard_panel(panel):
    require(panel == 'BANKERS_MASKED_RL_PILOT_DIAGNOSTIC_V1',
            'HISTORICAL_VALIDATION_AND_PAPER_FORBIDDEN')


def guard_seed(seed):
    require(type(seed) is int and not 4000 <= seed <= 4017,
            'PAPER_SEALED_UNINSPECTED')


@contextmanager
def owner(directory=HERE):
    path = Path(directory) / '.exclusive-owner'
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        path.unlink()


def committed():
    for p in source_hashes():
        require(subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)
                == (ROOT / p).read_bytes(), 'uncommitted scientific source ' + p)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def actor_hash(agent):
    h = hashlib.sha256()
    for name, value in sorted(agent.actor.state_dict().items()):
        h.update(canonical_bytes(dict(name=name, shape=list(value.shape), dtype=str(value.dtype))))
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()
