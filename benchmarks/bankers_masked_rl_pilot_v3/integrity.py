"""V2 identities, immutable publication, ownership and held-out boundaries."""

import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from benchmarks.bankers_viability_shield_v4.serialization import (
    atomic_json, canonical_bytes, digest, file_hash, fsync_directory, require,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = 'e4dd7cb0e4300cb9b66cf4b62cc39340247b0e40'
REMOTE = 'https://github.com/dynamik1703/gym_longicontrol.git'
PANEL = 'BANKERS_MASKED_RL_PILOT_DIAGNOSTIC_V3'
PROTOCOLS = ('LEARNER_AND_REWARD_FREEZE.md', 'INDEPENDENT_AUDIT_CONTRACT.md',
             'TRAINING_PROTOCOL.md', 'RECOVERY_AND_EXACTLY_ONCE.md',
             'TRAINING_EVIDENCE_CONTRACT.md', 'RESOURCE_PROTOCOL.md',
             'DIAGNOSTIC_PROTOCOL.md', 'SCIENTIFIC_METHOD_FREEZE.md', 'LEARNER_CONTRACT.md')


def read_config():
    return json.loads((HERE / 'canonical.json').read_text())


def source_hashes():
    paths = [*HERE.glob('*.py'), *(HERE / 'tests').glob('*.py'), HERE / 'canonical.json']
    paths.extend(HERE / n for n in PROTOCOLS if (HERE / n).exists())
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def runtime():
    import gymnasium
    import numpy
    import torch
    return dict(python=sys.version, numpy=numpy.__version__,
                gymnasium=gymnasium.__version__, torch=torch.__version__)


@lru_cache(maxsize=1)
def identity():
    return dict(study='BANKERS_MASKED_RL_PILOT_V3', base=BASE,
                source_sha256=digest(source_hashes()), runtime=runtime(),
                configuration_sha256=file_hash(HERE / 'canonical.json'))


def verify_sources(expected=None, committed=False):
    require(subprocess.check_output(['git', 'remote', 'get-url', '--push',
        'gym_longicontrol_origin'], cwd=ROOT, text=True).strip() == REMOTE,
        'incorrect push remote')
    require(subprocess.run(['git', 'diff', '--quiet', BASE, '--', '.',
        ':(exclude)benchmarks/bankers_masked_rl_pilot_v3'], cwd=ROOT).returncode == 0,
        'historical source/artifact mutation')
    for package in ('bankers_masked_rl_pilot_v1', 'bankers_viability_shield_v4', 'bankers_masked_rl_pilot_v2'):
        directory = ROOT / 'benchmarks' / package
        manifest = json.loads((directory / 'artifact_manifest.json').read_text())
        require(all(file_hash(directory / p) == h for p, h in manifest.items()),
                package + ' sealed artifact mutation')
    if expected is not None:
        actual = dict(study='BANKERS_MASKED_RL_PILOT_V3', base=BASE,
            source_sha256=digest(source_hashes()), runtime=runtime(),
            configuration_sha256=file_hash(HERE / 'canonical.json'))
        require(actual == expected, 'frozen scientific source/config/runtime mutation')
    if committed:
        for p in source_hashes():
            require(subprocess.check_output(['git', 'show', 'HEAD:' + p], cwd=ROOT)
                    == (ROOT / p).read_bytes(), 'uncommitted scientific source ' + p)
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def guard_panel(panel):
    require(panel == PANEL, 'HISTORICAL_VALIDATION_AND_PAPER_FORBIDDEN')


def guard_seed(seed):
    require(type(seed) is int and not 4000 <= seed <= 4017, 'PAPER_SEALED_UNINSPECTED')


@contextmanager
def owner(directory=HERE, recover_stale=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / '.exclusive-owner'
    if path.exists() and recover_stale:
        previous_pid=int(path.read_text())
        try:
            os.kill(previous_pid,0)
        except ProcessLookupError:
            path.unlink()
            fsync_directory(directory)
        else:
            raise RuntimeError('execution owner still alive; recovery forbidden')
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        fsync_directory(directory)
        yield
    finally:
        os.close(fd)
        path.unlink()
        fsync_directory(directory)


def actor_hash(agent):
    h = hashlib.sha256()
    for name, value in sorted(agent.actor.state_dict().items()):
        h.update(canonical_bytes(dict(name=name, shape=list(value.shape), dtype=str(value.dtype))))
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()
