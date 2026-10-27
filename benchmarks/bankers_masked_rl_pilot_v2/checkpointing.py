"""Atomic full-state checkpoints with an embedded byte-integrity envelope."""

import hashlib
import gzip
import os
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from .integrity import file_hash, fsync_directory, identity, require


def save(path, runner, immutable=False, *, audit=None, journal_head=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = pickle.dumps(dict(schema='bankers-pilot-checkpoint-v2', identity=identity(),
        runner=runner, audit=audit, journal_head=journal_head,
        python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()),
        protocol=5)
    payload = pickle.dumps(dict(body=gzip.compress(body, compresslevel=1, mtime=0),
        sha256=hashlib.sha256(body).hexdigest(), compression='gzip-1'), protocol=5)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(temp, path)
            os.unlink(temp)
        else:
            os.replace(temp, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return dict(sha256=file_hash(path), bytes=path.stat().st_size,
                uncompressed_bytes=len(body),
                native=runner.native, updates=runner.agent.updates, attempt=runner.attempt)


def load_state(path, expected_hash, attempt):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = Path(path)
    require(expected_hash is None or file_hash(path) == expected_hash, 'checkpoint file SHA256')
    envelope = pickle.loads(path.read_bytes())
    require(envelope['compression'] == 'gzip-1', 'checkpoint representation')
    body = gzip.decompress(envelope['body'])
    require(hashlib.sha256(body).hexdigest() == envelope['sha256'],
            'checkpoint payload integrity')
    value = pickle.loads(body)
    require(value['schema'] == 'bankers-pilot-checkpoint-v2' and value['identity'] == identity(),
            'checkpoint scientific identity')
    require(value['runner'].attempt == attempt, 'checkpoint Attempt changed')
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    return value


def load(path, expected_hash, attempt):
    return load_state(path, expected_hash, attempt)['runner']
