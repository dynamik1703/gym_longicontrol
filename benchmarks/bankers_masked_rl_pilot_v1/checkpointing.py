"""Atomic complete same-Attempt checkpoints with strict provenance validation."""

import os
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from benchmarks.bankers_viability_shield_v4.serialization import file_hash, fsync_directory

from .integrity import identity


def save(path, runner, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Includes environment, memory, task cursor, masks, partial metrics,
    # separate RNG objects, replay, networks/optimizers and all counters.
    payload = dict(schema='bankers-pilot-checkpoint-v1', identity=identity(),
                   runner=runner, python=random.getstate(),
                   numpy=np.random.get_state(), torch=torch.get_rng_state())
    fd, temp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(payload, stream, protocol=5)
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
                native=runner.native, updates=runner.agent.updates, attempt=runner.attempt)


def load(path, expected_hash, attempt):
    if file_hash(Path(path)) != expected_hash:
        raise RuntimeError('checkpoint digest mismatch')
    with Path(path).open('rb') as stream:
        value = pickle.load(stream)
    if value['schema'] != 'bankers-pilot-checkpoint-v1' or value['identity'] != identity():
        raise RuntimeError('checkpoint source/config/runtime mismatch')
    runner = value['runner']
    if runner.attempt != attempt:
        raise RuntimeError('checkpoint Attempt mismatch')
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'])
    return runner
