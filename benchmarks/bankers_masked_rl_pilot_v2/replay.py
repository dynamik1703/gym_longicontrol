"""Proposed-action replay with immutable successor causal-state bindings."""

import numpy as np


class Replay:
    def __init__(self, capacity=300000):
        self.capacity, self.cursor, self.size = capacity, 0, 0
        self.arrays = {}
        for key, shape, dtype in (
            ('observation', (10,), np.float32),
            ('next_observation', (10,), np.float32),
            ('action', (1,), np.int64), ('executed', (1,), np.int64),
            ('reward', (1,), np.float32), ('costs', (3,), np.float32),
            ('discount', (1,), np.float32),
            ('mask', (21,), np.bool_), ('next_mask', (21,), np.bool_),
            ('binding', (), 'S64'), ('next_binding', (), 'S64'),
            ('insertion', (), np.int64),
        ):
            self.arrays[key] = np.zeros((capacity, *shape), dtype=dtype)

    def append(self, row):
        if set(row) != set(self.arrays):
            raise ValueError('replay fields changed')
        if row['insertion'] != self.size or self.size >= self.capacity:
            raise ValueError('duplicate replay insertion or pilot capacity exceeded')
        if len(row['binding']) != 64 or len(row['next_binding']) != 64:
            raise ValueError('missing state binding')
        if not any(row['mask']) or not any(row['next_mask']):
            raise ValueError('empty policy support')
        for k, v in row.items():
            self.arrays[k][self.cursor] = v
        self.cursor += 1
        self.size += 1

    def sample(self, rng, batch_size=256):
        if self.size < batch_size:
            raise ValueError('insufficient replay')
        ids = rng.integers(0, self.size, size=batch_size)
        return {k: v[ids].copy() for k, v in self.arrays.items()
                if k not in ('binding', 'next_binding', 'insertion', 'executed')}
