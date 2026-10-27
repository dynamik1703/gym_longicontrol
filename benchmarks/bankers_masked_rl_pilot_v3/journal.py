"""Fsynced hash-chain rows, immutable gzip segments and write-ahead intents."""

import gzip
import json
import os
from pathlib import Path

from benchmarks.bankers_viability_shield_v2.serialization import strict_load

from .integrity import atomic_json, canonical_bytes, digest, fsync_directory, require


class Journal:
    def __init__(self, directory, attempt, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.attempt, self.identity = attempt, identity
        self.genesis = digest(dict(attempt=attempt, identity=identity, schema='journal-v2'))
        self.live = self.directory / 'live.jsonl'
        self.pending = self.directory / 'pending.json'

    def intent(self, prepared, head):
        atomic_json(self.pending, dict(prepared=prepared, previous=head,
            attempt=self.attempt, identity=self.identity), immutable=True)

    def clear_intent(self):
        self.pending.unlink()
        fsync_directory(self.directory)

    def append(self, row, previous):
        frame = dict(previous=previous, row=row, sha256=digest(dict(previous=previous, row=row)))
        payload = canonical_bytes(frame) + b'\n'
        fd = os.open(self.live, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'ab') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(self.directory)
        return frame['sha256']

    def records(self):
        previous, count = self.genesis, 0
        recent = {}
        paths = [*sorted(self.directory.glob('segment_*.gz'))]
        if self.live.exists():
            paths.append(self.live)
        for path in paths:
            opener = gzip.open if path.suffix == '.gz' else open
            with opener(path, 'rb') as stream:
                for line in stream:
                    require(line.endswith(b'\n'), 'torn journal: verified recovery unavailable')
                    frame = strict_load(line)
                    row = frame['row']
                    n = row['native']
                    if path == self.live and n < count:
                        require(recent.get(n) == frame['sha256'], 'conflicting rotated live journal')
                        continue
                    require(n == count and frame['previous'] == previous, 'journal discontinuity')
                    require(frame['sha256'] == digest(dict(previous=previous, row=row)), 'journal row hash')
                    require(row['attempt'] == self.attempt and row['identity'] == self.identity,
                            'journal Attempt/source identity')
                    previous = frame['sha256']
                    recent[n] = previous
                    if len(recent) > 1001:
                        del recent[min(recent)]
                    count += 1
                    yield frame

    def rotate(self):
        if not self.live.exists() or self.live.stat().st_size == 0:
            return
        payload = self.live.read_bytes()
        lines = payload.splitlines()
        first, last = json.loads(lines[0])['row']['native'], json.loads(lines[-1])['row']['native']
        target = self.directory / f'segment_{first:06d}_{last:06d}.gz'
        compressed = gzip.compress(payload, compresslevel=6, mtime=0)
        if target.exists():
            require(target.read_bytes() == compressed, 'immutable segment conflict')
        else:
            temp = self.directory / ('.' + target.name)
            with temp.open('xb') as stream:
                stream.write(compressed)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temp, target)
            temp.unlink()
            fsync_directory(self.directory)
        self.live.unlink()
        fsync_directory(self.directory)
