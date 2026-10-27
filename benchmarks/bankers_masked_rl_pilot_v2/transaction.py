"""Same-Attempt deterministic event reconstruction, without fresh restarts."""

import json
import time
from pathlib import Path

from .checkpointing import load_state, save
from .integrity import atomic_json, canonical_bytes, digest, identity, require
from .journal import Journal
from .verification import Auditor, check_task_identity


class Engine:
    def __init__(self, directory, runner, *, checkpoint_interval=1000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.interval = checkpoint_interval
        self.journal = Journal(self.directory / 'journal', runner.attempt, runner.identity)
        self.head = self.journal.genesis
        self.audit = Auditor(synthetic=runner.synthetic,
            initial_state=runner.fixture_initial_state,
            max_episode_steps=runner.environment._max_episode_steps,
            expected_identity=runner.identity)
        self.reconstructed_events = 0
        atomic_json(self.directory / 'STARTED.json', dict(attempt=runner.attempt,
            identity=runner.identity, method=runner.method, seed=11, started_unix=time.time(),
            native_budget=50000, update_budget=5000), immutable=True)
        require(runner.native == 0 and runner.pending_reset, 'Attempt must start before reset')
        self.checkpoint()

    def checkpoint(self):
        return save(self.directory / 'technical.pkl', self.runner,
                    audit=self.audit, journal_head=self.head)

    def step(self, *, hook=None, full_evidence=False, learn=True):
        require(self.runner.native < 50000, 'native budget exhausted before prepare/reset')
        hook = hook or (lambda _: None)
        prepared = self.runner.prepare()
        if self.runner.episode_step == 0:
            check_task_identity(self.runner.environment.unwrapped.track, self.runner.task)
        self.journal.intent(prepared, self.head)
        hook('after_intent')
        row = self.runner.advance(prepared=prepared, hook=hook,
                                  full_evidence=full_evidence, learn=learn)
        self.audit.verify(row, self.runner.environment.unwrapped.track,
                           self.runner.method, full=full_evidence)
        self.head = self.journal.append(row, self.head)
        hook('after_commit')
        self.journal.clear_intent()
        if self.runner.native % self.interval == 0:
            self.journal.rotate()
            hook('before_checkpoint')
            self.checkpoint()
            hook('after_checkpoint')
        return row

    @classmethod
    def recover(cls, directory, attempt, *, checkpoint_interval=1000,
                full_evidence=False, learn=True):
        directory = Path(directory)
        require(not (directory / 'COMPLETED.json').exists(), 'completed Attempt rerun forbidden')
        marker = json.loads((directory / 'STARTED.json').read_text())
        require(marker['attempt'] == attempt and marker['identity'] == identity(),
                'recovery STARTED identity')
        saved = load_state(directory / 'technical.pkl', None, attempt)
        self = object.__new__(cls)
        self.directory, self.runner, self.audit = directory, saved['runner'], saved['audit']
        self.interval = checkpoint_interval
        self.journal = Journal(directory / 'journal', attempt, self.runner.identity)
        self.head = saved['journal_head']
        require(self.audit.rows == self.runner.native, 'checkpoint audit/native cursor')
        checkpoint_n = self.runner.native
        checkpoint_bound = checkpoint_n == 0 and self.head == self.journal.genesis
        self.reconstructed_events = 0
        last_frame = None
        for frame in self.journal.records():
            row = frame['row']
            if row['native'] < checkpoint_n:
                if row['native'] == checkpoint_n - 1:
                    require(frame['sha256'] == self.head, 'checkpoint journal-prefix binding')
                    checkpoint_bound = True
                last_frame = frame
                continue
            require(checkpoint_bound, 'checkpoint prefix missing')
            prepared = self.runner.prepare()
            require(digest(prepared) == row['prepared_identity'], 'reconstructed action intent')
            reconstructed = self.runner.advance(prepared=prepared, full_evidence=full_evidence, learn=learn)
            require(canonical_bytes(reconstructed) == canonical_bytes(row),
                    'deterministic same-Attempt event mismatch')
            self.audit.verify(reconstructed, self.runner.environment.unwrapped.track,
                               self.runner.method, full=full_evidence)
            self.head = frame['sha256']
            self.reconstructed_events += 1
            last_frame = frame
        require(checkpoint_bound, 'checkpoint prefix unavailable')
        if self.journal.pending.exists():
            pending = json.loads(self.journal.pending.read_text())
            require(pending['attempt'] == attempt and pending['identity'] == identity(), 'pending identity')
            n = pending['prepared']['native']
            if n == self.runner.native:
                require(pending['previous'] == self.head, 'pending prefix binding')
                prepared = self.runner.prepare()
                require(canonical_bytes(prepared) == canonical_bytes(pending['prepared']),
                        'pending deterministic action/RNG binding')
                row = self.runner.advance(prepared=prepared, full_evidence=full_evidence, learn=learn)
                self.audit.verify(row, self.runner.environment.unwrapped.track,
                                   self.runner.method, full=full_evidence)
                self.head = self.journal.append(row, self.head)
                self.reconstructed_events += 1
            else:
                require(last_frame is not None and n == self.runner.native - 1
                    and digest(pending['prepared']) == last_frame['row']['prepared_identity']
                    and pending['previous'] == last_frame['previous'],
                    'ambiguous pending action; restart forbidden')
            self.journal.clear_intent()
        if self.journal.live.exists():
            self.journal.rotate()
        self.checkpoint()
        return self
