# Execution recovery

This document describes execution infrastructure only. The scientific files and
configuration listed in `scientific_freeze.json` remain byte-for-byte frozen.

## Ordering and cadence

The worker performs each real transition as one deterministic transaction:

1. complete the environment transition and real-episode PID update, if due;
2. complete the scheduled model refresh and synthetic generation, if due;
3. complete all scheduled RL updates;
4. run Development only at the frozen 50k multiples;
5. write and verify a recovery checkpoint at every 10k multiple.

Thus 50k checkpoints have both scientific and recovery roles, while the other
10k multiples perform no Development evaluation. A pre-refresh heartbeat is
written from already-known state because a learned-model refresh may run for a
long time. Heartbeats neither sample nor advance an RNG.

## Exact state and provenance

Schema 2 stores the policy and targets, all optimizers, automatic entropy state,
PID/Lagrange state, real and synthetic replays with their counters, learned-model
members/optimizers/elites/normalization, current environment and collector,
training-track state, episode state, counters, completed Development checkpoints,
and Python/NumPy/PyTorch plus model/synthetic RNG state. It also stores the frozen
configuration and scientific hashes, execution-source hashes, runtime versions,
training seed and attempt identity.

A resume is accepted only after the file SHA-256, schema, configuration, source,
runtime, seed, condition and attempt identity agree. It continues the same attempt;
a fresh attempt requires a separately recorded authorization.

## Atomicity and retention

Checkpoint data is written to a sibling temporary file, flushed, fsynced, atomically
replaced and followed by a directory fsync. The committed file is loaded again and
its counters are checked before metadata and the `latest.json` pointer are published.
The two newest non-scientific recovery checkpoints are retained. Frozen 50k
scientific checkpoints are not removed by recovery retention.

## Detached execution

`detached_launcher.py` installs a one-shot per-user `launchd` job. `KeepAlive` is
disabled, so a failed worker is never silently restarted. `matrix_worker.py` executes
the exact six-policy plan serially, records launcher/worker PID, process group,
session, command, timestamps, exit code or observed signal, and stops on the first
failure. Each policy has a persistent `worker.log`; launcher stdout and stderr are
also persistent.

After a crash, `recovery_admin.py resume-ready` will classify an attempt as resumable
only after verifying the latest exact checkpoint. Otherwise recovery stops for an
explicit decision; it never performs an automatic fresh restart.
