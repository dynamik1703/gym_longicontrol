# Validation V2 Exclusive-Execution Correction

## Scope

This pre-authorization correction changes orchestration only. It adds one
durable, non-reclaimable single-writer lease around the already registered V2
transaction, validates each frozen evaluator result before persistence, and
provides aggregation-only final reporting.

The lease is created atomically with `O_CREAT|O_EXCL`, fsynced with its parent
directory, and bound to the existing transaction ID, PID, host, acquisition
time, and ownership token. All evaluator paths verify the on-disk owner before
pair selection, before `STARTED`, and before evaluator entry. The file is never
deleted or automatically taken over. A crash leaves a permanent blocker; a
successful transaction changes it to `COMPLETED_SEALED`.

## Scientific non-interference

No controller, actor, learner, reward, cost, SMDP transition, Shield, causal
fallback, non-learning strategy, simulator, checkpoint, split, registered pair,
metric definition, or acceptance threshold is changed. The original frozen
single-arm evaluator is called without altered arguments. The correction can
change only whether an evaluation is allowed to start, the durable provenance
around that call, validation of the returned record, and read-only aggregation
after completion. It cannot change the controllers' mathematical action or the
simulator trajectory.

Exact correction-source and frozen-scientific-source hashes are recorded in
`execution_correction.json` after formatting and verification.
