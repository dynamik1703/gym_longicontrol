# Hybrid RL Validation V2 Protocol

## Scope and state

`HYBRID_RL_VALIDATION_V2` is a newly frozen, post-incident 36-task panel.
Preparation constructs task content and verifies immutable policies but runs no
controller trajectory. The committed authorization contract keeps
`preparation_lock=true`, `v2_validation_authorized=false`, and
`v2_validation_enabled=false`.

V1 remains `INCIDENT_ABORTED`; it can never return to `FROZEN_UNOPENED`.
Paper remains `SEALED_UNINSPECTED`. Paper seeds 4000–4017 are only numeric
exclusions and their task content is never constructed or fingerprinted.

## Split construction

Starting at numeric seed 10,200,000, select the first 36 candidates passing
Route-clean Policy R. Use the existing canonical generator, V2 semantics, and
content-fingerprint implementation. Reject a candidate on any numeric, route,
or full-task collision with protected content, including all 2,595 reachable
Hybrid training rows, historical opened evaluations, physical-reference
Development/Holdout, Hybrid Diagnostic V1, and all 36 V1 Validation tasks.

Every candidate, rejection reason, content hash, and algorithm-independent
canonical task serialization is retained in `split_ledger.json`.

## Frozen comparison

The later authorized comparison contains exactly five identities:

1. `CAUSAL60`;
2. `TIME_AWARE` through the identical frozen Shield;
3. `H11` at the frozen 300k endpoint;
4. `H29` at the frozen 300k endpoint;
5. `H47` at the frozen 300k endpoint.

There are 36 × 5 = 180 unique controller-task pairs. No Oracle is included.
The original `evaluate_validation` convenience function is prohibited.

## Exactly-once transaction

`pair_ledger.json` registers all 180 identities before authorization. Pair
states are `PENDING`, `STARTED`, `COMPLETED`, and
`INTERRUPTED_UNRECOVERABLE`. The ledger is atomically replaced and both file
and directory are fsynced.

For future evaluation, `STARTED` must be durably readable before the evaluator
may create or reset an environment. A crash or exception after `STARTED`
becomes `INTERRUPTED_UNRECOVERABLE`; the pair is never silently rerun. A
completed pair contains an immutable result hash and can never start again.
Aggregation recovery reads completed results only and invokes no evaluator.

## Frozen acceptance hierarchy

For each Hybrid seed:

- feasible route-success rate: at least 35/36;
- arithmetic mean of paired per-track energy savings versus `CAUSAL60`: at
  least 10%;
- arithmetic mean of paired per-track energy savings versus `TIME_AWARE`: at
  least 3%.

Energy comparisons include only jointly feasible trajectories. Track instances
are the statistical sampling units; learned seeds are replications. Thresholds
must not change in response to V1 or V2 results.
