# Source Registry

The synthesis starts from commit
`9802f9262ea7f753365fc4bdd20b1fe08379291d` on the linear completed-research
history. Previous scientific branches and artifacts are read-only.

## Production semantics

| Source | Role |
| --- | --- |
| `src/gym_longicontrol/domain/dynamics.py` | timestep, constant-acceleration position update, velocity clipping |
| `src/gym_longicontrol/domain/track.py` | right-continuous speed-limit lookup |
| `src/gym_longicontrol/domain/metrics.py` | right-endpoint speed measurement |
| `src/gym_longicontrol/domain/task.py` | inclusive feasibility inequalities |
| `src/gym_longicontrol/envs/longicontrol.py` | integration/measurement/termination order |
| `benchmarks/content_split_audit/fingerprints.py` | canonical route and full-task hash definition |

## Task metadata ledgers

The nine explicit `LEDGER_SOURCES` in `feasibility.py` are the only task
serialization inputs. They cover completed constrained-collector, physical
reference, Hybrid, SlackTrader, causal-MPC, temporal-option, and
counterfactual-option studies. No reconstruction helper is called.

## Witness records

The explicit `WITNESS_SOURCES` in `feasibility.py` retain per-task metrics for
the physical reference, Hybrid Validation V2, causal MPC, Temporal Options,
the limited SlackTrader MPC development run, and Counterfactual normal-sampling
trajectories. Aggregate RSR-only reports are excluded from witness formation.

## Historical limitations

Older Scalar, Constrained V1/V2, requirement-conditioned, reward-search, HER,
GCSL, and model-based artifacts do not all retain canonical serialized route
content in the current commit. Their aggregate findings remain evidence for the
method registry, but they cannot be silently promoted to route-level witnesses.
The machine audit records only tasks whose exact content was already committed.

File byte hashes and result reconciliation are checked by `verify.py` and
reported in `INTEGRITY_REPORT.md`.

## Machine manifest and local checkpoint audit

`source_registry.json` freezes the source commit/tree and SHA-256 values for
the production semantics plus the immutable files behind mandatory headline
results. Because the base tree identifies the entire historical snapshot, the
verifier also rejects every branch change outside this synthesis directory.

The same manifest records a byte-only availability audit for Hybrid and
Constrained V2 checkpoint candidates. Hybrid hashes were already embedded in
the historical Validation provenance. Constrained V2 hashes were measured from
the locally retained final files during synthesis but were not embedded in the
original results; this distinction is preserved and no model is deserialized.
