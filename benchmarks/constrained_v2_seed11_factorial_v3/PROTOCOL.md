# Frozen protocol

V3 descends from preserved V2 terminal commit
`ad2ad65df655ef1797e52e9e9f8f82b70b1c5d75`. The frozen algorithm and task are
validated by `config.py` against the Constrained V2 configuration hash.

## Preparation gates

Training is authorized only when all of these machine-checked conditions hold:

- `floor(300000 / 347) + 1` derives exactly 865 reachable task exposures;
- the complete `T_HIST` and `T_CURRENT` prefixes reproduce their frozen hashes;
- the first 18 accepted candidates from the `9100000` namespace are unique and
  clean by route and full-task content against both complete streams;
- candidates are also content-clean against statically opened Development and
  Validation tasks, while Paper remains completely unconstructed;
- the known V2 seed `9000005` collision is rediscovered at `T_CURRENT[559]`;
- A/B/C/D learner parameters, optimizer state, alpha, and PID state initialize
  identically and reproduce expected current-runtime hashes;
- collector-order, exact budget, recovery-checkpoint, and diagnostic
  non-interference tests pass;
- projected serial walltime is at most 24 hours and retained disk at most 15 GB.

## Execution and recovery

Run A, B, C, then D serially. Recovery checkpoints are atomically written and
verified every 10,000 real transitions; scientific checkpoints are retained every
50,000. A verified checkpoint contains policy and optimizer state, PID state,
replay, environment and collector state, global RNG states, counters, diagnostics,
and Attempt identity. Automatic exact resume within the same Attempt is allowed.
Fresh restart is forbidden; an unresumable run ends as `FRESH_RESTART_REQUIRED`.

No performance-based stopping, split modification, hyperparameter change,
checkpoint selection, or collector change is allowed. Only the route-clean V3
Diagnostic set may be policy-evaluated. Development, Validation, and Paper policy
evaluation are fail-closed.

## Final interpretation

`finalize.py` applies the frozen factorial formulas and a descriptive classification
rule recorded in its source. The canonical-baseline recommendation targets
reproducibility rather than retrospective RSR maximization and must retain
`Route-clean Policy R` for future splits.
