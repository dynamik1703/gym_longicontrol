# Diagnostic Results

## Status

The scheduled Diagnostic ended in a fail-closed integrity stop. Of 900 frozen
controller/task pairs, 252 comparator pairs completed and their result and
native-evidence hashes were independently verified. The first learned pair is
left `RUNNING` in the immutable ledger and 647 pairs remain `NOT_STARTED`.
No learned-treatment Diagnostic episode executed.

The exact stopped identity is:

- Controller: `CHUNK3_RESIDUAL_SAC__S11__N100000`
- Numeric task seed: `11200000`
- Failure point: learned checkpoint loading, before
  `evaluate_learned_seed` was called.
- Loader error: `RuntimeError: checkpoint provenance changed: runtime_versions`

The historical Hybrid checkpoints require Python 3.13.16 and NumPy 2.5.4;
the newly trained checkpoints require Python 3.11.13 and NumPy 2.4.6. Both
require Gymnasium 1.4.0, Torch 2.14.1, and gym-longicontrol 1.0.0. The frozen
evaluator loads both checkpoint families into one process and compares each
stored runtime tuple against that one process. Consequently, no legitimate
single runtime can pass both strict provenance checks.

The protocol forbids rerunning an ambiguous `RUNNING` scientific pair. The
ledger was not reset, no loader was monkey-patched, and no version check was
bypassed. This preserves the failure as part of the scientific result.

## Completed comparator panel

All entries below use the same 36 route-clean
`CHUNK_RL_DIAGNOSTIC_V1` tasks. RSR is the raw feasible count. There were no
failure identities among the 252 completed comparator episodes.

| Controller | RSR | Complete | Deadline compliant | Exact speed compliant | Mean feasible energy kWh | Median feasible energy kWh | Mean travel s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAUSAL60 | 36/36 | 36 | 36 | 36 | 0.201985 | 0.199211 | 81.422 |
| TIME_AWARE | 36/36 | 36 | 36 | 36 | 0.177770 | 0.174595 | 127.747 |
| FixedPrice MPC V2 | 36/36 | 36 | 36 | 36 | 0.168263 | 0.166574 | 97.978 |
| AdaptivePrice MPC V2 | 36/36 | 36 | 36 | 36 | 0.168448 | 0.166391 | 99.586 |
| Frozen Hybrid H11 | 36/36 | 36 | 36 | 36 | 0.176307 | 0.172830 | 127.697 |
| Frozen Hybrid H29 | 36/36 | 36 | 36 | 36 | 0.175739 | 0.174427 | 127.461 |
| Frozen Hybrid H47 | 36/36 | 36 | 36 | 36 | 0.175083 | 0.172508 | 128.192 |

On these completed data, FixedPrice has the lowest mean feasible energy and
H47 the lowest mean among the historical Hybrid policies. This is a descriptive
comparison of completed comparator blocks, not evidence about CHUNK3.

## Learned treatments

CHUNK3 and RESIDUAL1S have zero Diagnostic episodes at 100k, 200k, and 300k.
Therefore per-seed RSR, failure identities, feasible-only energy, paired
savings, deterministic action influence, full-chunk execution, cancellation,
intervention, and mechanism statistics are all **not estimable**. Training
statistics are reported separately and are not substituted for held-out
evaluation.

## Protected panels

- Final Validation performance episodes: 0.
- Paper performance episodes: 0.
- Paper content fingerprints constructed: 0.
- Validation remains `FROZEN_UNOPENED`.
- Paper seeds 4000--4017 remain `SEALED_UNINSPECTED`.
