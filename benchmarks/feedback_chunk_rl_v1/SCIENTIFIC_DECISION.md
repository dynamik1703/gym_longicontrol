# Scientific Decision

## Decision

`CHUNK_RL_INTEGRITY_STOP`

The six preregistered training runs completed exactly, but the frozen
Diagnostic could not evaluate any learned checkpoint without violating strict
checkpoint provenance. Historical Hybrid checkpoints were produced under
Python 3.13.16 / NumPy 2.5.4; the new learned checkpoints were produced under
Python 3.11.13 / NumPy 2.4.6. The frozen evaluator requires both checkpoint
families in one process and each loader requires its own exact runtime tuple.

The first learned pair was marked `RUNNING` before its checkpoint load failed.
Although the traceback locates failure before environment evaluation, the
frozen recovery contract treats a `RUNNING` scientific pair as non-resumable.
Resetting that pair, splitting the evaluator, monkey-patching runtime metadata,
or weakening the loader would be a post-freeze protocol change. None was done.

## What the evidence does and does not establish

- Exact zero-residual parity, physical gates A--L, exact partial-chunk recovery,
  and all resource preconditions passed before training.
- All six runs consumed 300,000 native transitions and 30,000 gradient updates
  with finite sampled optimizer diagnostics and no recovery event.
- All 252 completed non-learning/historical comparator episodes were feasible,
  deadline compliant, and exactly speed compliant. FixedPrice MPC V2 had the
  lowest mean feasible energy among those completed comparator blocks; H47 was
  the lowest-energy historical Hybrid.
- CHUNK3 and RESIDUAL1S have zero held-out Diagnostic episodes. Their per-seed
  RSR, energy, failure identities, deterministic action influence, and paired
  differences are unknown.
- Training action statistics show nonzero and correlated stochastic CHUNK3
  outputs, but cannot establish useful temporal reasoning or held-out benefit.

Consequently:

- CHUNK3 cannot be called the strongest observed RL method on the new panel.
- Whether chunking beat its matched one-second ablation is undetermined.
- Whether CHUNK3 beat AdaptivePrice or the strongest observed non-learning MPC
  is undetermined.
- Success Levels A, B, and C are unevaluable, not failed performance tests.
- Independent final Validation is not scientifically justified from this
  study. A future attempt would require a new, predeclared protocol that makes
  mixed checkpoint-runtime handling coherent before any performance episode;
  it must not resume or overwrite this ledger.

## Protected content and final scope

Validation performance episodes remain zero and `CHUNK_RL_VALIDATION_V1`
remains `FROZEN_UNOPENED`. Paper performance episodes and constructed Paper
content fingerprints remain zero; seeds 4000--4017 remain
`SEALED_UNINSPECTED`. No Validation or Paper conclusion is drawn.

The integrity failure is the final scientific outcome of V1 and is retained
without an energy-based reinterpretation.
