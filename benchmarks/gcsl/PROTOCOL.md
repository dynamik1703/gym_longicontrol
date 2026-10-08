# Frozen execution protocol

## Budget and schedule

Three policies use seeds 11, 29 and 47. Each receives exactly 300,000 native
0.1-second simulator transitions: 900,000 total. Replay prefill is 10,000. One
batch-256 supervised update occurs after every 40 new transitions, first at
10,040 and last at 300,000, for 7,250 cycles and 1,856,000 sampled tuples per
policy. This LongiControl-matched schedule equals 6.4 replay samples per
post-prefill transition; it is not the official GCSL optimizer schedule.

The policy's own stochastic rollouts are the only training data. Historical
policies, demonstrations, Constrained-V2, HER, CRL, reference controllers,
curricula and success seeds are prohibited. Energy is evaluation-only.

## Splits

- Training: a dedicated seeded stream in `StochasticTrack-v1` that rejects
  every seed in 1000-1008, 2000-2008, 3000-3008 and 4000-4017.
- Development: 2000-2008 at 50k, 100k, 150k, 200k, 250k and 300k.
- Validation: 3000-3008 once, only after all three policy checkpoints freeze.
- Paper-final: 4000-4017 remains sealed; the runner has no opening operation.

The primary outcome is per-policy final canonical Validation RSR (`x/9`), plus
a pooled descriptive `x/27`; tracks are not policy replicates. Secondary
outcomes are completion, deadline and speed compliance, mutually exclusive
failure modes, final progress, Development RSR, real canonical training
success onset and feasible-only energy. Energy is `null` with no feasible
episode.

## Diagnostics

Every sampled batch records tuple count, unique source coverage, physical lag
in steps and seconds, progress distance, terminal-future fraction,
requirement-bit composition, canonical targets, goal uniqueness/collisions,
progress support, action dispersion within repeated goal and exact state-goal
conditions, minimum distance to `[1,1,1]`, and batches containing canonical
supervision.

Learning diagnostics record exact action NLL, descriptive absolute action
error, predicted mean/std, sampled entropy estimate, parameter/gradient norms
and nonfinite counts. Fixed-state action queries compare canonical, sampled,
safe/timely, safe/late, unsafe/timely and unsafe/late goals. Diagnostic sampling
uses a private constant RNG and cannot advance policy, replay or track RNGs.

## Attempt and checkpoint rules

The fixed root is `runs/gcsl-v1`; its immutable study ID is
`longicontrol-gcsl-v1`. Atomic creation prevents duplicate/free reruns. An
exclusive lock prevents concurrent writers. Each policy transitions through
`NOT_STARTED`, `RUNNING`, `INTERRUPTED`, and `COMPLETED`.

Every Development artifact contains an atomic exact-resume checkpoint with
policy, optimizer, full replay and size, environment, current observation and
physical outcome, episode counters, policy/replay/track RNG states, track
history, diagnostics, outcomes, source hashes and dependency versions. An
exception preserves the attempt and an interruption checkpoint. Exact resume
requires identical hashes. A fresh restart requires a separately recorded
authorization; attempts are never silently discarded.

The `run`, `resume`, and `validate` commands require a clean
`research/gcsl` commit and all three authorization/readiness flags. Validation
also requires exactly three completed 300k/7,250 policies and all Development
checkpoints. In this preparation commit authorization remains off.
