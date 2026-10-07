# Projected-goal CRL execution contract

This document describes execution mechanics only. The scientific design in
`canonical.json` is byte-frozen with SHA-256
`659e139034d9f3aed25a07d5244bc6ac4c86ce63dd0394f315ca5341ae0a78fc`.
The runner verifies that hash, the preparation commit, the pinned CRL source
hashes, and the historical frozen-artifact manifest before training can begin.

## Components

- `runner.py` owns one physical collector and the six-policy matrix.
- `replay.py` is append-only and samples a uniform eligible source followed by
  a strict same-episode future proportional to `0.99**lag`.
- `checkpointing.py` writes a single atomic schema-versioned checkpoint.
- `diagnostics.py` inspects the already sampled batch; it has no sampling RNG.
- `evaluation.py` creates a separate deterministic environment for the complete
  Development split.
- `execution.py` owns provenance, the dedicated track stream, locks, state
  transitions, and the Validation/paper gates.

The public simulator, projected-goal adapter, learner, losses, architecture,
and sampler semantics are not modified by this infrastructure.

## Exact schedule

Transitions 1 through 10,000 populate replay without an optimizer update. One
complete actor/alpha/critic cycle follows transitions 10,040, 10,080, and every
40 transitions through 300,000 inclusive. This is exactly 7,250 cycles per
policy. Batch size is 256. Collection and deterministic evaluation use the
canonical command `[1,1,1]`; actor and critic learning use the same sampled
projected strict future.

## Training track stream

Each policy has an independent NumPy PCG64 stream constructed from the pair
`[training_seed, 0x43524C31]`. Draws equal to historical exploratory tracks
1000–1008, Development 2000–2008, Validation 3000–3008, or paper-final
4000–4017 are rejected. The bit-generator state and every used reset seed are
checkpointed. The design does not claim identical realized trajectories across
policies.

## Checkpoint and interruption semantics

Checkpoint schema 1 stores actor, both critic encoders, alpha, all optimizer
states, replay rows and indices, collector episode bookkeeping, both JAX RNG
keys, future-sampler RNG, track RNG, the full Gymnasium environment, current
observation and raw outcome, diagnostics, physical episode outcomes, counters,
completed Development evaluations, source/config hashes, and runtime versions.
Writes use a temporary file, `fsync`, and atomic `os.replace`.

The Gymnasium environment was verified on Development seed 2002: after a
pickle round trip the next fixed-action transition was bit-identical. Unit
tests additionally prove that checkpoint load reproduces the next environment
transition, sampler RNG draws, deterministic prediction, and complete learner
update.

A caught interruption writes an exact checkpoint and marks the policy
`INTERRUPTED`; `resume` verifies its hash and continues the same attempt.
Resume also requires the currently checked-out configuration, scientific
sources, and execution sources to match the manifest provenance.
`COMPLETED` and `RUNNING` policies cannot be started again. A hard process or
host failure between durable checkpoints may leave `ACTIVE.lock` and a
`RUNNING` manifest without a current exact checkpoint. Such an attempt is not
silently rolled back: execution remains blocked, durable artifacts and the
last durable resource counters are preserved, and a fresh restart requires
explicit recorded authorization. Transitions after the last durable counter
cannot be reconstructed after a hard host failure and must be declared in the
restart provenance rather than silently treated as zero.

## Evaluation gates

At 50k, 100k, 150k, 200k, 250k, and 300k the current policy is atomically
checkpointed and evaluated with deterministic actions on all Development
tracks 2000–2008. Evaluation receives no training RNG object, uses a separate
environment, never enters replay, and the runner asserts that collection,
future-sampling, and update RNG states remain unchanged. Development cannot
select a checkpoint; 300k is always final.

Validation remains sealed until all six manifest entries are `COMPLETED`, each
has exactly 300,000 transitions and 7,250 updates, all six Development results
exist, and checkpoint/config/source hashes match. The gate is single-use and
this runner exposes no command that opens it. The paper gate always raises.

## Diagnostics

Every learning batch records pair coverage, decision/physical lag, progress
distance, terminal/late/unsafe/timely-safe futures, exact projected-goal
duplicates, requirement-bit frequencies, canonical-positive support, and
safe/timely progress. The first simulator transition that makes an actually
observed canonical future available is recorded separately from the first
learning batch that samples one. Losses and optimization scalars are recorded
every update; gradient norms, embedding/score statistics, collapse indicators,
and canonical-versus-sampled query behavior are computed from the same batch at
a fixed interval. Nothing is fed back into learning. Reference columns are
reference-distribution samples, not unreachable-negative labels.

Physical canonical success is recorded separately and only through
`EpisodeMetrics` plus `is_feasible`. Contrastive pairs and intermediate goals
never count as task success; energy is reported only for genuinely feasible
episodes.

## Authorization and final Validation

`python -m benchmarks.contrastive_rl.runner preflight` performs a read-only
check. `run`, `resume`, and `validate` require both authorization flags in
`preparation_status.json` to be true. They are true only because the six-policy
matrix and its single final Validation pass were explicitly authorized after
the execution-ready commit. No run is started by installation, import, tests,
or preflight. `validate` first enforces the frozen all-six-complete gate, marks
Validation opened before evaluating, writes each policy result atomically, and
cannot be invoked a second time. A failed Validation attempt remains opened and
must not be silently rerun.
