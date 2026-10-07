# DRAFT protocol — shallow versus deeper Contrastive RL

Status: **DRAFT / EXECUTION DISABLED**. This document records matched settings,
but cannot be frozen until the actor-side goal-set estimator in `DESIGN.md` is
resolved and tested. It authorizes no training.

## Research questions

1. Does the finalized CRL task specification produce canonical success?
2. Does greater residual depth improve RSR under matched rules and budget?
3. Is any difference repeatable across seeds and tracks?
4. Do association-score diagnostics correspond to actual control outcomes?

## Planned factorial comparison

- Conditions: residual depth 4 versus 16.
- Seeds: 11, 29, 47.
- Width: 256; embedding: 64; LayerNorm; Swish; residual block every four
  internal Dense layers.
- Actor, state-action encoder, and goal encoder scale together.
- Input/output projections are excluded from the named depth and reported
  separately.
- Fixed width means depth 16 has more parameters and compute; this is neither
  parameter-matched nor FLOP-matched.
- Budget: 300,000 native simulator transitions per policy, including prefill;
  1,800,000 total first-tier transitions.
- No shallow-success gate: both depths run if and only if semantics are ready.

This tier is a low-budget LongiControl adaptation, not a replication of the
paper's 100M–400M-transition experiments.

## Draft update accounting

The pinned code collects `62 × 512 = 31,744` transitions and performs 800
minibatch updates. With batch 256 that is about 0.0252 gradient updates and 6.45
sampled replay items per collected transition. A serial equivalent is one
batch-256 update after every 40 native transitions (6.4 replay samples per new
transition). Subject to semantic readiness, the draft fixes that schedule for
both depths, uses a 10,000-transition prefill included in budget, and never
changes batch size after seeing control performance. The prefill is a
LongiControl resource adaptation: upstream's 1,000 rows × 512 environments is
512,000 transitions and exceeds this entire tier.

Replay capacity is provisionally 300,000 transitions. This differs from
upstream's `max_replay_size=10000`, which counts time rows each containing 512
environment transitions (up to 5.12M slots), not 10,000 scalar transitions.
Adam learning rates are `3e-4`, gamma is `.99` per decision, target entropy is
`-0.5 * action_dim`, and the logsumexp penalty is `.1`.

Batch 256 follows the pinned code default; the README example and paper table
use 512. The final frozen protocol must resolve this disclosed source mismatch
before training, without a performance sweep.

## Data, goals, and sampling

- Training tracks come from a dedicated RNG stream excluding all Development,
  Validation, and paper-test seeds.
- Positive future goals are strict future exact outcomes from the same episode
  and uninterrupted collector segment, sampled proportional to `.99**lag`.
  At simulator `dt=0.1 s`, one decision lag is 0.1 physical seconds.
- Reference goals are all in-batch goal columns; they are samples from a
  reference distribution, never claims of physical unreachability.
- Real rollout and actor goal distributions are **TBD BLOCKERS**. They must be
  identical across depths and cannot fabricate canonical success.
- No absorbing completion state is introduced. A real terminal completion may
  be a future goal for an earlier source but is never repeated; timeout remains
  an observed failure outcome.
- This real-rollout goal treatment will differ from the historical HER study;
  HER versus CRL is therefore not a one-factor causal comparison.

## Checkpoints and evaluation

- Development checkpoints: 50k, 100k, 150k, 200k, 250k, 300k.
- Development tracks: 2000–2008, deterministic actions, evaluation RNG isolated
  from collection/sampling RNG.
- Final checkpoint rule: 300k, not best-on-Development.
- After every one of the six policies and source/config hashes is frozen,
  Validation 3000–3008 is opened once for deterministic evaluation.
- Paper tracks 4000–4017 remain sealed.
- A completed `(depth, seed)` manifest prevents duplicates. Partial runs resume
  only from an atomic checkpoint with matching source/config/model hashes;
  otherwise they are discarded and rerun within the same fixed budget.

## Required diagnostics

Record native transitions, gradient updates, real canonical training successes,
RSR/completion/deadline/speed metrics by seed and track, feasible-only energy,
pair counts and unique source coverage, lag seconds and target distances,
valid/invalid outcome composition, duplicate rates, positive/reference scores,
embedding norms/collapse, gradients, entropy/alpha, and canonical goal-region
queries after their estimator is defined. Diagnostic computation must use a
separate RNG and may not alter training samples.

## Execution gate

Main training remains disabled until `task_mapping_verified` is true, actor and
real-rollout goal distributions are frozen, set aggregation has synthetic
parity tests, and the batch-size source mismatch is resolved. No budget increase
or additional main depth is authorized here.
