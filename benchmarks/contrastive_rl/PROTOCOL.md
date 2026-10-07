# Frozen pre-training design — projected-goal Contrastive RL

Status: **SEMANTICALLY FROZEN / EXECUTION DISABLED**. The task mapping and
matched settings are fixed below, but this document does not authorize training.
Execution infrastructure and a separate explicit authorization are still
required.

## Research questions

1. Does projected-goal CRL produce canonical first-arrival success?
2. Does residual depth 16 improve RSR over depth 4 under matched rules?
3. Is any difference repeatable across seeds and tracks?
4. Does improved contrastive association correspond to improved control?

## Fixed comparison

- Conditions: depth 4 and depth 16, counted only as Dense layers inside
  four-layer residual blocks.
- Seeds: 11, 29, 47.
- Width 256; embedding 64; LayerNorm; Swish; residual connections.
- Actor, state-action encoder, and goal encoder scale together.
- Goal dimension 3; policy-state dimension 12; action dimension 1.
- Batch **256**, explicitly frozen from the pinned code default. The paper table
  and repository example use 512; this disclosed difference is not tuned.
- Adam learning rates `3e-4`; gamma `.99` per decision; target entropy
  `-0.5 * action_dim`; logsumexp penalty `.1`.
- Fixed-width depth scaling is neither parameter- nor FLOP-matched.
- Depth 16 runs independently of depth-4 success.

This 300k tier is a low-budget LongiControl adaptation, not a replication of
the paper's 100M–400M-transition scale. Its failure cannot refute depth scaling.

## Goal and sampling rules

- Real collection and deterministic evaluation command `[1,1,1]`.
- Critic positives use projected raw future outcomes recorded strictly later in
  the same uninterrupted physical episode, sampled proportional to `.99**lag`.
- Actor training uses those sampled projected future outcomes, not the fixed
  collection command, matching the pinned scaling implementation.
- At `dt=0.1 s`, decision lag `k` means `0.1k` physical seconds; absolute
  deadline time is never restarted.
- Every in-batch goal column remains a reference sample. Duplicate columns are
  measured and not masked or given multi-positive labels.
- Completion terminates. Its real terminal outcome may be sampled once as a
  future, never repeated or used as a post-terminal source. Timeout remains a
  failed observed outcome.
- Training tracks use a dedicated RNG stream excluding Development, Validation,
  and paper-test seeds.

The real-rollout goal treatment differs from the historical HER study, so the
historical HER/CRL comparison is not a one-factor causal test.

## Exact transition and update budget

Each policy receives exactly 300,000 native simulator transitions, including
10,000 prefill transitions. Transitions 1 through 10,000 populate replay and
perform no optimizer update. Thereafter one complete ordered cycle

```text
actor update -> alpha update -> critic update
```

occurs after transitions 10,040; 10,080; ...; 299,960; and 300,000. Thus

```text
(300000 - 10000) / 40 = 7,250 complete update cycles per policy
```

The final transition is included and followed by the final cycle. With batch
256 this samples 6.4 replay rows per new post-prefill transition, closely
matching the pinned implementation's aggregate sample-use accounting. Replay
capacity is 300,000 scalar transitions; upstream's 10,000 rows each contain 512
environment transitions and are not equivalent.

Historical Goal/HER SAC used train frequency 1 and one gradient step after each
eligible interaction, roughly forty times more update cycles after warmup.
Equal simulator interactions are therefore not equal optimizer or compute
budgets.

Total first-tier budget is `2 depths × 3 seeds × 300,000 = 1,800,000` planned
main-study transitions. No increase or additional main depth is authorized.

## Checkpoints and evaluation

- Development checkpoints: 50k, 100k, 150k, 200k, 250k, 300k.
- Development tracks: 2000–2008, deterministic evaluation with RNG isolated
  from collection and replay sampling.
- Final policy is the 300k checkpoint, never best-on-Development selection.
- Only after all six policies and hashes are frozen may Validation 3000–3008 be
  opened once under a future authorization.
- Paper tracks 4000–4017 remain sealed.

## Restart and resource accounting

Every attempt gets an immutable provenance record. Interrupted attempts and all
their consumed simulator transitions, update cycles, wall time, checkpoints,
and technical retries are preserved; there is no silent deletion or free rerun.

Exact resume requires matching model parameters, all optimizer states, replay
contents and positions, sampler RNG, environment/collector state, transition
counters, source/config hashes, and evaluation state, or a documented
equivalent that demonstrably reproduces the next transition and update. If
exact resume is unavailable, execution stops and requests an explicit
provenance-recorded restart decision. Reports distinguish the planned 1.8M
main-study transitions from all resources consumed across attempts. Historical
accounting is untouched.

## Required diagnostics

Record native transitions and complete update cycles; real canonical training
successes; RSR/completion/deadline/speed outcomes by seed and track;
feasible-only energy; source coverage and future lags; valid/invalid projected
goal composition; duplicate rates; positive/reference scores; embedding norms
and collapse; actor/critic gradients; entropy/alpha; and `[1,1,1]` query
behavior. Diagnostic RNG must not perturb training.

## Execution gate

Semantic and numerical mapping readiness is satisfied. Main execution remains
disabled because an end-to-end collector/replay/checkpoint runner with the exact
resume contract has not been implemented or verified, and this task grants no
training authorization.
