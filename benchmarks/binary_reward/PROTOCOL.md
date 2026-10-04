# Binary Success Reward V1 protocol

Status: **preregistered on 2026-10-04 before smoke testing or policy training**.

## Research question

Can a standard continuous-control learner solve the canonical LongiControl task
when the developer specifies only terminal success or failure? This isolates a
minimal task specification, not a new optimizer or a search for more methods.

Secondary questions are whether training ever leaves the zero-success regime,
when successes first occur, whether learning is abrupt or gradual, how strongly
it depends on the seed, which physical failure mode dominates, and how its
sample efficiency compares with frozen scalar and constrained formulations.

## Canonical task and reward

The immutable task is route completion within 140 seconds with maximum measured
speed excess exactly zero. The wrapper constructs `EpisodeMetrics` from terminal
`info` and calls the existing `is_feasible(metrics, task)` function. That
external evaluator is authoritative.

```text
all intermediate transitions: 0.0
terminal success:              1.0
terminal failure:              0.0
```

There is no failure penalty, progress, distance, velocity, completion-only,
deadline-only, speed-only, energy, time, slack, deficit, constraint, or shaping
reward. Public observations, dynamics, termination, and environment IDs remain
unchanged. No reward or observation normalization is applied.

## Algorithm

Only Stable-Baselines3 2.9.0 SAC is used. Its configuration is copied exactly
from frozen Scalar SB3 SAC: 256x256 ReLU actor/critics, learning rate `3e-4`,
replay capacity 1,000,000, 100 learning-start interactions, batch 256,
`tau=0.005`, `gamma=0.99`, one update after every interaction, and automatic
entropy tuning. No hyperparameter sweep or PPO fallback is permitted.

## Splits, seeds, budget, and checkpoints

- Training seeds: 11, 29, 47.
- Development: 2000--2008.
- Validation: 3000--3008.
- Historical 1000--1008: unused.
- Paper-final 4000--4017: sealed and untouched.
- Budget: exactly 300,000 native simulator interactions per seed.
- Deterministic external evaluation: 50k, 100k, 150k, 200k, 250k, 300k.

Each checkpoint evaluates all nine Development and nine Validation tracks in a
separate environment. Evaluation does not enter replay. The fixed 300k policy is
the primary result; there is no best-checkpoint selection or automatic budget
extension.

Every completed training episode records its ending interaction, binary reward,
physical outcome, and exact failure mode. This distinguishes never observing a
success from observing successes that do not generalize.

## Primary metrics

Final Validation Requirement Satisfaction Rate (RSR) is computed externally,
overall and per seed. Additional outcomes are completion, deadline compliance,
strict speed compliance, failure-mode counts, successful training-episode count,
first success step, gaps between successes, checkpoint learning curves, and
peak-to-final change. Reward return is never used as evaluation.

Frozen comparisons are descriptive and are not retrained:

```text
Scalar SB3 SAC:       5/27
Constrained V1:       0/27
Constrained V2:      21/27
Requirement V1:     48/135 (different five-requirement matrix)
```

## Predefined credibility criteria

Binary success alone is a credible minimal specification only if all hold at
the fixed final checkpoint:

1. pooled Validation RSR is at least 50% (14/27);
2. every training seed reaches at least 3/9 RSR;
3. at least two seeds reach at least 5/9 RSR;
4. every seed observes at least one successful training episode;
5. no seed drops by more than 3/9 from its best checkpoint to 300k.

These thresholds demand majority-track learning and multi-seed evidence while
remaining deliberately below the established 21/27 constrained result.

## Decision gates

- **A -- credible binary specification:** every criterion passes.
- **B -- partial but non-robust learning:** final pooled RSR exceeds frozen
  Scalar SB3 SAC (5/27), every seed is nonzero, but Gate A fails.
- **C -- seed-dependent sparse learning:** at least one seed reaches 5/9 while
  another is below 3/9 or the final RSR range is at least 5/9.
- **D -- no escape from zero:** no training success is observed and final RSR
  is 0/27.
- **E -- training success without generalization:** training successes occur,
  but final pooled RSR is at most the frozen Scalar SB3 SAC result (5/27).
- **F -- mixed sparse evidence:** any remaining outcome.

Gate order is A, D, C, B, E, F. The full learning curves are reported even when
the final gate is clear.

No HER, relabeling, CRL, constraints, dense shaping, deadline state, requirement
conditioning, reward tuning, final-test evaluation, or next research method is
allowed in this study.
