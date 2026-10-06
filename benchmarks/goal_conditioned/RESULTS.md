# Goal-conditioned SAC versus SAC+HER results

## Decision

This study reaches preregistered empirical outcome **B**: HER creates positive
virtual replay samples, but canonical success does not improve. Both final
conditions obtain **0/27 feasible Validation episodes**, so the paired RSR
difference `sac-her - sac-no-her` is **0/27 (0 percentage points)**.

This is a negative result for the tested constraint-preserving, position-only
HER formulation, SAC configuration and 300,000-transition budget. It does not
show that every hindsight or goal-conditioned method must fail.

## Frozen execution

- physical task: complete 1,000 m by 140 seconds with zero measured speed
  excess;
- learner: SB3 2.9.0 SAC with identical Dict observations and sparse
  first-arrival reward in both conditions;
- only intended difference: zero virtual goals versus four future,
  position-only relabeled goals per real-goal share;
- training seeds: 11, 29 and 47;
- budget: exactly 300,000 native simulator transitions per policy, including
  1,800 warmup transitions;
- updates: 298,200 per policy after warmup;
- Development: tracks 2000--2008 at 50k intervals;
- final Validation: tracks 3000--3008 once per final policy;
- reserved paper-test tracks 4000--4017: unopened.

All six final models were frozen before Validation. External
`EpisodeMetrics`/`is_feasible`, not replay reward, determine the reported
successes.

## Final Validation

| Condition | Seed | Feasible | Complete | By 140 s | Speed compliant | Mean progress | Feasible energy |
|---|---:|---:|---:|---:|---:|---:|---:|
| SAC, no HER | 11 | 0/9 | 0/9 | 0/9 | 9/9 | 67.05 m | null (0) |
| SAC, no HER | 29 | 0/9 | 0/9 | 0/9 | 9/9 | 11.89 m | null (0) |
| SAC, no HER | 47 | 0/9 | 0/9 | 0/9 | 9/9 | 29.33 m | null (0) |
| **SAC, no HER pooled** | -- | **0/27** | **0/27** | **0/27** | **27/27** | **36.09 m** | **null (0)** |
| SAC + HER | 11 | 0/9 | 0/9 | 0/9 | 9/9 | 0.91 m | null (0) |
| SAC + HER | 29 | 0/9 | 0/9 | 0/9 | 9/9 | 2.52 m | null (0) |
| SAC + HER | 47 | 0/9 | 0/9 | 0/9 | 9/9 | 14.22 m | null (0) |
| **SAC + HER pooled** | -- | **0/27** | **0/27** | **0/27** | **27/27** | **5.88 m** | **null (0)** |

Every episode reaches the 180-second finite horizon. All 54 mutually exclusive
failure labels are `incomplete+deadline`; maximum and integrated speed excess
are both zero in every final Validation episode. Energy over feasible episodes
is therefore undefined (`null`), not zero. Energy was not optimized.

The paired 27 seed-track outcomes are:

| Outcome | Count |
|---|---:|
| Both succeed | 0 |
| HER alone succeeds | 0 |
| No-HER alone succeeds | 0 |
| Neither succeeds | 27 |

The RSR result is tied, but terminal progress is lower with HER (5.88 m versus
36.09 m pooled). This progress difference is descriptive and is not a second
success threshold.

## Real training outcomes

| Condition | Seed | Canonical successes | First success | Completed episodes | Partial final episode |
|---|---:|---:|---:|---:|---:|
| SAC, no HER | 11 | 0 | null | 171 | 421 transitions |
| SAC, no HER | 29 | 0 | null | 167 | 106 transitions |
| SAC, no HER | 47 | 0 | null | 171 | 789 transitions |
| SAC + HER | 11 | 0 | null | 169 | 1,485 transitions |
| SAC + HER | 29 | 1 | step 96,533 | 169 | 518 transitions |
| SAC + HER | 47 | 0 | null | 166 | 1,200 transitions |

No-HER records 0/509 successful completed training episodes. HER records one
success in 504 completed episodes (0.198%), entirely in seed 29. A sampled real
success can occur multiple times in replay; the 60 positive real replay rows
for HER seed 29 are not 60 distinct physical successes.

Development RSR remains zero for every condition, seed and checkpoint. There
are transient completions: HER seed 29 completes 4/9 Development episodes at
100k but violates speed in all four, and no-HER seed 47 completes 1/9 at 300k
but also violates speed. Neither becomes a feasible Development trajectory.

## Replay diagnostics

Rates below use virtual rows as denominator except the real-positive rate,
which uses real rows. They describe samples actually used for learning; no
extra diagnostic batches were drawn.

| Seed | Eligible virtual | Relabeled virtual | Fallback | Positive virtual | Mean target distance | Positive real rows |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 97.090% | 97.037% | 2.910% | 0.0792% | 91.17 m | 0 |
| 29 | 98.676% | 98.663% | 1.324% | 0.0843% | 37.88 m | 60 |
| 47 | 99.203% | 99.203% | 0.797% | 0.0820% | 44.98 m | 0 |
| **Pooled** | **98.323%** | **98.301%** | **1.677%** | **0.0818%** | **58.01 m** | **60** |

Across HER policies, 182,498,400 virtual rows are sampled. Of these,
179,437,694 are eligible, 179,397,817 actually change the target, 3,060,706
fall back to the original goal and 149,320 receive positive virtual reward.
The intended 4/5 HER ratio becomes an observed virtual-row fraction of
79.6875% because batch-size integer rounding yields 204 virtual and 52 real
rows in each 256-row batch.

Constraint preservation removes a substantial part of the otherwise useful
signal: 39,559,578 virtual rows (21.677%) fail the fixed deadline check and
6,449,781 (3.534%) fail the accumulated violation-history check. The positive
virtual fraction remains about 0.082%. These virtual successes demonstrate a
nonzero learning signal, but do not transfer to final canonical 1,000-m
success.

## Interpretation

The no-HER arm does not discover a real success. HER does provide numerous
valid intermediate first-arrival targets and one real training success, yet its
final deterministic policies converge to safe near-standstill behavior. Seed
29 also shows transient fast-but-unsafe Development completion before
regressing to non-completion. Thus HER neither consistently acquires the
completion/deadline conjunction nor improves canonical Validation RSR.

The matched result isolates the preregistered replay difference. Comparison to
historical Binary V1 is contextual only because goal information, warmup and
timeout treatment differ. No goal redesign, HER tuning, extra algorithm or
budget extension follows from this result.

## Interaction accounting and provenance

- main-study training: 1,800,000 simulator transitions;
- gradient updates: 1,789,200 total;
- Development evaluation: 577,306 simulator transitions;
- Validation evaluation: 97,200 simulator transitions;
- all evaluation interactions combined: 674,506;
- paper-test interactions: zero.

The initial execution attempt stopped before a recorded training transition
because a runner-only terminal-observation assertion compared unclipped route
overshoot with a clipped goal coordinate. The preserved attempt opened neither
Validation nor paper tracks. Commit `ef53e8e` corrected only that assertion,
added explicit authorized-restart provenance, and the full study then ran in a
fresh protected directory. This incident does not consume or alter the
1.8-million-transition main-study budget.

The one permitted Validation pass stored compact episode metrics rather than
full transient trajectories. Consequently no representative trajectory plot
is claimed or reconstructed: rerunning Validation for a cosmetic artifact
would violate the frozen once-only rule. The three tracked plots cover
Development RSR, final requirements/failure modes, and real versus virtual
training signal.

Reproducibility artifacts:

- `results.json`: integrity-checked summaries and replay accounting;
- `validation_episodes.json`: all 54 compact Validation episodes;
- `execution_manifest.json`: exact execution provenance and model hashes;
- `analysis.py` and `plot_results.py`: deterministic artifact generation;
- `plots/`: plots generated only from stored compact results;
- ignored `runs/goal-conditioned-her-v1-restart-1/`: models, checkpoints and
  raw run records.
