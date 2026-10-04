# Binary Success Reward V1 results

## Decision

Binary Success Reward V1 reaches **Gate D -- no escape from zero**. Across
three independently trained SB3 SAC policies, 900,000 native simulator
interactions and 604 completed training episodes, the learner never observes a
positive reward. The fixed 300k policies achieve **0/27 feasible Validation
episodes**.

This answers the preregistered primary question negatively for the tested
algorithm, budget and task: a terminal success bit alone is not sufficient for
standard SB3 SAC to learn the canonical LongiControl task under this protocol.
It is not a claim that binary specifications are impossible with every
algorithm, exploration scheme or interaction budget.

## Frozen design

- task: complete the route within 140 seconds with zero measured speed excess;
- reward: `1.0` only when terminal `is_feasible(EpisodeMetrics, task)` is true,
  otherwise `0.0` on every transition;
- learner: Stable-Baselines3 SAC with the frozen Scalar SB3 SAC configuration;
- training seeds: 11, 29 and 47;
- budget: exactly 300,000 simulator interactions per seed;
- deterministic checkpoints: every 50,000 interactions;
- Development tracks: 2000--2008; Validation tracks: 3000--3008;
- reserved paper-final tracks 4000--4017: not evaluated.

No reward shaping, negative failure reward, partial requirement reward, HER,
goal relabeling, constraint signal, deadline slack or requirement-conditioned
observation is present. The environment reward is not used as the benchmark
metric.

## Primary result

| Training seed | Feasible | Complete | Deadline compliant | Speed compliant | Training successes |
|---:|---:|---:|---:|---:|---:|
| 11 | 0/9 | 0/9 | 0/9 | 9/9 | 0/207 |
| 29 | 0/9 | 0/9 | 0/9 | 9/9 | 0/179 |
| 47 | 0/9 | 0/9 | 0/9 | 9/9 | 0/218 |
| **Pooled** | **0/27** | **0/27** | **0/27** | **27/27** | **0/604** |

Every final Validation episode reaches the unchanged 180-second environment
horizon without completing the route. All 27 final failures are classified as
`incomplete+time`; none violates the speed requirement.

## Learning dynamics

| Interactions | Feasible | Complete | Deadline compliant | Speed compliant | Dominant Validation failure |
|---:|---:|---:|---:|---:|---|
| 50k | 0/27 | 27/27 | 27/27 | 0/27 | `speed` (27/27) |
| 100k | 0/27 | 0/27 | 0/27 | 27/27 | `incomplete+time` (27/27) |
| 150k | 0/27 | 0/27 | 0/27 | 27/27 | `incomplete+time` (27/27) |
| 200k | 0/27 | 0/27 | 0/27 | 27/27 | `incomplete+time` (27/27) |
| 250k | 0/27 | 0/27 | 0/27 | 27/27 | `incomplete+time` (27/27) |
| 300k | 0/27 | 0/27 | 0/27 | 27/27 | `incomplete+time` (27/27) |

Learning does not emerge suddenly or gradually: Validation RSR is zero at every
checkpoint for every seed. At 50k, deterministic policies complete quickly
(pooled mean 42.83 seconds) but exceed the speed limit. By 100k, all three have
moved to the opposite trivial failure regime and remain there through 300k.

The raw training outcomes contain 459 `incomplete+time`, 137 `speed`, seven
`incomplete+time+speed`, and one `time+speed` episode. Thus exploration reaches
both major failure regions, but never their successful intersection.

SAC diagnostics are consistent with the absence of a learnable positive target:
final critic losses are approximately zero and automatic entropy coefficients
fall to between `3.5e-7` and `1.4e-6`. The final deterministic policies are not
all numerically identical: seed 11 remains at zero position, while seeds 29 and
47 reach at most 23.60 m and 78.75 m across Validation and exhibit substantial
action oscillation. None comes close to route completion.

## Preregistered criteria

| Criterion | Observed | Threshold | Pass |
|---|---:|---:|:---:|
| Pooled final Validation RSR | 0/27 | at least 14/27 | no |
| Minimum per-seed RSR | 0/9 | at least 3/9 | no |
| Seeds reaching at least 5/9 | 0 | at least 2 | no |
| Minimum training successes per seed | 0 | at least 1 | no |
| Maximum peak-to-final RSR drop | 0/9 | at most 3/9 | yes |

The stability criterion passes only because no policy ever rises above zero.
The earlier and stronger Gate D condition therefore applies: no training
success and no final Validation success.

## Frozen comparison

| Task specification | Final canonical Validation result |
|---|---:|
| Binary Success Reward V1 | **0/27** |
| Scalar SB3 SAC | 5/27 |
| Constrained RL V2 | 21/27 |

Requirement-Conditioned V1 reached 48/135 over a different five-requirement
matrix and is not directly pooled into the 27-episode table. The binary learner
has no successful sample within 900k pooled interactions; shaped Scalar SB3 SAC
first shows nonzero Validation feasibility at 250k per seed budget and finishes
at 5/27. Binary V1 is therefore less sample-efficient under the controlled SAC
comparison, while Constrained V2 remains the strongest canonical result.

## Interpretation

The task's success set is too narrow to be discovered by the frozen SAC setup
from unstructured continuous-action exploration. A fast trajectory provides no
information about how to remove its speed violation, and a speed-compliant
standstill provides no information about how to complete before the deadline.
Because every such outcome has exactly the same target value, the learner cannot
credit either partially correct behavior.

This is the intended scientific failure case for Binary V1. It supports the
broader task-specification question: external requirements can remain fixed
while the information exposed to the learner materially changes learnability.
No subsequent method or protocol modification is introduced by this study.

## Reproducibility artifacts

- `canonical.json`: immutable task, SAC settings, splits and acceptance gates;
- `results.json`: all derived summaries, learning curves, gates and frozen
  comparisons;
- `trajectories.json`: deterministic replays for objectively selected
  Validation tracks 3000 and 3001;
- `plots/`: learning, physical failure, optimizer and trajectory diagnostics;
- `runs/binary-reward-20261004/`: 36 raw evaluation results, final models,
  training outcomes and optimizer diagnostics.

The analysis validated all 36 expected result files and confirmed that no
reserved 4000--4017 seed occurs in results or trajectory artifacts.
