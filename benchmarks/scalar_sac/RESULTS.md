# Scalar SAC experiment at the 140-second requirement

## Result in one sentence

The fixed experiment completed all 24 training runs and 216 deterministic
evaluation episodes, but only 5 episodes (2.3%) were feasible. The dominant
result is not an energy ranking: it is a scalar-reward failure in which high
energy weight consistently favors near-stationary behavior, while weaker
energy weight produces seed-sensitive mixtures of late, speeding, and rarely
feasible trajectories.

The compact machine-readable result is
[`results-140s.json`](results-140s.json). Raw per-episode JSON, checkpoints,
trajectory samples, and generated plots remain in the ignored run directory
`runs/scalar-sac-140s-100k-20260924/`.

## 1. Training protocol

The canonical configuration was used without changing the scientific protocol:

- environment: `StochasticTrack-v1`, 1,800-step/180-second episode limit;
- task: complete 1,000 m in at most 140 s with exactly zero measured speed
  excess, then minimize signed net energy;
- 8 reward settings × training seeds 11/29/47;
- 10,000 random replay warm-up transitions plus 100,000 online SAC transitions
  per policy (2.64 million environment transitions over all policies);
- replay capacity 200,000, batch 256, MLP 64×64, learning rate 0.001,
  discount 0.99, target update factor 0.01;
- one-dimensional action in `[-1, 1]`;
- deterministic evaluation on the held-out track seeds 1000–1008.

Calibration seeds 2000–2008 and evaluation seeds 1000–1008 stayed disjoint.
Evaluation used physical `EpisodeMetrics`; training reward was never used as an
outcome measure. The semantic configuration hash is
`8b2e631873810b9a7b054b12e94060697b606eb34db98b9949b0a39937035d87`.

## 2. Completion status and reward sensitivity

All 24 policies trained to the configured 100,000 online steps and all 9 held-out
tracks were evaluated. RSR values below are `feasible tracks / 9`; the mean is
over the three training seeds.

| ID | `w_energy` | `w_speed` | `w_time` | seed 11 | seed 29 | seed 47 | mean RSR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e05-s0-t025 | 0.5 | 0 | 0.25 | 2/9 | 0/9 | 0/9 | 7.4% |
| e05-s0-t1 | 0.5 | 0 | 1 | 2/9 | 0/9 | 0/9 | 7.4% |
| e05-s2-t025 | 0.5 | 2 | 0.25 | 0/9 | 0/9 | 1/9 | 3.7% |
| e05-s2-t1 | 0.5 | 2 | 1 | 0/9 | 0/9 | 0/9 | 0% |
| e2-s0-t025 | 2 | 0 | 0.25 | 0/9 | 0/9 | 0/9 | 0% |
| e2-s0-t1 | 2 | 0 | 1 | 0/9 | 0/9 | 0/9 | 0% |
| e2-s2-t025 | 2 | 2 | 0.25 | 0/9 | 0/9 | 0/9 | 0% |
| e2-s2-t1 | 2 | 2 | 1 | 0/9 | 0/9 | 0/9 | 0% |

The training-seed range is 0–22.2 percentage points in the two best settings
and 0–11.1 points in `e05-s2-t025`; no configuration succeeds across seeds.

## 3. Completion, constraints, time, and feasible energy

Failure modes are exclusive combinations. `I`, `T`, and `S` denote incomplete,
over 140 s, and positive speed excess. Energy is shown only for feasible
episodes and as min/median/max.

| ID | completion | speed compliant | time min/median/max [s] | feasible energy [kWh] | F / I+T / I+T+S / T / T+S / S |
| --- | ---: | ---: | --- | --- | --- |
| e05-s0-t025 | 63.0% | 74.1% | 83.1 / 169.6 / 180.0 | 0.1688 / 0.1725 / 0.1762 (n=2) | 2 / 8 / 2 / 10 / 5 / 0 |
| e05-s0-t1 | 63.0% | 55.6% | 93.7 / 162.0 / 180.0 | 0.1625 / 0.1634 / 0.1642 (n=2) | 2 / 7 / 3 / 6 / 5 / 4 |
| e05-s2-t025 | 3.7% | 100% | 138.5 / 180.0 / 180.0 | 0.1563 (n=1) | 1 / 26 / 0 / 0 / 0 / 0 |
| e05-s2-t1 | 0% | 100% | 180.0 / 180.0 / 180.0 | — | 0 / 27 / 0 / 0 / 0 / 0 |
| every `w_energy=2` setting | 0% | 100% | 180.0 / 180.0 / 180.0 | — | 0 / 27 / 0 / 0 / 0 / 0 each |

Across all 216 episodes, the exclusive counts are 5 feasible, 176 `I+T`, 5
`I+T+S`, 16 `T`, 10 `T+S`, and 4 `S`. Exact compliance in the speed-penalized
half of the grid therefore does not imply useful behavior: it is mostly
obtained by not completing the route.

`w_speed=0` did not *reliably* cause speeding. It produced 19 violations among
108 episodes, all in the moving low-energy configurations; stationary
high-energy policies were trivially compliant. Conversely, `w_speed=2`
eliminated measured violations but also reduced completion to 1/108. Every
`w_energy=2` policy failed to complete every track, independent of the other
two weights.

## 4. Track-paired energy evidence

Feasibility was required on both sides of every energy comparison. Only one
within-SAC comparison survived this filter: on track 1007 and training seed 11,
`e05-s0-t025` consumed 0.17625 kWh and `e05-s0-t1` consumed 0.16252 kWh, a
difference of +0.01372 kWh for the former. There was no jointly feasible track
between different training seeds within any reward setting.

Against the two oracle references, the three successful SAC instances used less
energy on their very small jointly feasible subsets:

| SAC policy | paired tracks | vs fast oracle [mean kWh] | vs conservative oracle [mean kWh] |
| --- | ---: | ---: | ---: |
| e05-s0-t025, seed 11 | 2 | -0.08456 | -0.06031 |
| e05-s0-t1, seed 11 | 2 | -0.11300 | -0.08761 |
| e05-s2-t025, seed 47 | 1 | -0.12653 | -0.09822 |

These are selected, paired observations—not general energy rankings. With only
one or two jointly feasible tracks, infeasible observations cannot be treated
as missing at random and no significance claim is warranted.

## 5. Reference policies

The fast and conservative controllers read the complete future track and are
explicitly oracle-style physical anchors, not fair RL competitors.

| reference | privileged | RSR | completion | compliant | time mean/median [s] | feasible energy mean/median [kWh] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fast-compliant oracle | yes | 9/9 | 9/9 | 9/9 | 78.9 / 80.5 | 0.2372 / 0.2518 |
| conservative oracle | yes | 7/9 | 9/9 | 9/9 | 103.0 / 100.3 | 0.2294 / 0.2318 |
| random uniform actions | no | 0/9 | 0/9 | 8/9 | 180.0 / 180.0 | — |

The oracle results establish that all evaluation tracks are physically
completable and speed compliant within 140 s. The task itself is therefore not
the main explanation for SAC's failures.

## 6. Behavioral regimes and diagnostics

The data support distinct but mostly unsuccessful regimes:

- low energy weight without a speed penalty: policies often finish, but are
  usually too slow and sometimes speed; only seed 11 yields feasible episodes;
- low energy weight with a speed penalty: exact compliance, but predominantly
  safe/slow or stationary behavior;
- high energy weight: near-universal stationary collapse, yielding low measured
  energy and perfect speed compliance but zero completion;
- feasible and relatively energy-efficient behavior: observed on only five
  episodes, not robust across training seeds;
- feasible but energy-hungry learned behavior: not supported by this sample.

The automatically selected trajectory figure uses the highest-RSR seed of each
nonzero-RSR configuration (lowest seed breaks ties), the track feasible for the
largest number of those policies (track 1007), and the least-progress
high-energy zero-RSR policy as a failure control. It is reproducible with
`python -m benchmarks.scalar_sac.trajectories` and includes position, velocity,
speed limit, net energy, action, and acceleration.

The `e05-s0-t025` representative exhibits persistent small action and
acceleration oscillations after reaching cruise speed. Across all episodes that
configuration averages 257 acceleration-sign changes and action total
variation 9.70; `e2-s2-t1` has the highest aggregate action variation (23.95)
and mean absolute jerk (0.389 m/s³) despite completing nothing. This is
undesirable control behavior, but not evidence of a regenerative-energy exploit:
gross regeneration is small relative to traction energy in the moving learned
regimes. The clearer pathology is inactivity—some policies apply a steady
negative action at zero speed and accept roughly 0.04 kWh of idle consumption.

## 7. Why the scales create these outcomes

For a completed route, progress contributes approximately `+1`. At 140 s, the
time term is `-0.25` or `-1.0`; at the 180-second truncation it is `-0.321` or
`-1.286`. A plausible 0.20 kWh completed episode contributes `-0.4` for
`w_energy=0.5`, but `-1.6` for `w_energy=2`. One metre of integrated speed
excess contributes another `-2` when `w_speed=2`.

The observed stationary high-energy runs average approximately `+0.004`
progress, `-0.328` energy, and either `-0.321` or `-1.286` time. Under the
low-time-penalty/high-energy setting, a plausible completed episode around
0.20 kWh and 100 s scores about `+1 - 1.6 - 0.179 = -0.779`, while stationary
behavior scores about `-0.645`. The reward can therefore prefer inactivity
even before discounting delayed progress. A large speed penalty strengthens
the attraction of behavior that never approaches a constraint.

This arithmetic and the consistency across all 12 high-energy runs are stronger
evidence of pathological scalarization than training noise alone. The low-energy
settings still show substantial seed instability, so 100,000 online steps may
also be insufficient for reliable SAC convergence.

## 8. Plots and statistical limits

The stored-result plotting commands generate:

- reward configuration vs RSR with all seed points and min/max seed bars;
- feasible energy vs RSR, so low energy cannot be read without feasibility;
- exclusive failure-mode proportions;
- representative deterministic trajectories from persisted trajectory JSON.

There are only three training seeds and nine reused evaluation tracks. The
analysis therefore reports raw seed variation, medians, ranges, and paired
track differences without an independence-based hypothesis test or inflated
confidence claim.

## 9. Decision gate and next experiment

The primary classification is **C: almost all reward settings fail**, with
category-D seed instability as a secondary finding. The references reject
"task impossible" as the main cause; reward scaling is demonstrably
pathological, and the training budget/convergence remains a plausible additional
cause. Observation sufficiency is not disproved, but is less likely because the
same observation supports occasional feasible learned behavior.

Do not weaken the 140-second canonical task in response to failed reward
engineering. A post-hoc rescore of these same policies—not a new training
sweep—would yield 3/216, 5/216, and 16/216 feasible episodes at 120, 140, and
160 s. The fast oracle is 9/9 at all three budgets; the conservative oracle is
7/9, 7/9, and 8/9. The planned **120/140/160 s** range remains defensible, but
the full sweep should wait.

The next scalar stage should be a small convergence and scaling check: extend a
few representative runs beyond 100,000 steps and define a scalar control whose
completed-route energy penalty cannot be dominated by stationary idle behavior.
Freeze that stronger scalar baseline before comparing it with explicit
constraints or requirement-conditioned control.

## 10. Answer to the paper question

**Yes, the benchmark exposes a real reward-engineering problem strongly enough
to motivate later non-scalar comparisons.** Small interpretable weight changes
produce qualitatively different failure modes, and one apparently reasonable
energy weight makes inactivity preferable to satisfying the task. However,
these runs are not yet a fair performance contest between RL paradigms. A short
convergence/scaling control is needed first, after which the benchmark should be
kept fixed for the later comparison.
