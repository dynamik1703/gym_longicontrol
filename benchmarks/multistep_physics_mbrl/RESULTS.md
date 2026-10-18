# Multi-Step Physics MVE results

## Outcome

The single authorized Validation opening is complete. The six fixed 300k policies were
each evaluated once on tracks 3000--3008, for exactly **54 episodes**. Evaluation used
only the ordinary eight-dimensional observation, the frozen deterministic learned
actor, and its action. No physics model was queried, no lookahead or planning was used,
and no policy was trained, adapted, selected, or modified.

The primary Validation comparison is **D -- degradation**. Matched real-only achieves
10/27 pooled RSR; Physics MVE achieves 0/27, a difference of -10/27. MVE also has lower
completion (9/27 versus 16/27), deadline compliance (9/27 versus 10/27), and speed
compliance (14/27 versus 27/27).

This does not alter the preregistered Development result. Development remains
**E -- unstable**, with **D -- degradation** as its efficacy comparison without the
instability override. The overall study classification remains **E -- unstable**
because Physics MVE seed 47 crossed the frozen numerical-stability threshold during
Development.

## Validation results

| Condition / seed | RSR | Completion | Deadline | Speed | Exclusive outcomes |
|---|---:|---:|---:|---:|---|
| real-only 11 | 0/9 | 2/9 | 0/9 | 9/9 | 2 completed too slowly; 7 partial/crawling |
| real-only 29 | 4/9 | 5/9 | 4/9 | 9/9 | 4 fully feasible; 1 completed too slowly; 4 partial/crawling |
| real-only 47 | 6/9 | 9/9 | 6/9 | 9/9 | 6 fully feasible; 3 completed too slowly |
| **real-only pooled** | **10/27** | **16/27** | **10/27** | **27/27** | 10 fully feasible; 6 completed too slowly; 11 partial/crawling |
| MVE 11 | 0/9 | 0/9 | 0/9 | 5/9 | 9 partial/crawling |
| MVE 29 | 0/9 | 9/9 | 9/9 | 0/9 | 9 completed with speed violation |
| MVE 47 | 0/9 | 0/9 | 0/9 | 9/9 | 9 standstill |
| **MVE pooled** | **0/27** | **9/27** | **9/27** | **14/27** | 9 partial/crawling; 9 speed-violating completions; 9 standstills |

The requested seed-level RSR statement is therefore:

```text
Real-only:
seed 11 0/9
seed 29 4/9
seed 47 6/9
pooled 10/27

Physics MVE:
seed 11 0/9
seed 29 0/9
seed 47 0/9
pooled 0/27
```

## Physical metrics and feasible-only energy

| Condition / seed | Mean travel time (s) | Mean / max maximum speed excess (m/s) | Mean / total integrated speed excess (m) | Feasible energy mean / median (kWh; n) |
|---|---:|---:|---:|---:|
| real-only 11 | 173.600 | 0 / 0 | 0 / 0 | n/a; n=0 |
| real-only 29 | 138.278 | 0 / 0 | 0 / 0 | 0.172582 / 0.172249; n=4 |
| real-only 47 | 126.322 | 0 / 0 | 0 / 0 | 0.169169 / 0.167932; n=6 |
| **real-only pooled** | **146.067** | **0 / 0** | **0 / 0** | **0.170534 / 0.168150; n=10** |
| MVE 11 | 180.000 | 3.446850 / 10.539035 | 6.347653 / 57.128877 | n/a; n=0 |
| MVE 29 | 34.700 | 26.893216 / 31.444444 | 499.676156 / 4,497.085402 | n/a; n=0 |
| MVE 47 | 180.000 | 0 / 0 | 0 / 0 | n/a; n=0 |
| **MVE pooled** | **131.567** | **10.113355 / 31.444444** | **168.674603 / 4,554.214279** | **n/a; n=0** |

Energy is reported as an optimization outcome only for feasible episodes. MVE has no
feasible Validation episode, so it has no feasible-only energy statistic and no energy
comparison is made from its infeasible episodes. `results.json` retains complete compact
metrics for every episode, including travel time, maximum and integrated speed
violation, energy, completion, feasibility, final position, and the exclusive outcome.

## Frozen interpretation

The frozen A--D rules resolve as follows:

- A fails: MVE is below 24/27 RSR and does not have 27/27 completion or deadline
  compliance; it also does not exceed real-only.
- B fails: MVE does not exceed real-only by at least 3/27.
- C fails: the absolute RSR difference is 10/27, not less than 3/27.
- D passes: MVE is 10/27 below real-only and also worsens completion and deadline.

The final Validation performance classification is therefore **D -- degradation**.
The overall classification remains **E -- unstable** under the already-triggered
Development instability override.

## Development curves and Validation consistency

Counts below are out of nine Development tracks per seed, at 50k, 100k, 150k, 200k,
250k, and 300k real transitions. These frozen Development results are unchanged and
were not reinterpreted after Validation.

| Policy | Development RSR | Completion | Deadline | Speed |
|---|---|---|---|---|
| real-only 11 | 0, 3, 2, 3, 0, 2 | 0, 9, 9, 9, 9, 4 | 0, 6, 6, 6, 9, 2 | 9, 4, 3, 4, 0, 9 |
| real-only 29 | 0, 0, 5, 6, 4, 5 | 0, 9, 9, 9, 5, 5 | 0, 9, 6, 9, 4, 5 | 9, 0, 7, 6, 9, 9 |
| real-only 47 | 0, 0, 4, 7, 3, 6 | 0, 9, 9, 9, 9, 7 | 0, 9, 8, 7, 8, 6 | 9, 0, 4, 9, 4, 8 |
| MVE 11 | 0, 0, 1, 0, 4, 0 | 0, 1, 3, 9, 6, 0 | 0, 0, 1, 9, 6, 0 | 9, 7, 7, 0, 5, 6 |
| MVE 29 | 0, 0, 0, 0, 0, 0 | 0, 0, 1, 0, 0, 9 | 0, 0, 0, 0, 0, 9 | 9, 9, 9, 9, 9, 0 |
| MVE 47 | 0, 0, 0, 0, 0, 0 | 0, 0, 0, 0, 0, 0 | 0, 0, 0, 0, 0, 0 | 9, 9, 9, 9, 9, 9 |
| pooled real-only | 0, 3, 11, 16, 7, 13 | 0, 27, 27, 27, 23, 16 | 0, 24, 20, 22, 21, 13 | 27, 4, 14, 19, 13, 26 |
| pooled MVE | 0, 0, 1, 0, 4, 0 | 0, 1, 4, 9, 6, 9 | 0, 0, 1, 9, 6, 9 | 27, 25, 25, 18, 23, 15 |

The final failure regimes generalize descriptively:

- Real-only seed 11 remains speed-compliant but dominated by partial/crawling and late
  completion; Validation RSR falls from 2/9 to 0/9.
- Real-only seed 29 retains 5/9 completion and 9/9 speed compliance; RSR changes from
  5/9 to 4/9 because one completion is late.
- Real-only seed 47 retains 6/9 RSR and 6/9 deadline compliance, while completion and
  speed improve to 9/9; the remaining three episodes complete too slowly.
- MVE seed 11's partial-progress/crawling regime generalizes: 0/9 RSR and 0/9
  completion in both splits.
- MVE seed 29's regime generalizes exactly: nine on-time completions with nine speed
  violations in both splits.
- MVE seed 47's regime generalizes exactly: nine standstills, 0/9 completion, and 9/9
  speed compliance in both splits.

This comparison is descriptive only and was not used for checkpoint or policy
selection.

## Numerical stability and target decomposition

Physics MVE seed 47 first breached the frozen critic-loss threshold at 200k real
transitions (5.657e6 versus the 1e6 threshold). At 300k its critic loss was
2.243e9, target magnitude 4.272e6, and alpha 7,056.36. Values remained finite;
the final speed and deadline multipliers were 1.8749 and 3.6892, and no multiplier
reached 1,000.

The final count-weighted target decomposition is shown as `target = real two-step +
model expansion + terminal/bootstrap`:

| Seed / critic | Target | Real two-step | Model expansion | Terminal/bootstrap |
|---|---:|---:|---:|---:|
| MVE 11 / objective | -1.9235 | -0.0018 | -0.9508 | -0.9709 |
| MVE 11 / speed | -1.1530 | 0.0514 | -0.2963 | -0.9082 |
| MVE 11 / deadline | -0.8109 | 0.0243 | -0.3850 | -0.4502 |
| MVE 29 / objective | -157.5431 | -0.0012 | -66.6299 | -90.9120 |
| MVE 29 / speed | -138.6998 | 0.1339 | -62.9347 | -75.8990 |
| MVE 29 / deadline | -155.9755 | 0.0347 | -65.7801 | -90.2301 |
| MVE 47 / objective | -2,030,932.2586 | -0.0002 | -927,564.4380 | -1,103,367.8205 |
| MVE 47 / speed | -2,030,461.6538 | 0.0000 | -927,564.4341 | -1,102,897.2197 |
| MVE 47 / deadline | -2,031,547.8630 | 0.0495 | -927,563.3306 | -1,103,984.5819 |

For seed 47, the model-expansion and terminal/bootstrap terms dominate while the real
two-step contribution remains near zero. Together with the loss, target, and alpha
growth, this is evidence consistent with long-horizon target instability. It is not a
new causal claim.

## Runtime, accounting, and integrity

Validation took 40.855 wall-clock seconds and 74,961 simulator transitions. It made
zero training updates. Main training remains exactly 1,800,000 real transitions and
180,000 RL updates across six policies. Every final policy still has exactly 300,000
real transitions and 30,000 updates; the 300k checkpoint is the fixed endpoint, with
no best-checkpoint selection.

| Policy | Frozen final checkpoint SHA-256 |
|---|---|
| real-only 11 | `994568f614c9d2428b805fafafc9526717fa2d6e4ca73783adfcb361feb4090d` |
| real-only 29 | `f32757afc979166793ba2f3725df7511af728dd446de1b2ded4d482a3034f5e8` |
| real-only 47 | `e0262a618f0435f85b792286cdbd9aebb3a6958e01a88ae101724ed8c98cade6` |
| MVE 11 | `8125ed0cc385e0eead5435e82f9d35b2a69607b1ce54aad8f7b30cc9ea566d9b` |
| MVE 29 | `66ca2345b311ad19f7455bf742316866a00cfdbd52f60bedfdf021fe77b767dd` |
| MVE 47 | `e1c02028c368111e2807ad75733112c0502d5118aa545795c96ba37ac21aa88e` |

All six checkpoint hashes still match the bytes and run manifest. Actor-state hashes
were identical immediately before and after every Validation pass. Training counters
did not change. The 36 Development artifacts retain their pre-Validation aggregate
SHA-256 `c8590f0312a1909ad59b375bac7760b5643aca6f689471596c2c6a8e4ffcb321`.
Frozen hashes pass after the narrowly scoped authorization transition. The opening is
recorded once at `2026-10-18T18:31:42.091015+00:00`; each of the six policies has one
result containing exactly the nine unique tracks.

The completed main-training commit is
`03d84c6677854406b52e84c52a6d3cec11c9652a`; the direct Validation-authorization
descendant is `67660cba9a3f314241f039c71f49480551795832`. Full per-episode metrics,
Development curves, actor hashes, raw-result hashes, diagnostics, target decomposition,
runtime, and provenance are retained in `results.json`.

**Paper final remains SEALED.** Its `opened_at` remains null and its episode count is
zero.
