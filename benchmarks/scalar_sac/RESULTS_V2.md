# Scalar SAC baseline hardening: V2 results

## Executive result

Reasonable scalar reward engineering improved the best mean exploratory RSR
from 7.4% in V1 to 22.2% in V2-B, but it did **not** establish a strong or fair
Scalar-SAC baseline. No candidate met the acceptance criterion, at least one
training seed of every candidate remained at zero RSR, and validation learning
curves were non-monotonic or collapsed after transient improvements.

The supported decision gate is **C: training instability remains dominant**.
Scalar SAC should not yet be used as the supposedly competent scalar comparator
in a Rewards-vs-Requirements paper.

Compact data are in [`results-v2.json`](results-v2.json), while
[`reward-development.json`](reward-development.json) records the factual tuning
path. Large checkpoints, raw per-episode results, trajectory samples, and plots
remain under ignored `runs/scalar-sac-v2*-20260924/` directories.

## 1. Why V1 failed

Scalar Study V1 is preserved unchanged in `canonical.json`, `RESULTS.md`, and
`results-140s.json`. Its progress term contributed about +1 for a complete
route, while a plausible 0.20 kWh trip with `w_energy=2` contributed about
-1.6. With the low time weight, representative completion could return -0.779
while standstill returned about -0.645. All 12 high-energy runs consequently
failed to complete any evaluation track. V1 produced only 5 feasible episodes
out of 216 and exposed a genuine reward-scale inversion.

## 2. Analytical redesign

The V2 formula retains four conceptual components:

```text
route achievement = w_progress * delta_position / 1000 m
                  + B_on_time on an on-time completion
energy            = -w_energy * step_energy / 0.25 kWh
time              = -w_time * delta_time / 140 s
speed violation   = -w_speed * speed_excess * delta_time / 1 m
                    -P_event once at episode end if any excess occurred
```

The terminal event penalty expresses the exact no-violation preference without
changing benchmark feasibility. Evaluation continues to use only
`EpisodeMetrics` and `is_feasible`.

For a 180 s/0.0404 kWh standstill, a 115 s/0.23 kWh feasible drive, a 115
s/0.30 kWh feasible inefficient drive, a 90 s/0.28 kWh drive with 5 m
integrated violation, and a 160 s/0.23 kWh late completion, the frozen return
estimates were:

| candidate | standstill | feasible efficient | feasible inefficient | fast violating | slow completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2-A balanced | -0.402 | 2.335 | 2.195 | -4.721 | 0.254 |
| V2-A energy | -0.483 | 2.875 | 2.595 | -4.281 | -0.206 |
| V2-A compliance | -0.402 | 2.335 | 2.195 | -10.721 | 0.254 |
| V2-B balanced | -0.402 | 3.335 | 3.195 | 0.779 | 1.254 |
| V2-B energy | -0.483 | 3.875 | 3.595 | 1.219 | 0.794 |
| V2-B compliance | -0.402 | 3.335 | 3.195 | -0.971 | 1.254 |

Every serious candidate satisfied `feasible completion > standstill` before
training. Detailed preregistered reasoning is in
[`V2_PROTOCOL.md`](V2_PROTOCOL.md) and [`V2B_PROTOCOL.md`](V2B_PROTOCOL.md).

## 3. Reward iterations and engineering effort

V2-A used progress weight 1, energy weights 0.5/1, time weight 0.25,
completion bonuses 2/3, integrated speed weights 1/2, and event penalties 2/3.
It fixed the episode return ordering but remained locally difficult to learn:
all candidates had 0/27 validation RSR at 300k.

V2-B made one documented scale repair without adding a shaping concept:
progress increased to 2 and integrated speed weights fell to 0.1/0.25. Event,
energy, time, and completion semantics stayed intact. Across V1, V2-A, and
V2-B, 8 + 3 + 3 reward configurations were evaluated; V2 itself contains six
meaningful candidates in two traceable iterations rather than a hidden grid
search.

## 4. Predefined acceptance criterion

Before V2 evaluation, a configuration was declared credible only if at 300k:

1. every seed achieved at least 8/9 RSR;
2. every seed completed at least 8/9 tracks;
3. every seed was speed compliant on at least 8/9 tracks;
4. RSR range across seeds was at most 1/9;
5. each seed's paired mean energy was no more than 1.25 times the fast oracle.

No V2-A or V2-B configuration passed. The thresholds were not changed after
seeing results.

## 5. Training-budget experiment

Each V2 iteration trained three candidates × seeds 11/29/47. Every policy used
10k random warm-up transitions and a continuous 300k online training trajectory,
with validation every 50k and stored comparison points at 100k and 300k. SAC
retained the V1 MLP 64×64, batch 256, learning rate 0.001, gamma 0.99, and tau
0.01. Replay capacity increased from 200k to 400k so the longer run was not
clipped by capacity.

V2-B balanced validation mean RSR rose from 1/27 at 50k to 10/27 at 250k, then
fell to 2/27 at 300k. Its seed values at 250k were 0/9, 5/9, and 5/9. The
energy-oriented mean briefly reached 3/27 at 100k and returned to zero by 200k.
V2-A showed the same pattern on a smaller scale: one balanced seed peaked at
4/9 at 150k and collapsed to zero at 300k.

These curves reject the simple explanation that 100k was merely too short.
Because improvement was non-monotonic and multiple seeds were flat, an automatic
1M extension was not run.

## 6. RSR by configuration and seed

These are deterministic results on the already-exposed exploratory tracks
1000–1008. Values are feasible tracks out of nine.

| iteration/configuration | budget | seed 11 | seed 29 | seed 47 | mean RSR |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2-A balanced | 100k | 0 | 0 | 1 | 3.7% |
| V2-A energy | 100k | 0 | 0 | 0 | 0% |
| V2-A compliance | 100k | 0 | 0 | 0 | 0% |
| V2-A balanced | 300k | 0 | 0 | 0 | 0% |
| V2-A energy | 300k | 0 | 0 | 0 | 0% |
| V2-A compliance | 300k | 0 | 1 | 0 | 3.7% |
| V2-B balanced | 100k | 0 | 0 | 5 | 18.5% |
| V2-B energy | 100k | 0 | 3 | 0 | 11.1% |
| V2-B compliance | 100k | 0 | 0 | 0 | 0% |
| V2-B balanced | 300k | 0 | 4 | 2 | 22.2% |
| V2-B energy | 300k | 0 | 0 | 0 | 0% |
| V2-B compliance | 300k | 0 | 0 | 2 | 7.4% |

The best final configuration has a 44.4 percentage-point RSR range across
training seeds, four times the accepted maximum.

## 7. Completion and failure modes at 300k

| V2-B configuration | completion | speed compliance | feasible | exclusive failures |
| --- | ---: | ---: | ---: | --- |
| balanced | 44.4% | 77.8% | 6/27 | 13 incomplete+time; 2 incomplete+time+speed; 4 speed; 2 time |
| energy-oriented | 0% | 92.6% | 0/27 | 25 incomplete+time; 2 incomplete+time+speed |
| strong-compliance | 7.4% | 100% | 2/27 | 25 incomplete+time |

Balanced creates useful movement but trades completion against speed. Strong
compliance mostly obtains compliance by crawling or standing. Energy-oriented
returns to near-total inactivity despite its analytically favorable completed
return.

## 8. Travel time and feasible energy

At 300k, balanced travel time was 147.4 s mean and 180.0 s median; its six
feasible episodes used 0.18135 kWh mean and 0.17955 kWh median. Strong-compliance
had 172.7 s mean and 180.0 s median; its two feasible episodes used 0.21635 kWh.
Energy-oriented had no feasible energy observation.

The aggregate balanced number must not be interpreted as 27 independent trained
agents. By seed, balanced feasibility was 0, 4, and 2 tracks. Seed 29's four
feasible episodes averaged 0.19154 kWh; seed 47's two averaged 0.16098 kWh.

## 9. Track-paired oracle comparison

Energy was compared only where both the learned policy and fast oracle were
feasible:

| policy | paired tracks | mean delta learned−oracle [kWh] | mean ratio |
| --- | ---: | ---: | ---: |
| V2-B balanced, seed 29 | 4 | -0.05375 | 0.825 |
| V2-B balanced, seed 47 | 2 | -0.09609 | 0.634 |
| V2-B compliance, seed 47 | 2 | -0.04071 | 0.851 |

The accepted energy threshold is met on these subsets, but feasibility coverage
is far too low to rank methods by energy. Low-RSR policies are not credited for
low energy on selectively easy tracks.

## 10. Trajectory diagnostics

The deterministic selection rule chooses the best policy at 100k and 300k, the
remaining highest-completion 100k policy, the failed seed of the best 300k
configuration, and the best strong-compliance 300k policy. Track 1005 maximizes
common feasibility; track 1006 maximizes distinct failure modes.

On track 1005, four moving policies finish feasibly between 72.0 and 108.3 s,
while `v2b-balanced/seed11/300k` remains at position zero for 180 s. The moving
policies show repeated acceleration/braking and occasional jerk spikes above
30 m/s³. On track 1006:

- the best 100k balanced policy finishes at 147.2 s and is too slow;
- the high-completion energy policy finishes at 114.8 s but exceeds the limit
  by 5.27 m/s;
- the best 300k balanced policy stops near 601 m and violates speed earlier;
- the strong-compliance policy crawls to about 104 m and stops;
- the failed balanced seed never moves.

Mean gross regeneration at 300k is 0.0159 kWh for balanced versus 0.1374 kWh
traction. There is no clear regenerative-energy exploit. The important
undesirable behaviors are standstill, mid-route stopping, limit violation,
oscillatory actions, and aggressive braking/acceleration.

## 11. V1 to V2 improvement

Best mean exploratory RSR increases from 7.4% (`e05-s0-t025`) to 22.2%
(`v2b-balanced`), an absolute +14.8 percentage points and a threefold descriptive
increase. The number of feasible episodes in the selected aggregate rises from
2/27 to 6/27.

This improvement required explicit episode-return analysis, a terminal task
signal, exact-violation event semantics, a 3× longer budget, validation learning
curves, and a second scale repair. It did not yield reliability: the best V1
seed range was 22.2 points, while the best V2-B range is 44.4 points. Aggregate
completion also falls from 63.0% for the best V1 configurations to 44.4% for
V2-B balanced.

The supported statement is therefore: reward redesign changed and sometimes
improved behavior, but did not restore a reliable scalar baseline.

## 12. Remaining seed instability

Seed instability is not a small uncertainty band around a shared policy quality.
At identical hyperparameters, one seed can remain stationary, another can finish
most tracks while speeding, and another can be feasible on a few tracks.
Performance can also improve through 250k and then collapse by 300k. Selecting
the best checkpoint per seed would make results look better, but would not meet
the predefined all-seed reliability criterion and would add a validation-driven
selection degree of freedom.

## 13. Time budget and data splits

There is still no competent learned policy with which to judge whether 140 s is
too easy. The fast privileged oracle remains 9/9 and the conservative oracle
7/9 at 140 s, so the canonical requirement remains physically meaningful. Do
not run the 120/140/160 sweep yet; retain that candidate range until training
stability is resolved.

Tracks 1000–1008 were already inspected in V1 and are explicitly labeled an
exploratory development benchmark. V2 learning curves use new validation seeds
3000–3008. Final paper seeds 4000–4017 are present only as reserved identifiers:
they were not generated, evaluated, plotted, or inspected.

## 14. Decision and next research stage

The outcome is **C: training instability remains dominant**. Observation
insufficiency is not the leading diagnosis because some seeds produce competent
segments and occasional feasible trajectories from the unchanged observation.
The task is not physically impossible, as proven by the references. Reward
scaling matters, but does not explain the severe within-configuration and
over-time collapse by itself.

The next stage should investigate the scalar SAC training process—Q/entropy
scales, target and policy dynamics, normalization, checkpoint stability, and
modern SAC implementation parity—before any MORL, constrained-RL, or
requirement-conditioned comparison. No additional paradigm or 1M run is added
here.

## Answer to the paper question

**No. After reasonable and transparently documented reward engineering, this
Scalar SAC implementation cannot yet serve as a strong and fair baseline.** V2
demonstrates that scalar design can improve behavior, but the result remains
seed-dependent, non-monotonic, and far below the preregistered reliability
threshold. A future comparison should use a stabilized scalar learner rather
than either the pathological V1 reward or the best cherry-picked V2 checkpoint.
