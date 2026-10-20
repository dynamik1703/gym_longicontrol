# Mechanism synthesis

## Conclusion

The sign-changing collector effect is not cleanly attributable to one retained
mechanism. The collector schedule creates two large, directly observed contrasts:
EPISODIC performs zero actor updates while an episode is active and applies
essentially its entire update budget in post-PID episode-end blocks; TRANSITIONAL
updates throughout the episode and almost never updates in the terminal scheduler
call. Those contrasts are present in all three seeds and therefore do not themselves
change sign with RSR.

What changes by seed is the coupled trajectory. Lambda paths, constraint regimes,
alpha, episode length, and critic losses co-evolve differently. The only compact
internal feature whose E−T sign and ordering match `+3, -4, -7` is final deadline
lambda; it is an endogenous consequence of accumulated episode costs, not an
independent causal lever. The evidence supports trajectory-dependent feedback, not
a uniquely isolated mechanism.

## H1 — PID/Lagrange timing

`post_episode_lambda_update_exposure` means gradient updates executed in the same
scheduler call after the just-completed episode's PID event. It does not claim that
the corresponding episode was sampled. The large-jump threshold is the pooled 95th
percentile of all absolute lambda jumps, `13.792733`, chosen without
performance labels.

| Condition | PID events | mean/median/max abs(Δλ speed) | mean/median/max abs(Δλ deadline) | large S/D jumps | immediate updates | fraction |
|:--|---:|---:|---:|---:|---:|---:|
| E11_episodic_seed11 | 243 | 5.363/0.072/66.488 | 2.278/0.715/18.192 | 24/6 | 30,000 | 100.0000% |
| T11_transitional_seed11 | 233 | 4.083/0.153/57.246 | 2.031/0.557/18.029 | 20/2 | 35 | 0.1167% |
| E29_episodic_seed29 | 236 | 5.197/0.045/85.446 | 2.635/1.267/16.600 | 23/2 | 30,000 | 100.0000% |
| T29_transitional_seed29 | 238 | 3.217/0.036/46.414 | 2.254/0.801/14.490 | 18/1 | 57 | 0.1900% |
| E47_episodic_seed47 | 236 | 4.362/0.051/67.737 | 2.211/0.752/15.096 | 22/2 | 30,000 | 100.0000% |
| T47_transitional_seed47 | 206 | 3.635/0.020/49.038 | 2.263/1.177/15.461 | 18/2 | 96 | 0.3200% |

EPISODIC immediate exposure is 100% in every seed. TRANSITIONAL exposure is only
0.1167%, 0.1900%, and 0.3200% for seeds 11/29/47. EPISODIC speed-lambda volatility
is also higher in every pair. These facts strongly establish amplification timing,
but their invariant direction cannot explain why only seed 11 benefits.

## H2/H4 — policy nonstationarity and task exposure

| Condition | Within-episode actor updates mean/max | hash-changed episodes | episodes | mean/median length | unique routes/tasks | duplicate rate R/T |
|:--|---:|---:|---:|---:|---:|---:|
| E11_episodic_seed11 | 0.000/0 | 0 | 243 | 1234.6/1233.0 | 240/240 | 1.23%/1.23% |
| T11_transitional_seed11 | 128.605/180 | 233 | 233 | 1287.6/1263.0 | 231/231 | 0.86%/0.86% |
| E29_episodic_seed29 | 0.000/0 | 0 | 236 | 1271.2/1259.5 | 231/231 | 2.12%/2.12% |
| T29_transitional_seed29 | 125.811/180 | 238 | 238 | 1260.5/1207.5 | 233/233 | 2.10%/2.10% |
| E47_episodic_seed47 | 0.000/0 | 0 | 236 | 1271.2/1232.0 | 234/234 | 0.85%/0.85% |
| T47_transitional_seed47 | 145.165/180 | 206 | 206 | 1456.3/1800.0 | 204/204 | 0.97%/0.97% |

Hashes prove every EPISODIC actor is fixed from episode start through terminal;
every TRANSITIONAL episode with active updates changes actor hash. Per-episode actor
tensors were not retained, so no nonexistent per-episode L2 norm is reconstructed.
Scientific checkpoints do permit 50k actor L2 displacement; those values and hashes
are in `results.json`.

Task exposure is not a sufficient account: E11 has +10 episodes, E29 has -2, and
E47 has +30, while RSR differences are +3, -4, and -7. Seed 47 is the direct
counterexample to a simple more-tasks or shorter-episodes benefit.

## H3 — replay/update freshness

Schedule-level freshness is directly recoverable. EPISODIC update events occur at
episode terminals after a complete trajectory enters replay; TRANSITIONAL events
usually occur every ten new transitions while only a prefix of the active episode is
available. Replay occupancy and insertion gaps are retained and reported.

Actual sample ages and fractions from the last 256/1,000/5,000 transitions are
`NOT_RECOVERABLE`: sampled replay indices were not logged. The audit does not replace
them with expected uniform-sampling fractions.

## H5 — behavioral regimes

Each diagnostic episode is classified by frozen raw metrics. A checkpoint receives
the unique strict-majority category (>9/18), otherwise `MIXED`. The operational
order and all counts are in `results.json`.

- Seed 11: E ends `FEASIBLE`-majority (12/18 RSR) while T ends `MIXED` (9/18).
- Seed 29: both end `MIXED`; Validation reveals E's higher completion but worse
  deadline/speed tradeoff.
- Seed 47: E moves from crawling to speed violation to feasible at 250k, then falls
  back to crawling/partial at 300k. T recovers after 150k and ends mixed at 9/18.

All three pairs already differ on raw completion/deadline/speed counts at 50k. The
later direction is not monotone: seed 47 is the clearest late reversal.

## H6/H7 — alpha and optimization

| Condition | alpha 50k→300k | log-alpha change | actor-total 50k→300k | q-total 50k→300k |
|:--|---:|---:|---:|---:|
| E11_episodic_seed11 | 0.234607→0.000562 | -6.035 | 8.700→1.910 | 1.056→0.440 |
| T11_transitional_seed11 | 0.236332→0.000677 | -5.855 | 8.769→1.886 | 0.976→0.397 |
| E29_episodic_seed29 | 0.236364→0.000583 | -6.005 | 9.984→2.047 | 1.396→0.535 |
| T29_transitional_seed29 | 0.223319→0.001187 | -5.237 | 7.760→1.969 | 0.057→0.078 |
| E47_episodic_seed47 | 0.223483→0.000503 | -6.097 | 9.591→2.234 | 0.065→0.212 |
| T47_transitional_seed47 | 0.223442→0.004396 | -3.928 | 9.077→2.523 | 0.060→0.067 |

Final E/T alpha ratios are approximately 0.86, 0.50, and 0.11 for seeds 11, 29,
and 47. This magnitude ordering is consistent with worsening EPISODIC outcomes,
especially the T47 alpha rebound, but the paired alpha difference is negative in all
three seeds and is endogenous. Checkpoint actor-total and Q-total losses are
descriptive snapshots; no per-update trace or target statistics remain.

## Cross-seed sign reversal

No exogenous candidate feature reproduces the RSR sign reversal. Immediate PID
exposure, within-episode actor updating, replay scheduling, and EPISODIC's generally
higher speed-lambda volatility all keep the same direction. Final deadline lambda
does match the signs (`positive, negative, negative`), but it summarizes the very
cost trajectory being explained. With n=3 this is descriptive concordance only.

## Ranking

1. **coupled PID / constraint-regime trajectory feedback — PLAUSIBLE**. PID timing and immediate update amplification are direct, and seed-specific lambda/regime paths accompany outcomes; no exogenous PID feature uniquely reproduces the sign reversal.
2. **alpha / exploration trajectory — PLAUSIBLE**. Final log-alpha separation orders with outcome deterioration and T47 shows an entropy rebound, but alpha is jointly endogenous.
3. **critic / actor optimization dynamics — WEAK**. Some loss scale separation appears in seeds 29/47, but retained snapshots show no uniquely timed collapse and omit targets.
4. **task-exposure feedback — WEAK**. Seed 47 has 30 more EPISODIC completed tasks yet the worst EPISODIC outcome, defeating a simple exposure account.
5. **within-episode actor nonstationarity alone — WEAK**. The treatment contrast is exact and large but essentially fixed across seeds, so it cannot alone identify the sign reversal.
6. **actual replay-sample freshness — NOT_TESTABLE_FROM_RETAINED_DATA**. Scheduling and occupancy are retained; sampled indices and ages are not.

## Evidence not recoverable

- `NOT_RECOVERABLE` — sampled replay indices and transition IDs: No empirical sample-age or recent-window fractions.
- `NOT_RECOVERABLE` — per-episode actor parameter tensors: No per-episode actor L2 drift; hashes and update counts only.
- `NOT_RECOVERABLE` — per-gradient-update loss history: Abrupt sub-checkpoint optimization transients cannot be located.
- `NOT_RECOVERABLE` — critic target and Q-distribution statistics: Checkpoint scalar losses cannot diagnose calibration/collapse.
- `NOT_RECOVERABLE` — diagnostic behavior at transition zero: Zero row contains learner initialization only; no episode inferred.
- `PARTIALLY_OBSERVED` — append-only Validation invocation log: Single manifest open/completion record plus six sealed files and recovery code establish consistency, but not an external audit log.
