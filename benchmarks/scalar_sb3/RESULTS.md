# Stable-Baselines3 scalar baseline study

## Executive result

Stable-Baselines3 removes neither the low success rate nor the strong
seed-dependence of the scalar benchmark. At 300,000 interactions, SB3 SAC
achieved 5 feasible validation episodes out of 27 (18.5% mean RSR), split
4/9, 0/9, and 1/9 across seeds. SB3 PPO achieved 0/27: all final deterministic
policies selected full braking and remained at position zero.

Neither algorithm satisfies the frozen strong-baseline criterion. The supported
decision is **D: the difficulty lies beyond the historical SAC implementation**.
The historical implementation is a confounder—its validation curves collapse
late—but it is not the dominant explanation. Modern SAC improves the final
validation RSR and avoids late collapse in this run, yet remains far from a
credible scalar comparator; PPO shows that the failure is not SAC-specific.

Machine-readable results are in [`results.json`](results.json). The canonical
raw directory is `runs/scalar-sb3-20260925-diagnostics/`. Models, checkpoint
evaluations, diagnostics, and trajectory samples remain ignored. Small plots in
`plots/` are intentionally versioned.

## 1. Frozen common protocol

Both methods used Stable-Baselines3 2.9.0 and exactly the Scalar V2-B Balanced
reward:

```text
route = 2.0 * delta_position / 1000 m
      + 2.0 on an on-time completion
energy = -0.5 * step_energy / 0.25 kWh
time = -0.25 * delta_time / 140 s
speed = -0.1 * speed_excess * delta_time / 1 m
      - 2.0 once at episode end if any speed excess occurred
```

The task remained independent: complete 1,000 m within 140 s, never exceed the
speed limit, then compare energy only among feasible episodes. Evaluation used
`EpisodeMetrics` and `is_feasible`, never training return.

Common settings:

- training seeds 11, 29, and 47;
- exactly 300,000 environment interactions per policy;
- external deterministic evaluation every 50,000 interactions;
- development tracks 2000–2008 and validation tracks 3000–3008;
- no use of exploratory tracks 1000–1008;
- no generation, evaluation, plotting, or inspection of reserved seeds
  4000–4017;
- no observation normalization and no reward normalization;
- one environment and one CPU thread per policy.

The observation audit sampled 16,200 states and confirmed that the existing
eight features remained within their declared `[0, 1]` bounds. The audit is in
[`observation-audit.json`](observation-audit.json).

## 2. Algorithm configurations and deviations from defaults

No algorithm hyperparameter was changed from the SB3 2.9 defaults. Values are
explicit in [`canonical.json`](canonical.json) rather than inherited silently.

| setting | SB3 SAC | SB3 PPO |
| --- | ---: | ---: |
| learning rate | 0.0003 | 0.0003 |
| network | 256×256 ReLU | 64×64 Tanh |
| gamma | 0.99 | 0.99 |
| batch size | 256 | 64 |
| replay capacity / rollout | 1,000,000 | 2,048 steps |
| learning starts / epochs | 100 steps | 10 epochs |
| tau / GAE lambda | 0.005 | 0.95 |
| train frequency / clip range | 1 step | 0.2 |
| gradient steps / entropy coefficient | 1 | 0.0 |
| SAC entropy coefficient | automatic | — |

The only operational adjustment was exact sample-budget stopping. PPO's last
partial rollout was stopped at 300,000, so its final policy contains updates
through 299,008 interactions. This avoids the normal rollout overshoot without
changing PPO's standard rollout length. See [`PROTOCOL.md`](PROTOCOL.md).

## 3. Completion and reproducibility

All 6/6 primary policies finished. The study produced 72 episode-level result
files: 2 algorithms × 3 seeds × 6 checkpoints × 2 splits. That corresponds to
648 external evaluation episodes, including 324 on validation.

An initial complete pass captured all physical results but no SB3 loss values
because the diagnostic sink did not inherit SB3's `KVWriter`. After correcting
that interface, all six policies were retrained from scratch in a separate
directory. Every episode record, aggregate summary, and policy-update count in
all 72 files was exactly equal between passes. The second pass is canonical and
contains complete diagnostics at every checkpoint.

Per-policy training wall time (external rollouts excluded, minor checkpoint I/O
included) was 1,518–1,534 s for SAC and 104–106 s for PPO on the recorded
one-thread CPU setup. SAC performed 299,899 actor/critic update steps per policy.
PPO completed 146 rollouts and 1,460 optimization epochs; its final partial
rollout was not optimized. Equal interaction budget does not mean equal compute.

## 4. Validation learning curves

RSR values are feasible tracks out of nine.

| algorithm | seed | 50k | 100k | 150k | 200k | 250k | 300k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| custom SAC V2-B | 11 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |
| custom SAC V2-B | 29 | 0/9 | 0/9 | 1/9 | 4/9 | 5/9 | 1/9 |
| custom SAC V2-B | 47 | 1/9 | 5/9 | 4/9 | 5/9 | 5/9 | 1/9 |
| SB3 SAC | 11 | 0/9 | 0/9 | 0/9 | 0/9 | 4/9 | 4/9 |
| SB3 SAC | 29 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |
| SB3 SAC | 47 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 1/9 |
| SB3 PPO | 11 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |
| SB3 PPO | 29 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |
| SB3 PPO | 47 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 | 0/9 |

The common external curve is plotted in
[`validation-rsr-learning-curves.png`](plots/validation-rsr-learning-curves.png).
Custom SAC seeds 29 and 47 peaked at 5/9 and collapsed to 1/9. SB3 SAC did not
show a peak-to-final decline, but useful behavior appeared only after 200k and
remained highly seed-dependent. PPO was flat at zero throughout; there is no
empirical sign that it was still improving at 300k.

## 5. Final validation metrics

`F` is feasible and `I+T` is incomplete and over time. All final learned
policies were speed compliant, but this is often the trivial consequence of
not moving.

| algorithm | seed | RSR | completion | time compliant | speed compliant | feasible energy mean / median [kWh] | travel time min / median / max [s] | F / I+T |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| SB3 SAC | 11 | 4/9 | 4/9 | 4/9 | 9/9 | 0.19560 / 0.19689 | 104.1 / 180.0 / 180.0 | 4 / 5 |
| SB3 SAC | 29 | 0/9 | 0/9 | 0/9 | 9/9 | — | 180.0 / 180.0 / 180.0 | 0 / 9 |
| SB3 SAC | 47 | 1/9 | 1/9 | 1/9 | 9/9 | 0.21274 / 0.21274 | 100.1 / 180.0 / 180.0 | 1 / 8 |
| SB3 PPO | 11 | 0/9 | 0/9 | 0/9 | 9/9 | — | 180.0 / 180.0 / 180.0 | 0 / 9 |
| SB3 PPO | 29 | 0/9 | 0/9 | 0/9 | 9/9 | — | 180.0 / 180.0 / 180.0 | 0 / 9 |
| SB3 PPO | 47 | 0/9 | 0/9 | 0/9 | 9/9 | — | 180.0 / 180.0 / 180.0 | 0 / 9 |

SAC's final mean RSR was 18.5%, with a 44.4-point seed range. PPO's final mean
was zero. Neither is close to the preregistered requirement that every seed
reach at least 8/9 RSR and that the seed range be at most 1/9.

## 6. Energy and references

| reference | privileged | RSR | completion | speed compliance | feasible energy mean / median [kWh] |
| --- | --- | ---: | ---: | ---: | ---: |
| fast compliant oracle | yes | 9/9 | 9/9 | 9/9 | 0.19805 / 0.18706 |
| conservative oracle | yes | 7/9 | 8/9 | 9/9 | 0.19686 / 0.18236 |
| random | no | 0/9 | 0/9 | 7/9 | — |

SAC seed 11 was jointly feasible with the fast oracle on four tracks and used
1.0018 times its paired mean energy. SAC seed 47 had one paired track and a
ratio of 1.0650. Those energy checks pass individually, but 5/27 feasibility is
far too sparse for an efficiency ranking. PPO had no feasible episode, so there
is no valid paired SAC–PPO energy comparison.

## 7. Representative behavior

The predefined selector chose the best and lowest final seed per algorithm.
Track 3000 maximized common feasibility; track 3005 maximized progress among
infeasible selected policies.

- SAC seed 11 completes both tracks without speed excess, but uses visibly
  oscillatory acceleration and action segments.
- SAC seed 29 stands still on track 3000 and reaches about 513 m on track 3005
  before stopping.
- all three PPO final policies produce the same physical failure on every
  validation track: deterministic full braking, position 0 m, 180 s duration,
  and approximately 0.04044 kWh auxiliary energy.
- no final learned trajectory exploits the speed constraint; inactivity and
  mid-route stopping dominate instead.

The plots are
[`track 3000`](plots/representative-trajectories-track-3000.png) and
[`track 3005`](plots/representative-trajectories-track-3005.png).

## 8. Internal diagnostics

At the final available logger dump, SAC's entropy coefficient had fallen to
roughly `6.4e-5`–`1.2e-4`; actor and critic losses remained finite. PPO reported
explained variance around 0.95–0.96, small approximate KL, and near-zero value
loss while its deterministic policy still stood still. These are useful
software diagnostics, but they demonstrate why internal convergence or mean
training reward cannot substitute for physical benchmark metrics.

One likely common difficulty is temporal credit assignment. With 0.1 s steps,
successful completion occurs after roughly 1,000 or more transitions. At
`gamma=0.99`, the terminal completion bonus is therefore discounted by about
`0.99^1000`, even though the undiscounted V2 analytical check strongly prefers
completion to standstill. This is an inference from the frozen reward and
trainer settings, not proof of a single root cause, but it explains why a
locally stable inactive policy can coexist with a sensible undiscounted return.

## 9. Historical SAC diagnosis and optional sanity check

The implementation comparison is documented in
[`CUSTOM_SAC_DIAGNOSTIC.md`](CUSTOM_SAC_DIAGNOSTIC.md). Major differences
include the custom separate value/target-value networks, a different Bellman
target, smaller networks, higher learning rate and tau, shorter replay, and
10,000 random transitions outside its nominal 300k online budget.

The late collapse of custom SAC is materially different from the SB3 SAC curves,
so the historical trainer should remain a diagnostic result rather than the
paper baseline. However, SB3 SAC's 18.5% mean RSR shows that changing the
implementation is insufficient.

The optional Pendulum experiment was not run. It would require a separate fair
cross-implementation protocol and would not overturn the direct same-task
LongiControl result that both modern reference algorithms fail acceptance.

## 10. Decision and next step

The outcome is **case D: both SB3 SAC and PPO remain poor**.

The previous instability was **not primarily caused by the historical SAC
implementation, and it is not SAC-specific**. Modern SAC removes the observed
late collapse and raises final common-validation RSR from 2/27 for custom SAC
V2-B to 5/27, but remains unreliable. PPO fails every track and seed. The common
scalar reward, long-horizon credit assignment, and/or task representation remain
substantial confounders.

Neither SB3 SAC nor PPO should be presented as a strong scalar paper baseline.
SB3 SAC is the only justified candidate for a bounded follow-up because two
seeds first improve at 250k–300k; PPO's flat curve does not justify automatic
extension. Do not launch 1M training blindly. First preregister a small
horizon/discount and terminal-credit diagnostic while holding the physical task
and evaluator fixed.

Most importantly, this result triggers the trainer gate for the proposed
human-versus-agent reward-design study: automated reward engineering cannot yet
be evaluated fairly because trainer/task instability would dominate the
comparison.
