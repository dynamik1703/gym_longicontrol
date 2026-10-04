# Credit-assignment diagnostic protocol

Status: **preregistered before any run of this study**.

This study asks whether the weak and seed-sensitive Scalar SAC result is mainly
caused by the effective discount horizon or by asking the policy to choose an
action at every 0.1 s simulator integration step. It does not redesign the
reward and does not compare task-specification paradigms.

## Frozen common setup

- Environment: `StochasticTrack-v1`, with the unchanged 0.1 s physical step and
  a maximum of 1,800 simulator transitions per episode.
- Task: finish, travel time at most 140 s, and maximum speed excess at most
  0 m/s. Energy is evaluated only after feasibility.
- Training signal: the existing V2-B Balanced scalar reward, including its
  weights and normalization, without any added terminal signal.
- Learner: Stable-Baselines3 SAC with the preceding study's network, replay,
  optimizer, target update, entropy, and update-frequency settings.
- Training seeds: 11, 29, and 47.
- Development tracks: 2000--2008. Validation tracks: 3000--3008.
- Tracks 1000--1008 are historical only. Tracks 4000--4017 remain sealed and
  must not be generated, evaluated, inspected, or plotted.

The optional terminal-credit condition is deliberately omitted. No principled
fixed bonus magnitude was available before the experiment, and choosing one
would turn the diagnostic into reward tuning.

## Frozen 2 x 2 matrix and hypotheses

| Condition | Action repeat | SAC gamma | Intervention |
|---|---:|---:|---|
| A | 1 | 0.99 | baseline |
| B | 1 | 0.999 | discount horizon only |
| C | 5 | 0.99 | decision horizon only |
| D | 5 | 0.999 | both |

H1: B will outperform A because delayed completion consequences retain more
weight. H2: C will outperform A because the nominal 140 s decision horizon is
reduced from about 1,400 to about 280 decisions without changing physics. The
interaction is tested by D. All hypotheses concern physical validation metrics,
not training return.

At the start of a nominal 140 s episode, the terminal weight is approximately
`0.99^1400 = 7.8e-7` in A, `0.999^1400 = 0.2465` in B,
`0.99^280 = 0.0600` in C, and `0.999^280 = 0.7557` in D.

## Action-repeat and interaction accounting

The benchmark wrapper holds the selected action for up to five calls to the
unchanged environment. It sums the five V2-B rewards, returns the final
observation, and stops immediately on termination or truncation. Consequently,
`EpisodeMetrics`, speed-limit checks, dynamics, and energy integration still run
at 10 Hz. Historical environments are not modified.

Every condition receives exactly 300,000 underlying simulator transitions per
training seed. The repeat wrapper owns a lifetime simulator counter and shortens
only the final repeated decision if necessary to hit the budget exactly. SB3's
timesteps are recorded separately as `agent_decisions`; `_n_updates` is recorded
as `gradient_updates`. Evaluation checkpoints are triggered by the first agent
decision ending at or after 50k, 100k, 150k, 200k, 250k, and 300k simulator
transitions. Both the target and exact observed simulator count are persisted.
The final checkpoint is exactly 300k because the wrapper enforces the cap.

The repeated wrapper returns the **undiscounted sum** of internal per-step
rewards. SAC then applies gamma once per policy decision. Thus gamma is a
per-decision discount in the primary algorithmic comparison.

## Physical-time interpretation

For a delay `t`, reported real-time weight is
`gamma ** (t / (0.1 * action_repeat))`. The equivalent exponential time
constant is `-0.1 * action_repeat / log(gamma)`. The report must show weights at
10, 30, 60, and 140 s.

Holding gamma constant per decision is not physically time-consistent. Matching
A's 0.99 discount per 0.1 s at repeat 5 would require `0.99^5 = 0.9509900499`
per repeated decision. This value is interpretation only and is not a fifth
condition.

## Outcomes and analysis

The primary final outcome is validation requirement satisfaction rate (RSR),
reported per training seed, as the across-seed mean and range, and per track.
Completion, time compliance, speed compliance, and exclusive failure modes are
reported separately at every checkpoint. Validation RSR curves show every seed.
Training seed is the experimental unit; the 27 validation episodes are not
treated as 27 independent trained policies. Track-level comparisons are paired
where relevant.

Energy is lexicographically secondary. Mean, median, per-track, and paired energy
are reported only for feasible episodes; a stationary or rarely successful
policy is never called energy efficient.

The final trajectory selection is deterministic: for each condition choose the
training seed with median final validation RSR (lowest seed breaks ties), then
plot tracks 3000 and 3005 at native 10 Hz. Position, velocity and limit,
acceleration, jerk, action, and cumulative energy are retained. Standard SB3
actor, critic, and entropy-coefficient diagnostics are persisted without
changing SB3 internals.

## Predefined decision rule

The preceding acceptance target is retained: each seed must reach at least 8/9
RSR, completion, and speed compliance, and the RSR range must be at most 1/9.
A mean-RSR gain of at least 2/9 over A, present in at least two of three paired
training seeds, is called a material diagnostic improvement.

The final gate is classified as follows:

1. If no intervention is materially better and no intervention is robust: E,
   none materially improves robustness.
2. If an intervention improves materially but the best condition misses the
   stability bound or fails to improve at least two seeds: F, seed instability
   remains dominant.
3. Otherwise, B best supports A (discount horizon), C best supports B (decision
   horizon), and D best supports C (both required). Ties are broken by minimum
   per-seed RSR, then mean RSR, then the simpler intervention.

The study does not establish that scalar rewards are fundamentally unsuitable,
that SAC cannot solve LongiControl, or that another task formulation is better.
It only narrows the cause of instability. No 1M-step extension, gamma/repeat
sweep, PPO run, terminal bonus, or reward response is automatic.
