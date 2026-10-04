# Constrained RL V2.1 diagnosis protocol

## Status and ordering

The governing two-phase protocol was supplied before this diagnosis. This file
records that protocol before any V2.1 optimization. Phase A produced the formal
classification **F (mixed / inconclusive)**, so the stop rule was activated and
Phase B was not run. There is no V2.1 policy, training configuration, or result.

## Question

Why do six of the 27 frozen Constrained RL V2 Validation episodes violate the
strict speed requirement despite 27/27 completion and deadline compliance?

Phase A distinguishes:

- small boundary overshoots;
- late braking at speed-limit reductions;
- deadline-versus-speed pressure;
- weak Lagrange enforcement;
- general optimization instability; and
- a mixed result for which no single intervention follows.

## Frozen sources

Only these existing artifacts may be replayed or read:

- `benchmarks/constrained_rl_v2/results.json`;
- `benchmarks/constrained_rl_v2/trajectories.json`;
- final 300k checkpoints and raw logs under
  `runs/constrained-rl-v2-20261002/`;
- Validation tracks 3000--3008.

Scalar V1/V2, custom SAC, SB3 SAC/PPO, credit assignment, Constrained V1,
Constrained V2, `TaskSpecification`, `EpisodeMetrics`, and public environments
are frozen. Seeds 4000--4017 remain sealed, and seeds 1000--1008 are not used.

The replay must reproduce all stored final episode metrics before its samples
can enter the diagnosis. It performs no optimizer update.

## Event definitions

A violation event is a contiguous run of native 10-Hz samples with strictly
positive `velocity_m_s - speed_limit_m_s`. Each post-step sample represents its
preceding integration interval.

For every event:

```text
duration = sum(dt)
integrated overspeed = sum(max(0, v - v_limit) * dt)
distance while violating = final position - pre-event position
```

The nearest speed-limit transition is selected by absolute longitudinal
distance. Its crossing time is linearly interpolated between native samples;
signed distance and time are positive after the transition.

An event that starts within one native control interval after a downward limit
transition is labelled late braking. A separate descriptive boundary category
uses the physical maximum one-step velocity change,
`3 m/s² * 0.1 s = 0.3 m/s`; this never changes external feasibility.

## Comparisons

Every failed episode is compared with a successful episode from the same
training seed. The comparison track minimizes a fixed distance over number of
transitions, number/sum/maximum of downward changes, initial limit, and mean
limit; ties use the smaller track seed.

Successful and failed episodes are compared on speed severity, deadline state,
acceleration, deceleration, jerk, action variance, action sign changes, and
action total variation. FSRL training logs supply cost returns, multipliers, and
checkpoint Validation RSR.

## Phase-B gate

Exactly one V2.1 intervention would be permitted only if Phase A identified one
dominant, generalizable mechanism. All other algorithm, objective, deadline
cost, observations, actions, 10-Hz semantics, seeds, splits, training budget,
and external evaluation would remain frozen.

The acceptance threshold would remain at least 24/27 feasible Validation
episodes, with success in every training seed, no systematic standstill, and no
material completion or deadline regression.

If mechanisms are mixed and a margin-only or anticipatory-only change cannot
address all failure classes, the preregistered outcome is Gate F: do not train
V2.1, freeze Constrained RL, and separately preregister Requirement-Conditioned
RL as the next research stage.

