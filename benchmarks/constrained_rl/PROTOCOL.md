# Explicit constrained-RL benchmark protocol

Status: **preregistered before every main training or validation run**. A library
compatibility import and wrapper smoke test were allowed before freezing this file;
neither produced validation outcomes.

## Scientific question and frozen CMDP

Does separating the optimization objective from operational requirements improve
learning compared with the frozen scalar formulation?

- Simulator: unchanged `StochasticTrack-v1`, 0.1 s physical integration,
  `action_repeat = 1`, at most 1,800 steps per physical episode.
- Objective reward at native step `t`:
  `r_t = -step_energy_kwh / 0.25 kWh`. Energy is signed net battery energy, so
  regeneration produces positive objective reward. The fixed scale is numerical
  only. There is no progress, time, speed, or completion reward.
- Cost 1, `speed_integral_m`:
  `max(0, velocity_m_s - speed_limit_m_s) * 0.1 s`, evaluated at the same
  right-endpoint sample as `EpisodeMetrics`. Its episodic limit is 0 m.
- Cost 2, `task_failure`: zero except on termination/truncation, where it is 0
  exactly when the route is complete within 140 s and 1 otherwise. Its expected
  episodic limit is 0.
- Cost order is frozen as `[speed_integral_m, task_failure]`; the two costs use
  separate critics and separate multipliers. Speed is deliberately absent from
  the terminal task cost because it already has its own constraint.

The final budget may truncate one otherwise unfinished training episode. That
episode receives task-failure cost 1. This affects at most one episode per seed and
prevents the accounting boundary from being mislabeled as successful.

FSRL optimizes discounted reward/cost critic targets and adapts multipliers from
undiscounted collected episode costs. These are expected-cost constraints, not hard
trajectory guarantees. External episode-wise `TaskSpecification`, `EpisodeMetrics`,
and `is_feasible` remain authoritative; validation reward and internal costs are
never used as outcomes.

## Learner and compatibility adapter

The learner is the established FSRL SAC-Lagrangian implementation at Git commit
`e056fc9498d5d037869533da7cf976acf462f918`, with Tianshou 0.5.1. It supports
continuous Box actions and constructs one reward critic plus one critic per cost.
The pinned FSRL revision has two narrow vector-cost plumbing defects: its n-step
helper does not split cost columns and its collector sums columns when returning
episode statistics. The local adapter only splits the existing cost vector and
restores the wrapper's exact per-constraint episode totals. It does not alter SAC,
critic targets, losses, PID updates, or actor optimization.

Frozen library defaults/recommendations:

- actor/critic learning rates: 5e-4 / 1e-3;
- two hidden layers of 128 units; replay 100,000; batch 256;
- gamma 0.99; tau 0.05; two-step returns; 0.1 updates per transition;
- automatic entropy tuning, alpha LR 3e-4, effective initial alpha 1.0;
- PID coefficients `(Kp, Ki, Kd) = (0.05, 0.0005, 0.1)`;
- initial multipliers 0/0; Lagrangian rescaling enabled;
- one completed episode per collection; deterministic external evaluation.

The PID integral coefficient `Ki=0.0005` is the closest analogue to a multiplier
learning rate; FSRL does not expose a separate multiplier LR. No sweep is run.

## Data split, budget, and evaluation

- Independent training seeds: 11, 29, 47.
- Development/calibration tracks: 2000--2008.
- Validation tracks: 3000--3008.
- Historical tracks 1000--1008 are not used for tuning.
- Sealed paper tracks 4000--4017 must not be generated, evaluated, inspected, or
  plotted.
- Budget: exactly 300,000 underlying simulator transitions per training seed.
- Evaluations occur after the first completed training episode ending at or after
  targets 50k, 100k, 150k, 200k, 250k, and 300k. Target and actual count are both
  persisted; the lifetime wrapper makes the final count exactly 300k.

Every checkpoint records all external physical metrics per track, completion/time/
speed rates, exclusive failure modes, objective return, physical energy, both costs,
both multipliers, all critic losses, actor loss, entropy alpha, interactions, and
updates. Representative trajectories use the median-final-RSR seed (lower seed
breaks ties) on fixed tracks 3000 and 3005 and show native-step position, velocity
and limit, acceleration, jerk, action, and cumulative energy.

## Frozen acceptance and material-improvement rules

The strong baseline threshold is unchanged: every seed needs at least 8/9 RSR,
8/9 completion, and 8/9 speed compliance; the across-seed RSR range must be at
most 1/9. No seed may lose more than 2/9 RSR from its best checkpoint to 300k.
Among sufficiently feasible policies, paired mean energy may be at most 1.25 times
the fast compliant reference.

Frozen SB3 SAC V2-B has final validation RSR 4/9, 0/9, and 1/9 for seeds 11, 29,
and 47: pooled mean 5/27. "Material improvement" requires both a mean gain of at
least 2/9 (therefore at least 11/27 final successes) and a positive paired-seed
RSR improvement in at least two of three seeds.

Systematic standstill/crawling is flagged when pooled completion is at most 1/3
while speed compliance is at least 8/9. Severe optimizer instability is systematic
only if at least two seeds have a non-finite required diagnostic, a multiplier above
1,000, or a critic loss above 1e6.

Decision order:

1. **E** if constrained optimization is systematically unstable.
2. **A** if every strong-baseline, stability, and feasible-energy check passes.
3. **D** if the predefined standstill/crawling condition holds.
4. **B** if material improvement passes without A, D, or E.
5. **C** otherwise: constrained learning mostly reproduces scalar failure modes.

No automatic 1M extension is permitted. A longer run may only be recommended if
all three validation curves are clearly and consistently improving at 300k, not
merely fluctuating. No algorithm or task-formulation response is made during the
frozen main study.
