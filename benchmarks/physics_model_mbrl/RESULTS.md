# Physics-Dynamics MBRL results

The evaluated controller is a learned SACLag actor trained with exact
one-step physics-model augmentation. It is not MPC and performs no model
query at evaluation.

## Validation

- Seed 11: 2/9 RSR
- Seed 29: 0/9 RSR
- Seed 47: 7/9 RSR
- Pooled RSR: 9/27
- Completion: 20/27
- Deadline compliance: 16/27
- Speed compliance: 18/27
- Preregistered interpretation: `NO_MATERIAL_IMPROVEMENT`

Relative to frozen Constrained V2, the episode-count deltas are -12 RSR,
-7 completion, -11 deadline compliance and -3 speed compliance. These raw
deltas are clear degradation context under the preregistered wording. The exact
categorical label remains `NO_MATERIAL_IMPROVEMENT`, because no separate
quantitative degradation cutoff was preregistered.

## Energy and failures

Feasible-only energy uses 9 real Validation episodes; mean = 0.20377195538338214, median = 0.17719600888504017 kWh.

Exclusive failure modes: `{"deadline_and_speed": 2, "deadline_only": 2, "feasible": 9, "incomplete": 7, "other": 0, "speed_only": 7}`.

## Development and accounting

Development RSR counts at 50k, 100k, 150k, 200k, 250k and 300k real
transitions were:

- Seed 11: 0, 1, 3, 2, 9, 0
- Seed 29: 0, 0, 0, 0, 0, 0
- Seed 47: 1, 2, 3, 3, 5, 7

The curves show strong seed dependence and, for seed 11, a complete late-stage
collapse after a feasible 250k checkpoint.

Each policy used exactly 300,000 real transitions, 2,902,500 generated
synthetic transitions, 3,712,128 sampled synthetic rows, 30,000 RL gradient
updates and 1,161 model refreshes. Pooled totals are therefore 900,000 real
transitions, 8,707,500 generated synthetic transitions, 11,136,384 sampled
synthetic rows and 90,000 RL updates. All policies completed in one attempt
without resume or fresh restart. Recorded training wall-clock was 48,446.94 s
(13 h 27 min 27 s), of which seed 47 accounted for 40,017.87 s during a period
of reduced host availability.

The bounded Physics parity confirmation used 18 new simulator transitions and
matched all recorded physical, observation, energy and cost quantities with
zero absolute error; terminal semantics accuracy was 1.0. The reused frozen
495-transition artifact remained hash-identical.

## Context

Frozen Constrained V2 achieved 21/27 RSR. This is a close historical
comparison, not a perfect one-factor causal control, because the Physics condition mixes one-step synthetic targets with the frozen two-step real-data path.

Development curves, full compact episode metrics, source-wise TD diagnostics, learner losses, multipliers, costs, accounting, parity and wall-clock are
stored in `results.json`.
