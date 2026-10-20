# Design

## Reference A: Full-route Physics Oracle

The Oracle receives the complete generated `Track`, the current production
`VehicleState`, exact dynamics and canonical task. It builds the set of all
future speed transitions and selects a deterministic action as follows:

1. Invert the production action interpolation to approach the current cruise
   cap without crossing the current limit.
2. Simulate that proposed action through `advance`.
3. Check `position[t+1]`, `velocity[t+1]` and the right-endpoint limit.
4. For every future lower limit, integrate exact action `-1` and require the
   braking distance plus the frozen 0.5 m position margin to fit.
5. If the proposal is unsafe, execute maximum braking.

The frozen velocity margin is `1e-6 m/s`; the evaluator retains exactly zero
tolerance. A 37 m/s initial rollout establishes a time-efficient feasible
witness. Ten deterministic cruise-cap bisections then seek lower energy while
retaining all hard requirements. Only candidates actually executed in the
original environment are eligible. The feasible candidate with lowest signed
net energy is retained; otherwise the initial witness remains.

This procedure does not certify global energy optimality. "Oracle" denotes
privileged route information, not a proven theoretical optimum.

## Reference B: Observation-limited controller

The causal controller receives only the ordinary normalized 8D observation.
It decodes current velocity/acceleration, current and two preview limits, and
their visible distances. It has no `Track` object, no Oracle action and no
precomputed trajectory.

Disclosed memory consists of:

- elapsed time, position, velocity and acceleration propagated from reset using
  its own executed actions and exact known dynamics;
- absolute positions/limits of transitions previously observed through the
  sensor.

It uses the same next-endpoint and braking checks as the Oracle, but only for
remembered/visible transitions. A frozen 60 km/h cruise cap is the additional
causal safety rule. It covers an unseen legal 60→20 km/h reduction within the
150 m preview, including consecutive-downshift cases. It is deliberately
conservative and does not use holdout-specific tuning.

The controller is not equivalent to a memoryless 8D neural actor. Its strong
result motivates testing recurrent state estimation and physics-structured
action generation in a future, separately authorized RL study.

## Lexicographic objective

The order is fixed:

1. complete 1,000 m;
2. finish by 140 s;
3. record exactly zero positive speed excess;
4. among feasible Oracle candidates, minimize signed net energy.

Energy never compensates for a hard-requirement failure. Infeasible energy is
not included in feasible-only means or medians.

## Fixtures

Deterministic tests cover constant 20/100 km/h, upward changes, a maximum legal
downshift, consecutive downshifts, a short braking window, endpoint boundary,
insufficient braking, forced standstill and an apparently tight deadline. These
are correctness fixtures, not stochastic population samples.
