# Hybrid RL Design

## Architecture decision

| Criterion | A — native action + projection | B — target speed / cruise cap | C — residual around fallback |
| --- | --- | --- | --- |
| Controllability | Maximum but easy to propose physically erratic actions | Directly controls the energy/time operating point | Local authority can be too narrow for large energy changes |
| Speed safety | Requires projecting arbitrary traction/brake commands | Deterministic physical controller retains endpoint/braking semantics | Safe if residual bound is conservative, but bound is hard to justify |
| Deadline protection | Arbitrary earlier actions can destroy reserve quickly | Target can be screened against a causal completion bound | Small negative residuals can accumulate hidden delay |
| Smoothness | Must be regularized/projected | 1 Hz held target naturally reduces chatter | Usually smooth near fallback |
| Learning stability | Large intervention discontinuities | One bounded scalar with meaningful monotonic semantics | Stable, but learning signal may be tiny |
| Observability | Policy must infer low-level actuator consequences | 16D causal state exposes time/distance/fallback context | Must infer how residual composes with braking modes |
| Interpretability | Low | High: proposed km/h target and rejection reason | Medium |
| Energy potential | High in theory | High enough to trade the fallback's time reserve for lower speed | Potentially constrained by reference behavior |
| Implementation risk | Highest | Moderate | Moderate |

**Selected primary architecture: Option B, target-speed/cruise-cap control.**

The selection is frozen before training and is not a sweep. Option B gives the
policy meaningful energy authority while keeping native action generation,
right-endpoint speed checks and transition braking deterministic. It exposes
one interpretable scalar and makes shield interventions diagnosable. Option C
remains a future alternative only if this preregistered architecture is
scientifically terminated; it may not be selected from the same Validation.

## Runtime dataflow

```text
ordinary 8D observation + disclosed causal memory
                         |
                         v
         constrained SAC strategy (1 Hz)
             scalar target in 20–60 km/h
                         |
                         v
        exact-dynamics native-action generator
                         |
                         v
 endpoint speed + known braking + unknown preview + deadline viability
              | accepted                 | rejected
              v                          v
       proposed native action      frozen causal action
              \__________________________/
                         |
                         v
               StochasticTrack-v1
```

The scalar policy action `u ∈ [-1,1]` maps linearly to
`20 + 20(u + 1) km/h`. The policy refreshes it every ten native steps (1 s);
the target is held between decisions. The deterministic controller clips the
effective target to the current limit and frozen 60 km/h cap, inverts the exact
vehicle action interpolation, and proposes a native action.

## Exact strategy observation

The first eight fields are the unchanged ordinary observation. The remaining
eight are:

1. internal position / 1,000 m;
2. internal elapsed time / 140 s;
3. remaining distance / 1,000 m;
4. remaining time / 140 s;
5. previous executed native action;
6. previous target normalized to `[-1,1]`;
7. causal completion-bound slack / 140 s, clipped to `[-1,1]`;
8. current frozen-fallback native action.

Position, time, velocity and acceleration are propagated internally from reset
and executed actions. They are controller memory, not privileged environment
state. Equality against observation velocity/acceleration remains checked by
the frozen controller.

## Proposed learner

The treatment is shielded constrained Soft Actor-Critic:

- actor and twin reward/cost critics: MLP `256,256`, one scalar action;
- Adam; actor/critic/entropy learning rates `3e-4`, constraint-multiplier rate
  `1e-3`; target entropy `-1`;
- `gamma=0.99`, one-step targets, `tau=0.005`, batch 256;
- replay capacity 300,000; 10,000-transition warm-up;
- TRANSITIONAL collection, 0.1 update per native transition;
- exactly 300,000 real transitions and 30,000 updates per seed 11/29/47;
- diagnostics at 50k increments; the fixed 300k policy is the endpoint; no
  best-checkpoint selection.

The reward is `-1000 × signed step energy_kWh`. Hard costs are positive speed
excess and terminal infeasibility (incomplete or after 140 s), each with limit
zero. A `0.05 × |proposal - executed|` actor regularizer teaches actions that
survive the shield; intervention is not redefined as physical feasibility.

This optimization remains subordinate to the runtime shield and the evaluation
decision hierarchy. A finite learned multiplier is not treated as a safety
proof, and no energy result can compensate for a failed hard requirement.

## Logged intervention contract

Every native step records strategy target, proposed action, frozen fallback,
executed action, accepted/overridden flag, rejection reason, endpoint/braking/
preview/deadline booleans, completion-bound estimate, native-action influence,
physics-call count, energy increment and time-reserve change.

`SHIELD_DOMINATES_POLICY` is raised before main training if fixture override
fraction exceeds 0.85, native-action influence is below 0.10, or fewer than four
of six fixtures are influenced. The current fixture result is 0.176 override,
0.634 influence and six influenced fixtures. Safety thresholds are never relaxed
to improve influence.
