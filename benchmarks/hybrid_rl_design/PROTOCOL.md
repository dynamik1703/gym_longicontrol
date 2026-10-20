# Proposed Hybrid RL Experiment Protocol

Status: **design frozen; execution not authorized**.

## Question and scope

Can a learned causal target-speed strategy reduce signed net energy while
retaining the reliable feasibility behavior of the frozen causal physical
reference, and does it improve beyond a deterministic time-aware strategy using
the same information and shield?

This is an explicitly Hybrid RL study. It is not end-to-end RL, a pure algorithm
comparison, behavior cloning, policy distillation or imitation of the Oracle.

## Frozen conditions

- Training seeds: 11, 29, 47.
- Training stream per seed: immutable 865-row `CANONICAL_11/29/47` streams;
  ordered hashes are in `split_ledger.json`.
- Collector: current reproducible TRANSITIONAL convention.
- Budget: exactly 300,000 real native transitions and 30,000 gradient updates
  per policy; no environment-model or imagined transitions count as real.
- Diagnostics: fixed policies at 50k, 100k, 150k, 200k, 250k and 300k on the
  frozen 18-task `HYBRID_RL_DIAGNOSTIC_V1`.
- Endpoint: exactly 300k. No best-checkpoint, early-stopping or performance
  checkpoint selection.
- Final set: 36 frozen tasks in `HYBRID_RL_VALIDATION_V1`, opened once only
  after all three endpoints and both controls are frozen.
- Canonical environment: `StochasticTrack-v1`, 1,000 m, 140 s, zero speed
  tolerance, 0.1 s dynamics, 1,800-step wrapper, production metrics/feasibility.

Policy R checks all 2,595 reachable training rows against both new sets and all
protected prior content. Diagnostic and Validation use namespaces beginning
10,000,000 and 10,100,000. Validation is content-frozen but has zero Hybrid
performance episodes at preparation completion. Paper 4000–4017 is numeric-only
excluded and remains unconstructed.

## Algorithm and signals

The exact learner and controller constants live in `canonical.json`. The
strategy is constrained SAC with a 16D causal observation, one scalar target,
`256,256` actor/twin critics, batch 256, replay 300k, 10k warm-up, `gamma=.99`,
one-step targets, `tau=.005`, Adam and learned entropy.

At 1 Hz the actor proposes 20–60 km/h; the target is held for ten native steps.
Replay retains every native transition, proposed/executed actions, intervention
metadata, energy, time reserve and hard costs. TRANSITIONAL updates occur at the
frozen mean ratio 0.1 after warm-up until exactly 30k updates.

Primary optimization is lexicographic:

```text
Priority 1: satisfy completion, deadline and exact speed constraints
Priority 2: minimize signed net energy among feasible trajectories
```

The step reward is negative signed energy scaled by 1,000. Separate cost critics
use positive speed excess and terminal infeasibility with zero cost limits. The
actor also pays the frozen 0.05 proposal/execution deviation regularizer. The
runtime shield has final action authority; learned critics do not certify safety.

An episode that is incomplete, late or has any positive speed excess is
infeasible regardless of reward. Its energy is excluded from optimization
success comparisons. Energy cannot compensate for a hard violation.

## Pre-training gates

Training remains blocked unless all are true:

1. frozen controller/config hashes match;
2. disabled influence has exact action/trace/metric/energy parity;
3. endpoint-speed, known-braking and preview fixtures pass;
4. deadline-bound fixtures pass with zero falsifications;
5. `SHIELD_DOMINATES_POLICY` is false under frozen thresholds;
6. causal-information AST/API boundary passes;
7. Policy-R split and Paper fail-closed checks pass;
8. projected runtime ≤24 h and retained disk ≤12 GB.

All currently pass, but `training_authorized` remains false. Separate explicit
authorization is required.

## Training integrity

Future execution must use atomic manifests, single-run leases, 10k recovery
checkpoints, exact counters and deterministic task-stream cursors. Recovery may
resume only a verified interrupted checkpoint. It may not restart a completed
condition, repeat a Diagnostic policy-track pair or open final Validation early.

No contextual RL Validation outcome from Phase A may tune the action interface,
shield, reserve, rewards, network, update budget or new splits.

## Final evaluation and targets

The arms and Oracle role are frozen in `BASELINE_COMPARISON.md`. Report every
observed value. Primary target per Hybrid seed:

```text
RSR >= 35/36
```

Energy milestone per seed, on tracks where both Hybrid and frozen causal are
feasible:

```text
mean 100 * (energy_causal - energy_hybrid) / energy_causal >= 10%
```

Incremental-learning milestone per seed, on tracks where Hybrid, causal and the
non-learning control are all feasible:

```text
mean 100 * (energy_nonlearning - energy_hybrid) / energy_nonlearning >= 3%
```

Also report medians, ranges, per-track values and paired bootstrap intervals as
descriptive uncertainty. The fixed thresholds, not interval significance,
determine the preregistered labels. Missing feasibility automatically prevents
an energy-success label for that seed.

## Stop conditions

Stop and preserve evidence if fallback parity changes, source/split hashes fail,
Paper content is touched, a task pair would repeat, resource limits are exceeded,
the viability bound is falsified, or counters cannot be recovered exactly. Do
not repair a final result, loosen zero speed tolerance, change the shield to
increase influence or open another Validation set.
