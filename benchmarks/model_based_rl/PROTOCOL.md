# Frozen protocol

This document is frozen before main policy training. `canonical.json` is authoritative.

## Matrix and budgets

| Condition | Seeds | Real transitions per policy |
|---|---:|---:|
| Learned-Dynamics MBRL | 11, 29, 47 | 300,000 |
| Physics-Dynamics MBRL | 11, 29, 47 | 300,000 |

The total physical training budget is 1,800,000 transitions. Synthetic transitions,
model updates and RL gradient updates are reported independently. No third condition,
horizon sweep or ratio sweep is authorized.

## Frozen task and learner

Environment, 8D observation, action, 140 s task, zero speed tolerance, energy objective,
speed integral, optimistic deadline-deficit integral, seeds and real budget match
Constrained RL V2. FSRL SACLag remains: actor LR `5e-4`, critic LR `1e-3`, `[128,128]`,
automatic alpha (`3e-4`, initial 1), gamma `.99`, tau `.05`, n-step 2 for real data,
buffer 100k, batch 256, update-per-real-step `.1`, PID `(.05,.0005,.1)`, rescaling,
zero initial multipliers and deterministic evaluation.

Synthetic H=1 samples necessarily use a one-step bootstrap. This is common to both
conditions. PID receives only undiscounted costs from real completed episodes.

## Model schedule

- warmup: 10,000 real transitions; real-only RL updates before this point;
- refresh: every 250 real transitions, including 10,000 and 300,000;
- generated per refresh: 2,500;
- rollout horizon: exactly one native step (0.1 s);
- sources: real replay only;
- action: current stochastic actor;
- post-warmup update batch: exactly 128 real plus 128 model;
- RL updates: exactly 30,000 per completed policy.

Learned ensembles train only from the current run's real transitions. No historical,
demonstration, Validation, CRL/GCSL or oracle trajectory may initialize them.

## Splits and gates

Training uses a dedicated checkpointed RNG that rejects 1000-1008, 2000-2008,
3000-3008 and 4000-4017. Development 2000-2008 is evaluated at 50k increments with an
isolated environment and cannot perturb training RNG. Validation 3000-3008 may open
once only after all six runs are `COMPLETED`, exactly 300k/30k, final hashes exist and
Development is complete. Paper tracks 4000-4017 remain hard-blocked.

External `EpisodeMetrics` plus `is_feasible` is authoritative. Models, predicted safety
and synthetic success never count. Feasible-only energy is null when no real feasible
episode exists.

## Outcomes

Report per-seed and pooled descriptive RSR, completion, deadline and speed compliance,
exclusive failure modes, Development curves, seed robustness and feasible-only energy.
The 27 Validation episodes are three policies evaluated on nine tracks, not 27 policies.

## Interpretation gates (priority E, then A-D)

- **E — unsafe/unstable learned model:** any non-finite model/policy value, heldout
  learned false-safe rate at least 1%, or any false-safe speed excess above 0.1 m/s.
- **A — Physics robustly improves:** Physics RSR at least 22/27, speed compliance at
  least 24/27, and completion/deadline both 27/27.
- **B — both improve:** both RSR at least 22/27, both preserve 27/27
  completion/deadline, and Learned is within three successes of Physics.
- **C — Physics helps, Learned does not:** Physics satisfies A and either Learned is at
  most 21/27 or the Physics-Learned RSR gap is at least four.
- **D — neither improves:** neither condition exceeds 21/27 RSR.

If A and B overlap, B is the more specific interpretation. These are descriptive
thresholds, not significance claims. No threshold may be changed after results.

## Execution and failures

Run states are `NOT_STARTED`, `RUNNING`, `INTERRUPTED`, `COMPLETED`; every attempt is
retained. Exclusive lock files prevent duplicates. Checkpoints atomically include
policy/targets, optimizers, entropy, PID, both replays, model and optimizers (Learned),
normalizers, elites, environment/track/collector state, all RNGs, counters, diagnostics
and hashes. A checkpoint is accepted only after SHA-256 verification.

The preparation snapshot does not claim crash-transparent resume. A hard failure is
`INTERRUPTED`; resuming or restarting requires separate explicit authorization and a
selected verified checkpoint. Consumed resources are never silently erased. This is a
stronger safeguard than replaying from an uncertain state.

## Authorization

```text
main_training_authorized = false
main_training_enabled = false
validation_authorized = false
paper_test_authorized = false
```

Preparation does not change these values.
