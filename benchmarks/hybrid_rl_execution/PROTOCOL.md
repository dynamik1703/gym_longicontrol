# Executable Hybrid RL V1 Protocol — Revision 3

Status: **prepared; main training, final Validation, and Paper execution are
not authorized**.

## Scientific question

Can a learned causal target-speed strategy reduce signed net energy while
retaining the feasibility behavior of the unchanged frozen causal controller,
and can it improve beyond the frozen deterministic `TimeAwareEnergyStrategy`
using the identical shield?

## Revision relative to the frozen design

Revision 2 made one necessary pre-training amendment: replay, SAC actions, and
Bellman targets now follow the actual 1 Hz held-action SMDP. Native dynamics,
energy/speed measurements, and shielding remain at 10 Hz. The exact equations
are frozen in `TEMPORAL_SEMANTICS.md` and `SMDP_TARGETS.md`.

Revision 3 changes only update scheduling. There are no updates through native
transition 10,000 and no deferred 1,000-update burst. The cumulative target is
`0` for `N <= 10,000`, then
`min(30,000, ((N - 10,000) * 30,000) // 290,000)`. The first update is at
native transition 10,010 and the 30,000th at 300,000. SMDP, shield, streams,
budgets, model, losses, and all other frozen scientific parameters are unchanged.

No frozen physical parameter, controller margin, deadline, route split,
acceptance threshold, action bound, signal scale, network size, learning rate,
or budget was tuned from policy performance.

## Frozen treatment

- Causal 16D strategy observation; no route, Oracle, demonstration, or future
  realized state.
- One scalar normalized target mapping linearly to 20–60 km/h.
- Constrained SAC actor and critics with `256,256` hidden units.
- Twin reward critics; speed, terminal, and intervention critics.
- Automatic entropy, target entropy `-1`, `tau=.005`, batch 256.
- `-1000 * signed physical step energy_kWh` reward.
- Nonnegative native endpoint speed excess and terminal infeasibility costs,
  both with zero limits.
- `0.05` times the learned discounted sum of native actuator intervention
  deviation as a soft actor regularizer.
- Proposed target, not executed native action, in critic replay.

## Budgets and streams

- Training seeds: 11, 29, 47; existing immutable 865-row streams.
- Exactly 300,000 real native transitions and 30,000 SAC updates per seed.
- Ten-thousand-native-transition warm-up; deterministic post-warm-up linear
  update accounting described in `TEMPORAL_SEMANTICS.md`.
- Macro-decision counts are reported separately and do not replace the native
  interaction budget.
- Actor-only Diagnostic at 50k increments. No best-checkpoint selection; 300k
  is the primary endpoint.

## Comparison and acceptance

The three fair arms are the unchanged causal 60 km/h reference, the frozen
time-aware non-learning strategy through the same shield, and three separately
trained Hybrid policies through that shield. The privileged full-route Oracle
is contextual only.

The unchanged hierarchy is:

1. RSR at least 35/36 for every Hybrid seed;
2. at least 10% mean paired feasible-only energy reduction versus causal;
3. at least 3% mean paired feasible-only reduction versus non-learning.

Energy never compensates for completion, deadline, or exact speed failure.

## Recovery and execution

The control plane requires a clean `research/hybrid-rl-execution` worktree,
verified source/configuration hashes, a unique Attempt ID, and an exclusive
lease. It blocks silent fresh restart and duplicate execution. Heartbeats are
written every 250 native steps. Atomic technical checkpoints occur every 10k
native steps and may capture a partially executed strategy block; retained
scientific checkpoints and Diagnostics occur every 50k.

Same-Attempt recovery verifies the checkpoint hash and restores actor, all
critics/targets, optimizers, entropy, multipliers, replay, environment,
controller/shield memory, held target and partial block, task cursor, counters,
and Python/NumPy/Torch plus explicit generator states.

## Fail-closed conditions

Stop if source/configuration provenance changes, fallback parity fails, an
endpoint/braking/deadline fixture fails, the viability estimate is falsified,
resource gates exceed 24 hours or 12 GB, exact recovery diverges, a counter
cannot be recovered, Validation would open early, or Paper 4000–4017 would be
constructed. Do not relax the deadline, margins, exact zero tolerance, reward,
or acceptance thresholds.

Current authorization flags are intentionally:

```text
ready_for_main_training = true
main_training_authorized = false
main_training_enabled = false
validation_authorized = false
paper_authorized = false
```
