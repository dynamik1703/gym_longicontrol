# Frozen execution protocol

## Training

Only S11, S29, and S47 require new main training. Each uses:

```text
real transitions       300000
RL updates              30000
collector                TRANSITIONAL
gamma                    0.99
n-step                   2
batch                    256
replay capacity          100000
actor / critic           128 x 128
actor LR                 0.0005
critic LR                0.001
automatic alpha          true (initial 1, target entropy -1, LR 0.0003)
tau                      0.05
native dt                0.1 s
maximum episode          1800 steps
```

No action repeat, model augmentation, reward bonus, speed margin, progress shaping,
or separate tuning is permitted. S11/S29/S47 consume exactly the retained ordered
CANONICAL_11/29/47 streams respectively. Any stream recovery failure is a hard stop.

Scientific checkpoints occur at 50k intervals; 300k is always final and there is no
best-checkpoint selection. Recovery checkpoints are written atomically every 10k and
the latest recovery slot is replaced. Checkpoints contain policy, actor/critic/alpha
optimizers, log-alpha, replay, environment, collector, all RNG state, counters, source
hashes, runtime versions, Attempt ID, and walltime. Scalar checkpoints reject PID or
Lagrange fields.

The runner supplies detached launchd preparation, 250-transition heartbeat cadence,
immutable Attempt ID, and exact same-Attempt resume. A fresh restart remains
unauthorized.

## Diagnostic

At 50k/100k/150k/200k/250k/300k, every Scalar checkpoint is evaluated on the
18-task `SAMEINFO_DIAGNOSTIC_V1` with a deterministic actor only. After all Scalar
finals freeze, the same new Diagnostic is applied to the 18 retained C11/C29/C47
checkpoints. No checkpoint is selected from these curves.

## Validation

`SAMEINFO_VALIDATION_V1` remains `FROZEN_UNOPENED` until all Scalar finals and all
reference final hashes verify. A separate authorization gate then executes exactly:

| Arm | policies | episodes/policy | total |
|---|---|---:|---:|
| Constrained V2 | C11, C29, C47 | 18 | 54 |
| Same-information Scalar SAC | S11, S29, S47 | 18 | 54 |
| Total | 6 | 18 | 108 |

Primary reporting is per-seed and pooled RSR. Secondary reporting is completion,
deadline, speed compliance, exclusive failures, feasible-only energy, and training
stability. Energy is interpreted only among feasible episodes.

## Seals

- Main training authorization: false
- Main training enabled: false
- Validation authorization: false
- Paper authorization: false
- Paper content: never instantiated, hashed, inspected, compared, or evaluated

The Paper range 4000-4017 is used only as a numeric exclusion list.
