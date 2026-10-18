# Provenance

## Repository lineage

| Item | Historical Constrained V2 | Current Matched Real-Only | Identical? | Evidence |
|---|---|---|---|---|
| Run-producing commit | `UNKNOWN` | `7dc95d9d6d4aee3d6a4183387eb53d9fd8287b75` authorized the frozen runner | Unknown | Historical run config omitted a source commit; current checkpoint records source hashes |
| Branch | `UNKNOWN` | `research/multistep-physics-mbrl` | No | Historical branch was not retained in the artifacts |
| Earliest commit containing historical code and result | `56d0bc94341ffcfda9cee29596489977be8b82ea` | n/a | n/a | `research/task-specification-benchmarks`; commit postdates raw run files |
| Final result commit | historical result first archived at `56d0bc9` | `ebb4402f1bf5f457b265533a0ef95707c51f2a9b` | No | Local complete Git history |
| Audit base | n/a | `ebb4402f1bf5f457b265533a0ef95707c51f2a9b` | n/a | Branch `research/constrained-v2-repro-audit` |
| Worktree | retained artifacts read-only | `/Users/romanliessner/gitlab/gym_longicontrol_repro_audit` | n/a | Isolated audit worktree |

The exact historical run-producing commit and branch are therefore `UNKNOWN`. The
earliest archived snapshot contains the runner and result together, and every shared
scientific file inspected there is byte-identical to the audit-start version, but
there is no cryptographic link from the raw checkpoint to that Git commit.

## Scientific source integrity

Fifteen shared files covering the constrained adapter, V2 runner/config/costs,
external evaluation, dynamics, metrics, observation, task, track, vehicle, and the
environment were hashed at `56d0bc9` and at `ebb4402`. All 15 hashes match. The full
mapping is in `results.json` under
`provenance.shared_source_comparison.files`.

This proves no drift between those two repository snapshots. It does not prove that
the unrecorded historical execution used `56d0bc9`, although the archived code,
configuration hash, checkpoint metadata, and result format are mutually consistent.

## Historical retained artifacts

Raw root:
`/Users/romanliessner/gitlab/gym_longicontrol_task/runs/constrained-rl-v2-20261002`

| Seed | Run config SHA-256 | Final policy SHA-256 | Final Validation result SHA-256 |
|---:|---|---|---|
| 11 | `3d5fa5231fbf2725debd722270ea60d815b60c7d56ef22d991fcdd082b21e800` | `0fd0984dc49e26d159e1a251b84d400eb420285676fe6978365740796668dfc5` | `54d57a60c010ec44f87eb97210544a2730499b046c127176eb17e7d320e3c8f3` |
| 29 | `7b69c9c74936c09fb1d329c41f6f61c2de7661557fa893490fc206f79f92e0fd` | `97c42e52044f8313a05542a655fced6f47a50d0f9b5465452765b39a42fe20fa` | `074ff614fc42ca8c99b4d7c2b428ce16c3401a05003769f5907f440b5909e830` |
| 47 | `a900e7996af6c49759705f8f529bf0f8bba7f85ba6d1e5122691b0d9aa799005` | `98b67166bd95b309330b96d32184fcb7c8b2c07c33fec2d2f4e8c3bc1062bbf1` | `6e3be0279781b77c79ab050684b6738664a4b95a76db28001cd857c29b3a8c5d` |

The historical checkpoint contains only `policy_state_dict`, two Lagrangian states,
and metadata. It does not contain the replay buffer, environment state, RNG states,
runtime lock, or initial checkpoint. Its metadata records the configuration hash,
training seed, requested/actual simulator steps, and gradient-update count.

## Current retained artifacts

Raw root:
`/Users/romanliessner/gitlab/gym_longicontrol_multistep_mbrl/runs/multistep-physics-mbrl-v1`

| Seed | Final checkpoint SHA-256 | Final Validation result SHA-256 | Attempt |
|---:|---|---|---|
| 11 | `994568f614c9d2428b805fafafc9526717fa2d6e4ca73783adfcb361feb4090d` | `aa337605b8e7f870194eb04dec1d7c7f4256c0b18947776bc3568e5b9dcdbcb1` | one, no resume |
| 29 | `f32757afc979166793ba2f3725df7511af728dd446de1b2ded4d482a3034f5e8` | `9bcff44394907bf790a3e306834a2664977e1b0207137cdc664c59dcde9068f9` | one, no resume |
| 47 | `e0262a618f0435f85b792286cdbd9aebb3a6958e01a88ae101724ed8c98cade6` | `2eff7560402302d8f7d537009479261f5d5c1b43afd0ce9501d90ceff4669ec8` | one, no resume |

Current checkpoints retain policy and optimizer state, replay, environment and
collector state, RNG states, exact reset seeds, counters, source hashes, runtime
versions, diagnostics, and PID event counts. Each Real-Only policy has exactly
300,000 real transitions, 30,000 RL updates, and zero synthetic/model-target
transitions.

## Dependency/runtime table

| Component | Historical | Current | Identical? |
|---|---|---|---|
| Python | 3.11.13 | 3.11.13 | Yes |
| FSRL | 0.1.0, pinned `e056fc9498d5d037869533da7cf976acf462f918` | same | Yes by recorded version/revision |
| Tianshou | 0.5.1 | 0.5.1 | Yes by version |
| NumPy | `UNKNOWN` | 2.4.6 | Unknown |
| Torch | `UNKNOWN` | 2.14.1 | Unknown |
| Gymnasium | `UNKNOWN` | 1.4.0 | Unknown |
| Package source | earliest archive matches current shared files | audit-start source | Yes for inspected files |
| OS/architecture | `macOS-27.0-arm64-arm-64bit` | `macOS-27.0.1-arm64-arm-64bit` | No, patch-level drift |
| Device | CPU | CPU | Yes |
| Torch threads | 2 | 4 | No |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | unset / unset | `UNKNOWN` | Unknown |

The current installed dependency source-file hashes are retained in `results.json`.
They characterize the audit/current runtime, not the missing historical wheel bytes.
The historical FSRL revision and Tianshou version are pinned, but NumPy, Torch, and
Gymnasium behavior cannot be reconstructed exactly from the retained provenance.
