# Changelog

All notable changes to this project are documented in this file.

## Next - Unreleased

- Add the first scalar SAC reward-sensitivity benchmark with a fixed task,
  disjoint training/evaluation seeds, algorithm-independent physical evaluation,
  JSON results, calibration tooling, and headless plots.
- Add a separately versioned Scalar Reward V2 hardening study with analytical
  scale checks, terminal task signals, 100k/300k validation curves, predefined
  acceptance criteria, reserved paper-test seeds, and reward-iteration
  provenance while preserving the V1 experiment unchanged.
- Add a controlled Stable-Baselines3 2.9 SAC/PPO study with a shared V2-B
  reward, exact interaction budgets, external learning curves, internal loss
  diagnostics, episode-level results, reference comparisons, and reproducible
  trajectory selection.
- Record the preregistered stop decision for agent-designed reward experiments:
  neither SB3 trainer meets the existing all-seed acceptance criterion, so no
  reward-agent development or final-test evaluation was started.
- Add a preregistered scalar credit-assignment diagnostic with a benchmark-only
  action-repeat wrapper, exact simulator/decision/update accounting, a 2 x 2
  gamma/repeat matrix, physical-time discount analysis, native-resolution
  trajectories, and a case-E stop result without touching final-test tracks.
- Add a preregistered FSRL SAC-Lagrangian benchmark with an energy-only
  objective, separate speed and completion/deadline costs, exact native-step
  accounting, multi-cost adapter tests, external physical evaluation, and a
  case-D standstill diagnosis without touching final-test tracks.
- Add a separate Constrained RL V2 study that replaces only the delayed binary
  task cost with a dense speed-limit-aware deadline-deficit integral, producing
  27/27 on-time completions and 21/27 RSR at 300k, a case-B strong but not yet
  credible constrained baseline, without touching final-test tracks.
- Add a post-hoc replay diagnosis of the six Constrained V2 speed failures,
  separating four small constant-section boundary errors from two meaningful
  late-braking failures at downward limit changes. The mixed case-F result
  triggers the preregistered stop rule, so no V2.1 training or final-test
  evaluation is performed.
- Add a separately preregistered Requirement-Conditioned SACLag study with a
  benchmark-only deadline/time observation, physics-selected balanced
  20/40/60-s training margins, zero-shot 30/50-s interpolation, exact
  episode-specific feasibility, controllability metrics, deterministic
  trajectories, and a Gate-D result (48/135 primary RSR; 12/27 at canonical
  140 s) without touching final-test tracks.
- Add a separately preregistered Binary Success Reward study using the frozen
  SB3 SAC setup and exact external feasibility semantics. Across 900k pooled
  interactions it observes 0/604 successful training episodes and finishes at
  0/27 Validation RSR (Gate D), with deterministic trajectories and diagnostics
  showing a transition from fast speed failures to speed-compliant standstill,
  without touching final-test tracks.
- Add the Phase-1 LLM Reward Engineering harness without generating or training
  a reward candidate: frozen 5+3+3 search budget, typed physical reward API,
  leakage-resistant fresh-session packages, static/runtime source checks,
  Development-only screening and reflection, task-first lexicographic ranking,
  deterministic parents, single-repair accounting, winner hash gate and a
  191-file frozen-artifact checksum manifest.
- Rename the unreleased integrated speed metric to the dimensionally correct
  `integrated_speed_violation_m` before freezing the benchmark API.
- Add immutable task requirements, reward-independent episode metrics and pure
  feasibility evaluation with explicit, inclusive floating-point boundaries.
- Expose per-step/terminal physical snapshots and contiguous speed-violation
  events through additive info keys, without changing v1 dynamics, rewards,
  observations, action semantics or termination.

## 1.0.0 - Unreleased

### Added

- Gymnasium-native `DeterministicTrack-v1` and `StochasticTrack-v1`
  environments.
- Reproducible stochastic reset behavior through `reset(seed=...)`.
- Independently testable vehicle, track, dynamics, observation, and reward
  components.
- Explicit-unit diagnostics and signed reward components in `info`.
- Optional headless-safe rendering and a modern import-safe PyTorch training
  entry point.
- Deprecated four-value adapters and optional Gym 0.23.1 v0 registration.
- Automated tests, environment validation, package-build checks, and CI.

### Changed

- Packaging now uses `pyproject.toml`, a `src` layout, explicit dependency
  groups, and complete package-data declarations.
- The runtime vehicle estimator is evaluated from portable NumPy weights.
- Episode completion distinguishes task termination from time-limit truncation.
- Rendering mode is selected when the environment is created.
- The renderer uses Matplotlib; the original Pyglet dashboard is archived.
- Corrected declared observation bounds and maximum-velocity overshoot.
- Corrected replay-buffer indexing, time-limit bootstrapping, and checkpoint
  continuation; training histories are now JSON.

### Security

- Core runtime code no longer unpickles the bundled scikit-learn estimator.

### Removed

- Legacy Gym as the primary environment API (an optional adapter remains).
- scikit-learn, pandas, PyTorch, and GUI libraries as mandatory dependencies of
  the environment package.

See [MIGRATION.md](MIGRATION.md) for concrete API updates.
