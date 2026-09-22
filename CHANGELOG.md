# Changelog

All notable changes to this project are documented in this file.

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
