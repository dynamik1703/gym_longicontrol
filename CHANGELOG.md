# Changelog

All notable changes to this project are documented in this file.

## Next - Unreleased

- Add MO-Gymnasium-compatible vector rewards without changing scalar v1.
- Add optional CAPQL / SAC weight-sweep comparison with equal total budgets,
  local artifacts and per-seed fixed-reference Pareto/hypervolume reports.
- Add MORL integration tests and Python 3.10/3.13 CI; document native dependencies.

- Add an optional `examples` extra with Stable-Baselines3 and JupyterLab.
- Add a CPU SAC quickstart, a runnable introductory notebook, and a bundled
  demonstration checkpoint with provenance and recorded simulation results.
- Check example training, loading, evaluation, and notebook execution in CI.
- Keep example dependencies and weights outside the core wheel.

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
