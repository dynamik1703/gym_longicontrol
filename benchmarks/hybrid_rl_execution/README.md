# Executable Hybrid RL V1

This directory prepares the frozen target-speed Hybrid design for an exact,
recoverable 1 Hz SMDP study while preserving the unchanged 10 Hz physical
controller and shield from `benchmarks/hybrid_rl_design/`.

Protocol Revision 3 replaces the former warm-up credit burst with the frozen
post-warm-up cumulative update schedule in `TEMPORAL_SEMANTICS.md`. No main
training or reserved evaluation was run for this amendment.

Main training is deliberately fail-closed. `canonical.json` records readiness
but keeps `main_training_authorized=false` and `main_training_enabled=false`.
The prepared launchd file is not loaded by preparation.

## Modules

- `macro_transition.py`: held-action aggregation and analytical Bellman targets.
- `agent.py`: constrained SAC with automatic entropy and separate physical
  cost/intervention critics.
- `replay.py`: decision-boundary replay containing proposed target actions.
- `runner.py`: native-step state machine, exact budgets, stream cursor, and
  partial-block state.
- `checkpointing.py`: atomic, hash/provenance-bound exact recovery.
- `evaluation.py`: actor-only Diagnostic and fail-closed Validation/Paper paths.
- `execution.py`: leases, heartbeat, technical/scientific checkpoints, launchd.
- `safety.py`, `resource.py`, `verification.py`: preparation-only gates using
  unreserved deterministic physical fixtures.

## Preparation verification

```bash
python -m benchmarks.hybrid_rl_execution.prepare verify
pytest benchmarks/hybrid_rl_execution/tests
ruff check benchmarks/hybrid_rl_execution tests/benchmark/test_hybrid_rl_execution.py
```

Preparing a detached job writes a plist and metadata but does not load it:

```bash
python -m benchmarks.hybrid_rl_execution.execution prepare-launchd
```

Calling the `matrix` command currently raises `PermissionError`. A separate,
explicitly authorized protocol revision is required before any 300k run.
