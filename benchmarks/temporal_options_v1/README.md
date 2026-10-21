# Temporal Options V1

This package tests whether preserving progress through the inherited 17 physical
driving profiles changes phase realization and closed-loop performance. It is a
non-learning architecture experiment: no actor, critic, world model, demonstration
or gradient update is present.

The four frozen arms are `FROZEN_MPC_V2`, `RESET_FEEDBACK`,
`CONTINUE_FEEDBACK` (primary treatment) and `COMMIT_FEEDBACK`. The new arms reuse
the production dynamics, power model, AdaptivePrice law, terminal heuristic,
physical fallback, Shield checks and profile parameters from commit
`c7c6175b22e373f476aa60c10a755cb7072d9d5f`.

Primary entry points:

```text
python -m benchmarks.temporal_options_v1.semantics_audit
python -m benchmarks.temporal_options_v1.evaluation prepare
python -m benchmarks.temporal_options_v1.evaluation calibration
python -m benchmarks.temporal_options_v1.evaluation capability
python -m benchmarks.temporal_options_v1.report
```

Capability is exactly four controllers × 18 fresh tasks. Historical Validation
panels are not used; Paper 4000–4017 is rejected numerically before construction.

