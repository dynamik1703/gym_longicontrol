# Same-Information Scalar SAC vs Constrained V2

Preparation status: **science and execution prepared; all main execution remains
unauthorized and disabled**.

This study asks whether fixed scalar SAC can match explicit Constrained V2 when both
receive the same objective, speed-cost, and dense deadline-feasibility information on
every transition. It uses the frozen execution convention:

```text
TRANSITIONAL collector
+ explicit canonical reset-seed streams
+ Route-clean Policy R
+ deterministic actor-only evaluation
+ frozen 300k endpoint
+ seeds 11 / 29 / 47
```

`TRANSITIONAL` is an execution convention, not a demonstrated superior collector.
The existing immutable T11/T29/T47 Constrained policies are complete and reusable;
only S11/S29/S47 require future main training.

Key frozen values:

- `lambda_speed_star = 1.4009421299572349`
- `lambda_deadline_star = 1.5752363435900516`
- `w_reward = 0.2514977651664276`
- `w_speed = 0.35233381481173953`
- `w_deadline = 0.39616842002183283`
- Diagnostic: `SAMEINFO_DIAGNOSTIC_V1`
- Validation: `SAMEINFO_VALIDATION_V1` (`FROZEN_UNOPENED`)
- Paper: `SEALED_UNINSPECTED`

The scalar runner is deliberately fail-closed. `run`, `resume`, and `matrix` reject
execution while `main_training_authorized` or `main_training_enabled` is false.
Validation is separately gated. Fresh restart is never automatic; only exact resume of
the same Attempt is represented by the recovery protocol.

See [SOURCE_AUDIT.md](SOURCE_AUDIT.md), [DESIGN.md](DESIGN.md),
[SCALARIZATION.md](SCALARIZATION.md), [PROTOCOL.md](PROTOCOL.md),
[SPLIT_LEDGER.md](SPLIT_LEDGER.md), and [RESOURCE_REPORT.md](RESOURCE_REPORT.md).
