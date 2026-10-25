# Feedback Chunk RL — Diagnostic V2 research question

Can the immutable 300k CHUNK3 residual policies preserve route feasibility and
reduce signed net energy on a new 36-route Policy-R panel relative to their
matched RESIDUAL1S policies and the frozen AdaptivePrice and FixedPrice MPC V2
controllers?

The old `CHUNK_RL_INTEGRITY_STOP` remains immutable. Diagnostic V2 addresses its
mixed-runtime checkpoint incompatibility with native workers; it does not
reinterpret the stopped V1 study and is not final Validation evidence.

The frozen descriptive targets are:

- CHUNK3 versus RESIDUAL1S: no additional feasible-task failures and at least
  1% mean route-level paired energy saving.
- CHUNK3 versus AdaptivePrice: no additional feasible-task failures and at
  least 3% mean route-level paired energy saving.
- CHUNK3 versus FixedPrice is always reported.

Feasibility dominates energy. A negative result is valid.
