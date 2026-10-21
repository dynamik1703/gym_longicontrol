# Safety and Viability

Every proposed native action is checked by the frozen physical layer for exact
right-endpoint speed compliance, exact maximum-braking feasibility for known
lower limits, braking protection against an unseen 20 km/h limit at preview,
and the existing pessimistic deadline-recoverability estimate. Failure executes
the exact unchanged CAUSAL60 fallback and invalidates the current segment.

The deadline check remains an empirically supported estimate, not a theorem.
Any accepted-action counterexample or exact speed violation is an integrity
stop. A planner failure falls back and is reported; it is not evidence that the
physical task is infeasible.

