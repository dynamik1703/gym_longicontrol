# Capability Integrity Incident

`OPTIONS_V1_DEADLINE_BOUND_RETENTION_20261021`

All 72 Capability pairs completed exactly once and their immutable record hashes
verify. During post-run verification, the aggregate field for deadline-bound
falsifications could not be independently recomputed: the controller calculated
`completion_bound_s` online, but `TemporalDecision` did not retain that per-step
value in the pair records. Empty falsification lists therefore cannot be treated
as verified zero counts.

No raw pair record was changed and no pair was rerun. No physics, controller,
profile, safety, split or price code was changed. The aggregation fails closed:
the automatic `OPTIONS_VALID_EFFECT_UNCLEAR` label is superseded by
`SEMANTIC_OR_INTEGRITY_STOP`.

The remaining raw measurements—including completion, deadline/speed compliance,
energy, travel time, physical phase/action occupancy, options, switches,
interruptions, planner calls and resource use—remain available and are reported
descriptively. They cannot remove this stop classification.

A future study would need to add per-step `completion_bound_s` and its acceptance
flag before freezing new pair identities. This V1 study must not be rerun or
silently repaired.
