# Integrity Design

Every native record retains the causal state, observation, known transitions,
high-level and active option identities, proposed/executed/fallback actions,
acceptance decision, intervention reason, all four physical checks,
`completion_bound_s`, actual successor and signed energy.

An independent verifier replays production dynamics and recomputes the inherited
endpoint, known-transition, unknown-preview and pessimistic-completion tests.
Deadline-bound falsification means any mismatch between the retained bound/flag
and this fresh computation on the proposed-action successor. An empty list is
certifiable only after a nonempty complete contiguous trace passes replay.

This verifies use of the inherited heuristic. It does not turn the Shield or
completion bound into a mathematical global viability guarantee.

