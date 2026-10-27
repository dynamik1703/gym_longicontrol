# Engineering tests before scientific freeze

Run 01: 87 passed, including all three formerly failing treatment recovery/native
audit tests, multi-step/episode/preview/boundary/track-corruption fixtures.
Run 02: 118 passed in 779.58s wall, adding actual warm-up-boundary transaction
crash tests for all three treatments. Exact XML and run-02 source hashes are in
engineering/. No V1 test suite was rerun.

Subsequent engineering changes: compressed full checkpoints (representation only),
real-process crash coverage, completed scientific control plane, actual replay-slot
audits, frozen Diagnostic references, resource model and report/sealing code.
Their final engineering result is recorded in engineering_results.json before any
scientific freeze. Ordinary integration errors may be corrected here without
reclassifying a scientific Attempt; every frozen failure would instead stop.

Further pre-training review repairs: actual route identity checks are isolated
in the independent audit layer, no action may prepare/reset after native budget
exhaustion, stale execution ownership is recoverable only for a provably dead
owner, interrupted final seals reuse the verified immutable endpoint, and prior
engineering/probe/sealed-preflight costs are charged to the active budget.
GRID_TIE_INDEX_EXACT_V2 fixes the proven .2-between-.1/.3 floating tie anomaly
using integer grid-index distance and tests all nineteen interior adjacent ties.
No learner or reward tuning occurred.
