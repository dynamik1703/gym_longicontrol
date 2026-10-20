# Canonical GCSL split-scope gate

This study preserves the exact fixed canonical G11, G29, and G47 training
streams. They contain six pre-training-discovered route-and-full-task matches
with historically opened content. Those matches are permanently recorded in
`historical_overlap_ledger.json`; they do not authorize evaluation on any of
the matched historical sets.

The mandatory scientific condition is narrower and exact:
`GCSL_DIAGNOSTIC_V1` and `GCSL_VALIDATION_V1` must be route- and
full-task-disjoint from every reachable task in all three fixed canonical
training streams. Historically opened content is also excluded when choosing
both new evaluation sets. Paper seeds 4000--4017 remain numeric exclusions
only and their task content must never be instantiated or fingerprinted.

This amendment resolves the earlier mutually incompatible requirements by
separating historical disclosure from current-study train/evaluation
integrity. It changes neither a canonical training row nor any GCSL learning
semantics.
