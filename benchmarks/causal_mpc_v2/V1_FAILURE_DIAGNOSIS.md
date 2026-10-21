# Verified V1 Failure Diagnosis

Source: immutable `benchmarks/slacktrader_v1/mpc_development_results.json` plus
deterministic exact-dynamics fixtures. The historical outcome remains
`RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT`.

## Registered diagnostics

- Maximum-target selections: 322/707
  (45.545%).
- Terminal acceleration contribution across recorded sensitivity selections:
  0.000000000 kWh total; 0/
  12 nonzero. These recorded selected
  endpoints were already at the imposed reference, so zero here does not show
  that the formulation was inactive for discarded candidates.
- Recorded selected candidate time: 6.300–
  7.200 s (median 7.100 s).
- All feasible fixture-candidate times: 6.100–
  19.200 s (median 9.800 s), with
  36 distinct candidate
  action sequences.
- Across all fixture candidates, the V1 terminal-acceleration term was nonzero
  for 96/108, mean
  0.035884 kWh and maximum
  0.062684 kWh.
- Lambda changed selected duration on all
  4 recorded causal states;
  selected duration was non-increasing throughout.
- Executed physical action hashes:
  6
  distinct sequences across 8 MPC
  episodes.
- Replanning: 707 solves over
  6026 native proposals.
- Planner proposal identical to fallback: 5655/
  6026 (93.843%).
- Shield overrides: 40/6026
  (0.664%).
- Removing V1 terminal energy changed the first action in
  5/12 deterministic fixtures.

## Frozen mechanism conclusion

H1 is supported by construction: nine constant-target candidates exclude broad
acceleration/coast/brake sequencing. H2 is mechanistically active in the
controlled fixture even though the recorded *selected* sensitivity endpoints
incurred zero terminal charge. H3 is strongly supported by fallback identity;
Shield overrides add further loss of executed authority. These are mechanism
findings on opened V1 data, not V2 generalization evidence.
