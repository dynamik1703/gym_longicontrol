# Resource report

Hard gate: **PASS**.

The actual one-critic Scalar FSRL/Tianshou path ran for 2,048 preparation-only
transitions and 204 updates on route-clean fixture seeds 0 and 1. No policy was
evaluated and fixture outcomes were discarded.

Total simulator transitions consumed during preparation were 2,061: the 2,048 pilot
transitions plus 13 package/regression smoke transitions. The latter were fixed
verification fixtures, not throughput measurements or policy evaluations, and no
outcome entered the design.

- Walltime: 3.497 s
- Throughput including updates: 585.620 transitions/s
- Update throughput: 164.888 updates/s
- Pilot checkpoint: 55,773,429 bytes; 0.204 s write; 0.062 s load
- Projected one 300k Scalar policy: 512.278 s
- Projected all three Scalar policies: 1,536.834 s
- Planned new Diagnostic episodes: 648 (both arms, six checkpoints)
- Planned final Validation episodes: 108
- Projected evaluation time: 263.519 s
- Conservative 90-write checkpoint overhead: 1,419.985 s
- Projected total new main-study walltime: **0.895 h** (limit 12 h)
- Projected retained new artifacts: **3.177 GB** (limit 12 GB)

Evaluation timing uses the completed collector-replication preparation measurement of
0.348570 s/episode because the deterministic actor/evaluator path is identical and this
preparation task forbids new policy evaluation. Checkpoint projection uses the larger
151,275,789-byte / 15.7776-second archived full-checkpoint bound, not the favorable
pilot size alone.
