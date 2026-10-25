# Diagnostic V2 results

All 468 preregistered pairs completed exactly once and passed independent
native-evidence verification. Raw feasibility is reported first. Every
controller completed all 36 routes, was speed-compliant on all 36, and met the
deadline on 35. Thus every controller has raw RSR 35/36 = 97.222%. Task
11,400,026 completed in 170.2 s for every controller and is the single common
deadline failure; it was not excluded.

| Controller | Feasible | Mean feasible energy (kWh) |
|---|---:|---:|
| CAUSAL60 | 35/36 | 0.202759 |
| TIME_AWARE | 35/36 | 0.180850 |
| FIXED_PRICE_MPC_V2 | 35/36 | 0.172487 |
| ADAPTIVE_PRICE_MPC_V2 | 35/36 | 0.171201 |
| HYBRID_H11 | 35/36 | 0.179578 |
| HYBRID_H29 | 35/36 | 0.178997 |
| HYBRID_H47 | 35/36 | 0.178929 |
| CHUNK3_S11_300K | 35/36 | 0.173717 |
| CHUNK3_S29_300K | 35/36 | 0.172647 |
| CHUNK3_S47_300K | 35/36 | 0.172699 |
| RESIDUAL1S_S11_300K | 35/36 | 0.173648 |
| RESIDUAL1S_S29_300K | 35/36 | 0.172963 |
| RESIDUAL1S_S47_300K | 35/36 | 0.174096 |

Across 4,337 CHUNK3 chunks, 1,745 (40.235%) were full and 2,499
(57.620%) were cancelled; the remainder are non-cancelled terminal partial
chunks. CHUNK3 residuals influenced 59,430/128,511 native proposals (46.245%)
and 66,425/128,511 proposals (51.688%) were overridden/cancelled. Native
evidence contains 555,498 steps. All 523,037 applicable records retain every
safety/viability field; the 32,461 CAUSAL60 records mark those computations
explicitly `NOT_APPLICABLE`.

`results.json` is sealed as `COMPLETED_INDEPENDENTLY_VERIFIED`, with result
content hash
`95cbc308f435eb18c6606301ff69778aa1ec8776f2bebd98acaa36d652a0e246`.
