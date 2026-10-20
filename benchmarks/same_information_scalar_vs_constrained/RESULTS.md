# Same-Information Scalar vs Constrained — Final Validation Results

Status: **SAMEINFO_VALIDATION_V1 complete; final classification frozen.**

Exactly six frozen 300k policies were evaluated once on the 18 frozen Validation tasks (108 deterministic actor-only episodes). No training, adaptation, planning, checkpoint selection, or policy modification occurred. Paper remains sealed and uninspected.

## Primary endpoint: requirement satisfaction

| Pair | Constrained | Scalar | Δ count (C − S) |
| --- | ---: | ---: | ---: |
| 11 | 8/18 (44.4%) | 0/18 (0.0%) | +8 (+44.4 pp) |
| 29 | 7/18 (38.9%) | 0/18 (0.0%) | +7 (+38.9 pp) |
| 47 | 8/18 (44.4%) | 0/18 (0.0%) | +8 (+44.4 pp) |
| **Pooled** | **23/54 (42.6%)** | **0/54 (0.0%)** | **+23 (+42.6 pp)** |

## Secondary endpoints

| Policy | Completion | Deadline compliance | Speed compliance |
| --- | ---: | ---: | ---: |
| C11 | 17/18 (94.4%) | 9/18 (50.0%) | 15/18 (83.3%) |
| S11 | 0/18 (0.0%) | 0/18 (0.0%) | 18/18 (100.0%) |
| C29 | 14/18 (77.8%) | 9/18 (50.0%) | 15/18 (83.3%) |
| S29 | 0/18 (0.0%) | 0/18 (0.0%) | 18/18 (100.0%) |
| C47 | 13/18 (72.2%) | 8/18 (44.4%) | 16/18 (88.9%) |
| S47 | 0/18 (0.0%) | 0/18 (0.0%) | 17/18 (94.4%) |
| **Constrained pooled** | **44/54 (81.5%)** | **26/54 (48.1%)** | **46/54 (85.2%)** |
| **Scalar pooled** | **0/54 (0.0%)** | **0/54 (0.0%)** | **53/54 (98.1%)** |

Paired differences (Constrained − Scalar):

| Pair | Completion | Deadline | Speed compliance |
| --- | ---: | ---: | ---: |
| 11 | +17 (+94.4 pp) | +9 (+50.0 pp) | -3 (-16.7 pp) |
| 29 | +14 (+77.8 pp) | +9 (+50.0 pp) | -3 (-16.7 pp) |
| 47 | +13 (+72.2 pp) | +8 (+44.4 pp) | -1 (-5.6 pp) |
| **Pooled** | **+44 (+81.5 pp)** | **+26 (+48.1 pp)** | **-7 (-13.0 pp)** |

## Exclusive failure modes

| Policy | Exclusive counts | Behavior states |
| --- | --- | --- |
| C11 | `{"COMPLETED_DEADLINE_AND_SPEED_VIOLATION": 2, "COMPLETED_DEADLINE_VIOLATION": 6, "COMPLETED_SPEED_VIOLATION": 1, "FEASIBLE": 8, "INCOMPLETE_SAFE": 1}` | `{"crawling": 1, "speed_violation": 3, "standstill": 0, "timely_completion": 9}` |
| S11 | `{"INCOMPLETE_SAFE": 18}` | `{"crawling": 18, "speed_violation": 0, "standstill": 0, "timely_completion": 0}` |
| C29 | `{"COMPLETED_DEADLINE_AND_SPEED_VIOLATION": 1, "COMPLETED_DEADLINE_VIOLATION": 4, "COMPLETED_SPEED_VIOLATION": 2, "FEASIBLE": 7, "INCOMPLETE_SAFE": 4}` | `{"crawling": 4, "speed_violation": 3, "standstill": 0, "timely_completion": 9}` |
| S29 | `{"INCOMPLETE_SAFE": 18}` | `{"crawling": 18, "speed_violation": 0, "standstill": 0, "timely_completion": 0}` |
| C47 | `{"COMPLETED_DEADLINE_AND_SPEED_VIOLATION": 1, "COMPLETED_DEADLINE_VIOLATION": 4, "FEASIBLE": 8, "INCOMPLETE_SAFE": 4, "INCOMPLETE_WITH_SPEED_VIOLATION": 1}` | `{"crawling": 5, "speed_violation": 2, "standstill": 0, "timely_completion": 8}` |
| S47 | `{"INCOMPLETE_SAFE": 17, "INCOMPLETE_WITH_SPEED_VIOLATION": 1}` | `{"crawling": 18, "speed_violation": 1, "standstill": 0, "timely_completion": 0}` |
| **Constrained pooled** | `{"COMPLETED_DEADLINE_AND_SPEED_VIOLATION": 4, "COMPLETED_DEADLINE_VIOLATION": 14, "COMPLETED_SPEED_VIOLATION": 3, "FEASIBLE": 23, "INCOMPLETE_SAFE": 9, "INCOMPLETE_WITH_SPEED_VIOLATION": 1}` | `{"crawling": 10, "speed_violation": 8, "standstill": 0, "timely_completion": 26}` |
| **Scalar pooled** | `{"INCOMPLETE_SAFE": 53, "INCOMPLETE_WITH_SPEED_VIOLATION": 1}` | `{"crawling": 54, "speed_violation": 1, "standstill": 0, "timely_completion": 0}` |

## Feasible-only energy

Constrained has 23 genuinely feasible episodes: mean 0.190465 kWh, median 0.181843 kWh, range 0.151079–0.273991 kWh. The complete distribution is retained in `results.json`.

**No Scalar feasible-energy comparison is defined.** Scalar had zero feasible Validation episodes; energy from infeasible crawling episodes is not interpreted as better optimization.

## Frozen classification and interpretation

Classification: **CONSTRAINED_ROBUST_ADVANTAGE**.

In this benchmark, supplying the same energy, speed-constraint, and dense deadline-feasibility information through a strong fixed scalar reward is insufficient to match explicit constrained learning.

The earlier Constrained-V2 improvement cannot be attributed solely to access to the deadline-feasibility signal.

The result does not establish that constrained optimization alone causes the full difference: adaptive multipliers, separate reward/cost critics, FSRL critic entropy semantics, and actor-loss structure also differ.

The Scalar reward received the same raw task information and was fixed as:

```text
r_scalar = r_E - 1.4009421299572349 * c_speed
                 - 1.5752363435900516 * c_deadline
```

Scalar SAC received a reference-informed fixed scalarization derived from previously completed Constrained runs.
The lambda weights were not chosen from the present Diagnostic or Validation performance; the Scalar control was deliberately strong.

## Diagnostic-to-Validation consistency

Scalar's safe-crawling/incomplete regime reproduced exactly at the pooled level (0 feasible and 0 completed in both splits; 53 incomplete-safe plus 1 incomplete speed-violation episode). Constrained retained timely feasible completion, although Validation had 23 feasible episodes versus 28 on Diagnostic and more speed violations (8 versus 1).
Diagnostic results were not inputs to the frozen classification.

Scalar was speed-compliant in 53/54 Validation episodes but completed 0/54. All 54 were crawling/partial-progress episodes (53 incomplete-safe; one incomplete with a speed violation). This is constraint avoidance, not successful task completion under constraints.

## Provenance and integrity

- Configuration SHA-256: `96c587c9b14c0238257be60983bc9db7951fdabe2cebd867d5b9b13627e8c1f2`
- Validation ordered-set SHA-256: `d8a658e96b0090f8e0cdcdfdec3b291329045dc3dac9541e0603f34d916bf10a`
- Final endpoint: 300,000 transitions for every policy; no 50k–250k checkpoint was selected.
- Episode accounting: 6 policies × 18 tracks = 108 unique policy-track pairs; no repeats.
- All checkpoint hashes and actor hashes match the frozen artifacts; training counters stayed fixed.
- Validation authorization is closed; Paper is `SEALED_UNINSPECTED` with zero episodes.

The initial finalizer completed C11/C29/C47, then stopped before any Scalar episode on an FSRL empty-extra-state loader incompatibility. Those 54 immutable episodes were preserved. The continuation loaded the same frozen actors and evaluated only untouched S11/S29/S47. Actor hashes matched their final Diagnostic hashes and no condition was repeated.

Complete compact episode-level metrics for all 108 episodes, checkpoint and actor hashes, training counters, raw-artifact hashes, and all derived aggregates are in `results.json`.
