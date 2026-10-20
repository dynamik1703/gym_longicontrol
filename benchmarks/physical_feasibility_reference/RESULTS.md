# Results

Status: **PHYSICALLY_SOLVABLE_REFERENCE / CAUSAL_REFERENCE_STRONG on the tested
tasks**.

## Independent holdout

| Metric | Full-route Oracle | Observation-limited |
| --- | ---: | ---: |
| Feasible witness / RSR | 36/36 (100%) | 36/36 (100%) |
| Completion | 36/36 | 36/36 |
| Deadline compliance | 36/36 | 36/36 |
| Exact speed compliance | 36/36 | 36/36 |
| Mean travel time | 131.0139 s | 91.5417 s |
| Feasible-only mean energy | 0.1526616 kWh | 0.2058568 kWh |
| Feasible-only median energy | 0.1495775 kWh | 0.2108881 kWh |
| Mean solve time | 1.8553 s | 0.1242 s |
| Maximum solve time | 3.6655 s | 0.2058 s |

Oracle classifications: 36 `FEASIBLE_WITNESS`, 0 `PROVEN_INFEASIBLE`,
0 `UNRESOLVED`. All 36 witnesses have exact independent replay parity. The
Oracle's slower travel is expected: once feasibility was established, the
lexicographic second stage lowered energy while retaining the 140 s deadline.

Every paired task has zero Oracle-minus-causal difference for RSR, completion,
deadline and speed compliance. Therefore `ORACLE_GAP_LARGE` is not observed on
this panel. The result does not prove that no generated route can expose a
preview limitation.

## Development

Development was also 18/18 for both references in all four requirement metrics.
Oracle feasible-only energy was mean/median 0.1515841/0.1495775 kWh; causal was
0.1995699/0.1966885 kWh. All 18 Oracle witnesses replayed exactly.

Across both panels, the study stores 54 feasible Oracle witnesses, zero proven
infeasible cases and zero unresolved cases.

## Interpretation

The benchmark is physically solvable on every tested independent holdout task
under the original actions, dynamics, termination and exact zero-tolerance
speed metric. This is direct achieved evidence, not a theoretical maximum or a
claim about all generator outputs.

The causal reference also solves every tested holdout task without full-route
access. Its only inputs are the ordinary 8D sensor, exact known dynamics, fixed
task parameters and disclosed internally propagated state/history. Thus hidden
future route information is not necessary for reliable control on this panel.
Partial observability may still matter to a memoryless actor or on other tracks,
but it is not supported here as the sole explanation for prior RL failures.

Existing immutable context reports Same-information Scalar SAC 0/54 and GCSL
0/54 (despite GCSL completion 52/54). These are not task-paired comparisons to
the independent physical holdout. The optional post-hoc physics evaluation on
those validation tracks was deliberately omitted; no causal attribution is made
from cross-panel rates.

## Recommended next architecture

The next separately authorized RL study should test a recurrent constrained
actor with:

- an explicit internal elapsed-time/progress state estimator;
- physics-derived braking-envelope features for the visible transitions;
- an action parameterization that respects next-endpoint speed semantics;
- a lexicographic feasibility-first objective, with energy optimized only inside
  the feasible action region.

This targets the capability demonstrated by the causal reference without
feeding Oracle trajectories or hidden routes into learning. No such RL run was
started here.

## Integrity

- Development split hash:
  `4551ddfa2f4363aff35323a2ed5994631f5a26bb50948af66f9d78b7558d3d1f`
- Holdout split hash:
  `c02c4889333eaf3526277b10bc5c33101ac1aa97f9de836379088c661e72a30b`
- Scientific configuration hash:
  `f63717504a810c75d7080388aeb159505cf667ad13248dbe4a53ec0c4bb94cd1`
- Paper `4000–4017`: `SEALED_UNINSPECTED`, zero content fingerprints.
- Optional contextual analysis: not performed.
