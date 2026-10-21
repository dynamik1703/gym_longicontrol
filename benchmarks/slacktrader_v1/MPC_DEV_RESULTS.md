# MPC Development Results

## Protocol

Eight route-clean Development tasks were selected before performance. The first
four calibrated a finite grid; the last four (`10250005`, `10250006`,
`10250008`, `10250009`) formed the independent capability partition. All four
capability tasks now have feasible witnesses; their selection-time
classification remains recorded separately as unresolved.

## Capability results

| Controller | RSR | Mean feasible energy (kWh) | Mean time (s) |
| --- | ---: | ---: | ---: |
| CAUSAL60 | 4/4 | 0.197153 | 74.600 |
| TIME_AWARE | 4/4 | 0.179355 | 126.250 |
| FixedPrice MPC | 4/4 | 0.195834 | 74.725 |
| AdaptivePrice MPC | 4/4 | 0.196288 | 75.925 |
| Privileged reference | 4/4 | 0.149578 | 125.200 |

Paired arithmetic mean task-level savings for AdaptivePrice were +0.453% versus
CAUSAL60, -0.225% versus FixedPrice, -9.301% versus TIME_AWARE, and -31.228%
versus the contextual privileged reference.

There were no completion, deadline or speed failures and no planner feasibility
failures. AdaptivePrice recorded 20 deadline-shield overrides across 3,037
native proposals. Planner solve time was 10.145 s over 355 solves; the weighted
mean was 28.579 ms.

## Price sensitivity

Every capability track exposed at least one identical-state counterfactual in
which low and high price selected different durations and first native actions.
Observed marginal transition prices were 0.001625–0.007257 kWh/s, and duration
never increased with price.

## Decision

The price action is non-degenerate, but the MPC does not add credible energy
authority beyond TIME_AWARE and is mostly fallback-equivalent. The mandatory
outcome is `RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT`; no main RL run is justified.
