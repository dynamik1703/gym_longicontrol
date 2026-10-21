# Baseline Comparison

The Capability comparison is preregistered as:

| Control | Physical controller | Price rule | Status |
| --- | --- | --- | --- |
| CAUSAL60 | original causal controller | none | unchanged |
| TIME_AWARE | original target-speed controller and Shield | deterministic pace | unchanged |
| FixedPrice MPC V2 | V2 optimizer, terminal heuristic and Shield | calibrated constant | new |
| AdaptivePrice MPC V2 | identical V2 stack | calibrated reserve feedback | new |

Fixed and Adaptive differ only in scalar price selection. The privileged
full-route reference is preregistered as omitted: it is neither
information-matched nor a certified optimum, and omitting it reduces compute
without affecting any mandatory gate.

