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

## Capability outcome

All controls achieved RSR 18/18 with complete deadline and exact speed
compliance. Mean feasible energy was 0.196323 kWh (CAUSAL60), 0.180326 kWh
(TIME_AWARE), 0.172002 kWh (FixedPrice V2) and 0.170666 kWh (AdaptivePrice V2).
Adaptive saved 5.422% paired versus TIME_AWARE but only 0.801% versus FixedPrice;
the latter misses the frozen 1% dynamic-price gate.
