# Frozen Protocol

1. Source is SlackTrader V1 final commit
   `303fc16c95845579ad461873af790204b5e933a4`.
2. Preserve `StochasticTrack-v1`, 1,000 m, 140 s, zero speed tolerance, 0.1 s
   native integration, 1,800 steps, the BMW-i3 dynamics/power model, task and
   metric implementations, CAUSAL60, TIME_AWARE and the existing Shield.
3. Use a 100 m spatial horizon and the deterministic, fixed V2 profile lattice
   described in `DESIGN.md`. Candidate generation is independent of lambda.
4. Use the continuation heuristic in `TERMINAL_VALUE_AUDIT.md`; it is not a
   bound or an optimal value function.
5. Calibration has 12 route-clean tasks. The finite budget is five fixed-price
   values and three adaptive reserve laws. Priority is RSR, feasible energy,
   then computation time. Mechanistic fixtures do not spend panel episodes.
6. Freeze all method files and selected parameters before Capability.
7. Capability has 18 independently selected route-clean tasks and is opened
   once. Each of CAUSAL60, TIME_AWARE, FixedPrice V2 and AdaptivePrice V2 is run
   exactly once per task. The privileged full-route reference is not executed.
8. Gates frozen before Capability: Adaptive RSR at least TIME_AWARE RSR; mean
   paired saving versus TIME_AWARE at least 3%; mean paired saving versus Fixed
   at least 1%; Calibration executed-action intervention rate at least 2% and
   counterfactual distinct-first-action rate at least 10%; projected complete
   three-seed RL study at most 24 h and 12 GB.
9. Any hard physical-invariant or provenance failure produces
   `SAFETY_OR_INTEGRITY_STOP`. Solver failure is not physical infeasibility.
10. No RL training, no Validation, and no Paper evaluation are authorized.

The acceptance thresholds and decision labels cannot change after Capability
is opened.

