# Frozen Protocol

1. Preserve source commit `1c63ec85673ad0f3bf4afb4b05aa4f182c353ecc`
   and change only `benchmarks/slacktrader_v1/`.
2. Keep `StochasticTrack-v1`, 1,000 m, 140 s, exact zero speed tolerance,
   native 0.1 s integration and 1,800-step limit unchanged.
3. Use only the declared causal information boundary online.
4. Freeze the 100 m, nine-candidate planner, terminal continuation, tie rules,
   float64 numerics and logarithmic lambda range before any learned policy.
5. Calibrate FixedPrice and AdaptivePrice only on the first four frozen
   Development tasks; assess capability on the remaining four.
6. Stop if the price is degenerate, the planner is unsafe/unstable/too slow, or
   a strong MPC baseline and meaningful learning opportunity are not shown.
7. If and only if separately authorized after a passed preflight, use seeds
   11/29/47, 300k native transitions and 30k Revision-3 updates each.
8. Diagnostic evaluations would occur at 50k increments; 300k is always final.
9. Validation remains unopened until all three endpoints and baseline identities
   are frozen, then requires a separately authorized exactly-once transaction.
10. Never instantiate or inspect Paper seeds 4000–4017.

This preparation stopped at step 6. Steps 7–9 were not executed. Future work
requires a new explicitly preregistered planner revision or an explicit change
to the scientific question; it must not silently tune this V1 after the result.
