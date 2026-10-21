# Baseline Comparison

## Primary information-matched arms

- FixedPrice MPC: one route-independent lambda selected by a predeclared
  five-point Development grid. Frozen action: `1.0`, lambda `0.01 kWh/s`.
- AdaptivePrice MPC: a deterministic monotone law on the causal completion
  reserve. Frozen thresholds: 24 s maps to the low endpoint and 2 s to the high
  endpoint, with linear interpolation in normalized-action space.
- SlackTrader: the same controller with normalized price supplied by SAC.

All three use the identical 16D causal contract, planner, shield, fallback,
native action and 1 Hz update frequency.

## Contextual controls

CAUSAL60 and TIME_AWARE are unchanged frozen controllers. The full-route
physics reference is privileged, contextual and not a global optimum.

## Preflight conclusion

AdaptivePrice MPC maintained 4/4 feasibility and saved 0.453% versus CAUSAL60,
but lost 9.301% versus TIME_AWARE and 0.225% versus FixedPrice MPC on paired
feasible tracks. Only 261 of 3,037 native proposals (8.594%) differed from the
fallback after shielding. It is therefore neither the strong incremental
baseline nor the control-authority foundation required for the planned RL
claim. Baseline identities are frozen for reproducibility, but
`strong_nonlearning_baselines_frozen` is false in the readiness sense.
