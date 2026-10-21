# Causal MPC V2

This package contains the completed authorized non-learning successor to
SlackTrader V1. Its final classification is
`MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR`: AdaptivePrice MPC V2 passed feasibility,
the 3% TIME_AWARE energy gate, executed-authority and resource gates, but its
0.801% paired saving over FixedPrice missed the preregistered 1% dynamic-price
gate. SlackTrader RL V2 is therefore not recommended or authorized.

No RL training or Validation/Paper evaluation is part of this package. V1's
`RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT` remains historical fact.

Run focused tests with:

```bash
PYTHONPATH=src pytest benchmarks/causal_mpc_v2/tests
```

Study commands are fail-closed and recorded in `canonical.json` and the result
artifacts. Capability can be opened only after Calibration has frozen method
hashes and chosen parameters.
