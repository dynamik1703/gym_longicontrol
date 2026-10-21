# Causal MPC V2

This package contains the authorized non-learning successor to SlackTrader V1.
It diagnoses V1, implements a richer exact-dynamics spatial MPC, calibrates it
on a new route-clean panel and performs one independent Capability comparison.

No RL training or Validation/Paper evaluation is part of this package. V1's
`RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT` remains historical fact.

Run focused tests with:

```bash
PYTHONPATH=src pytest benchmarks/causal_mpc_v2/tests
```

Study commands are fail-closed and recorded in `canonical.json` and the result
artifacts. Capability can be opened only after Calibration has frozen method
hashes and chosen parameters.

