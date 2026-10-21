# SlackTrader V1

Status: **scientifically stopped before main RL training**.

SlackTrader tests whether a learned causal shadow price of time can improve
energy efficiency when the downstream physical optimizer, native actuator,
safety shield, observation contract and update frequency are held fixed. The
price action is physically effective, and bounded learner/recovery primitives
pass their tests. The mandatory MPC preflight nevertheless failed the
scientific go/no-go gate: on the independent four-track Development capability
partition, AdaptivePrice MPC was 9.301% worse than the existing TIME_AWARE
controller and changed the fallback's native action on only 8.594% of steps.

Therefore:

- `RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT` is recorded;
- `ready_for_main_training = false`;
- `training_infrastructure_ready = false` because the protocol stops before a
  canonical main-training launcher is finalized;
- main training remains unauthorized and disabled;
- Validation is frozen and has zero performance episodes;
- Paper seeds 4000–4017 remain `SEALED_UNINSPECTED`.

The machine-readable evidence is in `mpc_development_results.json`,
`resource_measurements.json`, `split_ledger.json`, and
`preparation_status.json`. No prior study implementation or result was changed.

Useful bounded commands:

```text
PYTHONPATH=src <python> -m pytest -q -p no:cacheprovider benchmarks/slacktrader_v1/tests
<ruff> check benchmarks/slacktrader_v1
git diff --check
```

Do not launch a training runner. `execution.require_main_training_authorized()`
fails closed under the frozen configuration.
