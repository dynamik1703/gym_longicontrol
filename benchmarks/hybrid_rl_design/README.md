# Feasibility-First Hybrid RL Design

Status: **prepared; training is not authorized and has not started**.

This package defines a controlled follow-up in which a learned strategy chooses
an energy-oriented target speed while the causal physics controller retains
authority over right-endpoint speed compliance, known-transition braking,
unknown-preview braking and deadline recoverability. It is explicitly a Hybrid
RL system, not end-to-end RL.

The selected interface is a scalar target-speed/cruise cap in 20–60 km/h. A
two-stage causal API exposes the ordinary 8D observation plus eight disclosed
controller-memory features to the future policy. The policy never receives the
environment, `Track`, privileged simulator state, future route, Oracle actions,
or future realized states.

## Preparation result

- The frozen physical fallback source and configuration hashes match.
- With strategy influence disabled, six deterministic fixtures reproduce the
  frozen causal controller exactly in actions, state traces, completion,
  deadline, speed and energy.
- Active shield fixtures have zero speed/deadline/feasibility failures and no
  falsified deadline bound.
- The fixture strategy changes 63.4% of executed native actions; only 17.6% of
  proposals are overridden. `SHIELD_DOMINATES_POLICY` is false.
- Route-clean Policy R excludes every new evaluation route/full task from all
  2,595 reachable canonical training rows and prior protected content.
- `HYBRID_RL_DIAGNOSTIC_V1` has 18 frozen tasks;
  `HYBRID_RL_VALIDATION_V1` has 36 frozen, unopened tasks.
- Projected future runtime is about 0.52 h after a conservative 3× factor, and
  retained disk is about 2.21 GB. Both hard preparation limits pass.
- Real training transitions: **0**. RL gradient updates: **0**. Hybrid final
  Validation performance episodes: **0**. Paper remains sealed and uninspected.

`preparation_status.json` is the machine-readable readiness authority. Its
`ready_for_future_separately_authorized_training` flag does not authorize a run.

## Files

- `SOURCE_AUDIT.md`: simulator, fallback and information-boundary audit.
- `DESIGN.md`: option comparison and selected target-speed architecture.
- `SAFETY_AND_VIABILITY.md`: acceptance rule, deadline test and limitations.
- `BASELINE_COMPARISON.md`: frozen causal, non-learning and Hybrid arms.
- `PROTOCOL.md`: proposed future training/evaluation preregistration.
- `RESOURCE_REPORT.md`: bounded timing, physics-call, memory and disk estimates.
- `canonical.json`: exact proposed algorithm and study constants.
- `split_ledger.json` / `SPLIT_LEDGER.md`: task streams and Policy-R proof.
- `preflight_results.json`: parity, safety, deadline and influence fixtures.
- `resource_measurements.json`: bounded resource measurements.
- `preparation_status.json`: final preparation gates and zero-training record.

Safe regeneration (no RL training and no Hybrid final evaluation):

```bash
PYTHONPATH=src:. .venv/bin/python -m benchmarks.hybrid_rl_design.prepare prepare
PYTHONPATH=src:. .venv/bin/python -m benchmarks.hybrid_rl_design.prepare verify
```

Neither command implements or starts a main-training loop.
