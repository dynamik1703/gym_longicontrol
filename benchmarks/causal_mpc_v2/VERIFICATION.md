# Verification

Final verification on 2026-10-21:

- `ruff check --no-cache benchmarks/causal_mpc_v2`: pass.
- Relevant production/V1/V2 test set: 157 passed, 2 optional matplotlib
  rendering tests skipped.
- `git diff --check`: pass.
- Frozen method hashes after Capability: pass.
- Capability exactly-once ledger: 72 unique pairs, 18 per controller, pass.
- Paper fail-closed state and zero Paper episodes: pass.
- Zero RL training episodes/gradient updates and zero Validation episodes: pass.

The focused tests cover production-physics replay parity, candidate feasibility,
terminal units/bias behavior, fixed-distance progress, lambda monotonicity,
distinct price/action response, exact fallback on planner failure, causal API,
right-endpoint compliance, deterministic solves, split cleanliness and final
scientific-gate integrity. Capability adds 18/18 exact speed compliance and no
deadline-bound falsifications for each controller.

## Unrelated historical/environment findings

Repo-wide Ruff reaches one pre-existing error outside this study:
`benchmarks/llm_reward/generations/g0/g0-c03.py:6` assigns unused
`budget_scale`. It was not modified.

An unrestricted full-suite collection in the available environment stops on
four unrelated optional-dependency imports: `stable_baselines3` (three test
modules) and `tianshou` (one). Six other optional suites skip because
`matplotlib`, `jax` or `fsrl` is absent. The relevant suite above avoids those
dependency-only collection failures; no Causal MPC V2 test is skipped.

