# Source Audit

- Frozen source commit: `b1613e62ec381932cca3fb43ae698d2f75ffaa5e`.
- Frozen source tree: `c1bc3f4eab0eb3a6d96966df25fe99a518493967`.
- Isolated branch: `research/feedback-chunk-rl-v1`.
- Isolated worktree: `/Users/romanliessner/gitlab/gym_longicontrol_feedback_chunk_rl`.
- Verified LongiControl remote: `gym_longicontrol_origin`, URL
  `https://github.com/dynamik1703/gym_longicontrol.git`.
- Unsafe local `origin`: `https://github.com/dynamik1703/trading-dashboard-workbench.git`;
  it must never receive this branch.

The audit inspected `hybrid_rl_design`, `hybrid_rl_execution`,
`hybrid_rl_validation_v2`, `causal_mpc_v2`, `temporal_options_v1`,
`counterfactual_option_value_v1`, and `paper_synthesis_v1`. Reused authorities
are the AdaptivePrice MPC V2 controller/planner, CAUSAL60 fallback, native
physical acceptance checks, corrected Hybrid SMDP targets and Revision-3
updates, execution-memory propagation, atomic checkpointing, leases, and
Policy-R split construction.

Historical results and classifications are immutable. The Hybrid Validation V1
exactly-once incident and Temporal Options `completion_bound_s` retention
incident are explicitly carried forward as integrity requirements.

