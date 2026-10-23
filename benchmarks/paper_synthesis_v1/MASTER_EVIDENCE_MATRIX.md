# Master Evidence Matrix

`master_evidence_matrix.csv` is authoritative. `PAIRED_SAME_TASKS` and `PAIRED_SAME_STREAM` rows support within-panel contrasts; `CONTEXTUAL_ONLY` rows must not be ranked as if populations and information contracts matched.

| Evidence | Methods | Basis | Routes | Result | Frozen classification | Qualification |
| --- | --- | --- | ---: | --- | --- | --- |
| sameinfo | Matched Constrained vs Same-Information Scalar | PAIRED_SAME_STREAM | 18 | 23/54 vs 0/54 RSR | CONSTRAINED_ROBUST_ADVANTAGE | not an algorithm-only ablation |
| gcsl | GCSL | PAIRED_SAME_TASKS | 18 | 0/54 RSR; 52/54 completion/deadline; 1/54 speed compliance | GCSL_ZERO | constrained reference is contextual |
| physical_reference | full-route Oracle and causal physical reference | PAIRED_SAME_TASKS | 36 | 36/36 RSR for both | PHYSICALLY_SOLVABLE_REFERENCE | Oracle privileged; neither 36/36 proves population-wide 100% |
| hybrid_v1 | Hybrid Validation V1 | NOT_COMPARABLE | NOT_RECORDED | no valid performance result | INCIDENT_ABORTED | two completed episodes lacked persisted results; one environment reset without action |
| hybrid_v2 | CAUSAL60, TIME_AWARE, H11, H29, H47 | PAIRED_SAME_TASKS | 36 | 35/36 RSR for every arm; H11/H29/H47 save 1.035/1.245/1.700% vs TIME_AWARE on 35 jointly feasible tasks | FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN | task 10200029 is now proven infeasible; raw RSR remains 35/36 |
| causal_mpc_v2 | CAUSAL60, TIME_AWARE, FixedPrice V2, AdaptivePrice V2 | PAIRED_SAME_TASKS | 18 | 18/18 each; Adaptive saves 5.422% vs TIME_AWARE and 0.801% vs Fixed | MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR | Fixed vs TIME_AWARE mixes profile-family and terminal-treatment changes |
| slacktrader | SlackTrader V1 preflight | PAIRED_SAME_TASKS | 4 | AdaptivePrice preflight 9.301% worse than TIME_AWARE | RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT | main RL training never started |
| temporal_options | FROZEN_MPC, RESET, CONTINUE, COMMIT | PAIRED_SAME_TASKS | 18 | 18/18 each; CONTINUE uses 0.562% more energy than RESET | SEMANTIC_OR_INTEGRITY_STOP | non-learning; completion bounds not retained |
| counterfactual | Counterfactual supervised predictor pilot | PAIRED_SAME_TASKS | 11 | 16/88 states have feasible sampled improvement >=0.2 Wh | INSUFFICIENT_IMPROVEMENT_OPPORTUNITY | not a completed RL method or controller evaluation |
| constrained_v2 | Constrained V2 | PAIRED_SAME_TASKS | 9 | 21/27 RSR; 27/27 completion/deadline; 21/27 speed | LEARNABLE_NOT_SEED_ROBUST | same nine routes reused across three trained seeds |
| reward_family | SB3 SAC, PPO, Binary, LLM reward | PAIRED_SAME_TASKS | 9 | 5/27, 0/27, 0/27, 1/27 RSR | MULTIPLE_NEGATIVE_OR_WEAK_RESULTS | not all reward implementations are one-factor matched |
| requirement_conditioning | Requirement-Conditioned SAC | PAIRED_SAME_TASKS | 9 | 48/135 overall; canonical 140 s subset 12/27 | RESPONSIVE_BUT_UNRELIABLE | full matrix contains distinct task requirements on shared routes |
| goal_sac | Goal SAC without/with HER | PAIRED_SAME_TASKS | 9 | 0/27 and 0/27 RSR | NO_HER_BENEFIT | virtual relabels are not physical successes |
| contrastive | Projected-goal CRL depth 4/16 | PAIRED_SAME_STREAM | 9 | 0/27 vs 1/27 RSR | NO_REPEATABLE_DEPTH_GAIN | 300k scale only |
| learned_mbrl | mixed learned-dynamics MBRL | NOT_COMPARABLE | 9 | 0/9 at 50k; no final endpoint | TERMINATED_RESOURCE_INFEASIBLE | termination was compute-based, not performance-based |
| physics_mbrl | exact one-step Physics MBRL | CONTEXTUAL_ONLY | 9 | 9/27 RSR | NO_MATERIAL_IMPROVEMENT | mixes one-step augmentation with a different real-data target |
| mve | Physics MVE vs matched real-only | PAIRED_SAME_STREAM | 9 | 0/27 vs 10/27 RSR | DEGRADATION_AND_UNSTABLE | one MVE seed crossed stability threshold |

## Interpretation boundary

The strongest algorithmically close comparison is Same-Information Scalar versus Matched Constrained on the same stream and tasks. Hybrid and MPC energy effects are paired within their own panels. Physical-reference 36/36, Hybrid 35/36, and MPC 18/18 are cross-panel results and are not a global controller ranking. The full-route Oracle is privileged contextual evidence.
