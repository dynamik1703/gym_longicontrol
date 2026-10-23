# Method Registry

`method_registry.json` is authoritative. Rows group training seeds only when the scientific configuration is identical; checkpoints are not counted as independent seeds.

| ID | Method/version | Family | Seeds | Training transitions | Panel | RSR | Classification |
| --- | --- | --- | --- | ---: | --- | ---: | --- |
| scalar_custom_grid | Historical custom Scalar SAC reward grid (140-second factorial) | A_STANDARD_REWARD_RL | [11,29,47] | 2640000 | historical holdout 1000-1008 | 5/216 | ALMOST_ALL_REWARD_SETTINGS_FAIL |
| sb3_sac_v2b | Scalar SB3 SAC (V2-B Balanced) | A_STANDARD_REWARD_RL | [11,29,47] | 900000 | Validation 3000-3008 | 5/27 | WEAK_SEED_DEPENDENT_SCALAR_BASELINE |
| sb3_ppo_v2b | Scalar SB3 PPO (V2-B Balanced) | A_STANDARD_REWARD_RL | [11,29,47] | 900000 | Validation 3000-3008 | 0/27 | ZERO_FULL_BRAKING |
| scalar_action_repeat_5 | Scalar SAC action-repeat (credit condition C) | A_STANDARD_REWARD_RL | [11,29,47] | 900000 | Validation 3000-3008 | 9/27 | NO_MATERIAL_ROBUSTNESS_IMPROVEMENT |
| binary_success_sac | Binary Success Reward SAC (V1) | A_STANDARD_REWARD_RL | [11,29,47] | 900000 | Validation 3000-3008 | 0/27 | GATE_D_NO_ESCAPE_FROM_ZERO |
| llm_reward_sac | Frozen LLM reward winner (g1-c02) | A_STANDARD_REWARD_RL | [11,29,47] | 900000 | Validation 3000-3008 | 1/27 | SPARSE_NONROBUST_SUCCESS |
| constrained_v1 | Explicit constrained RL (V1) | B_EXPLICIT_CONSTRAINTS | [11,29,47] | 900000 | Validation 3000-3008 | 0/27 | CONSTRAINTS_SATISFIED_BY_STANDSTILL |
| constrained_v2 | Dense-deadline constrained RL (V2) | B_EXPLICIT_CONSTRAINTS | [11,29,47] | 900000 | Validation 3000-3008 | 21/27 | LEARNABLE_NOT_SEED_ROBUST |
| sameinfo_scalar | Same-Information Scalar SAC (V1) | B_EXPLICIT_CONSTRAINTS | [11,29,47] | 900000 | SAMEINFO_VALIDATION_V1 | 0/54 | CONSTRAINT_AVOIDING_CRAWL |
| sameinfo_constrained | Matched Constrained RL reference (V2 checkpoints on SAMEINFO_VALIDATION_V1) | B_EXPLICIT_CONSTRAINTS | [11,29,47] | 900000 | SAMEINFO_VALIDATION_V1 | 23/54 | CONSTRAINED_ROBUST_ADVANTAGE |
| requirement_conditioned | Requirement-Conditioned SAC (V1) | C_FUTURE_GOAL_CONDITIONED | [11,29,47] | 900000 | five-requirement Validation matrix on 3000-3008 | 48/135 | RESPONSIVE_BUT_UNRELIABLE |
| goal_sac_no_her | Goal-conditioned SAC (without HER) | C_FUTURE_GOAL_CONDITIONED | [11,29,47] | 900000 | Validation 3000-3008 | 0/27 | ZERO |
| goal_sac_her | Goal-conditioned SAC (with HER) | C_FUTURE_GOAL_CONDITIONED | [11,29,47] | 900000 | Validation 3000-3008 | 0/27 | ZERO |
| gcsl | Goal-Conditioned Supervised Learning (canonical V1) | C_FUTURE_GOAL_CONDITIONED | [11,29,47] | 900000 | GCSL_VALIDATION_V1 | 0/54 | GCSL_ZERO |
| contrastive_depth4 | Projected-goal Contrastive RL (depth 4) | C_FUTURE_GOAL_CONDITIONED | [11,29,47] | 900000 | Validation 3000-3008 | 0/27 | NO_ROBUST_TRANSFER |
| contrastive_depth16 | Projected-goal Contrastive RL (depth 16) | C_FUTURE_GOAL_CONDITIONED | [11,29,47] | 900000 | Validation 3000-3008 | 1/27 | NO_REPEATABLE_DEPTH_GAIN |
| learned_dynamics_mbrl | Mixed learned-dynamics MBRL (terminated attempt) | D_MODEL_BASED_RL | [11] | 60500 | Development 2000-2008 at 50k only | 0/9 | TERMINATED_RESOURCE_INFEASIBLE |
| physics_model_mbrl | Exact one-step Physics-Dynamics MBRL (V1) | D_MODEL_BASED_RL | [11,29,47] | 900000 | Validation 3000-3008 | 9/27 | NO_MATERIAL_IMPROVEMENT |
| mve_real_only | Multi-step study real-only control (V1) | D_MODEL_BASED_RL | [11,29,47] | 900000 | Validation 3000-3008 | 10/27 | MATCHED_CONTROL |
| physics_mve | Multi-Step Physics MVE (V1) | D_MODEL_BASED_RL | [11,29,47] | 900000 | Validation 3000-3008 | 0/27 | DEGRADATION_AND_UNSTABLE |
| physical_oracle | Full-route privileged physical reference (V1) | E_PHYSICS_REFERENCE | "none" | 0 | PHYSICS_REFERENCE_HOLDOUT_V1 | 36/36 | PHYSICALLY_SOLVABLE_REFERENCE |
| physical_causal | Observation-limited causal physical reference (V1) | E_PHYSICS_REFERENCE | "none" | 0 | PHYSICS_REFERENCE_HOLDOUT_V1 | 36/36 | CAUSAL_REFERENCE_STRONG_ON_TESTED_TASKS |
| hybrid_h11 | Hybrid RL H11 (V1 frozen 300k) | F_PHYSICS_SHIELDED_LEARNING | [11] | 300000 | HYBRID_VALIDATION_V2 | 35/36 | FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN |
| hybrid_h29 | Hybrid RL H29 (V1 frozen 300k) | F_PHYSICS_SHIELDED_LEARNING | [29] | 300000 | HYBRID_VALIDATION_V2 | 35/36 | FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN |
| hybrid_h47 | Hybrid RL H47 (V1 frozen 300k) | F_PHYSICS_SHIELDED_LEARNING | [47] | 300000 | HYBRID_VALIDATION_V2 | 35/36 | FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN |
| causal60_capability | CAUSAL60 (V2 Capability) | G_IMPROVED_NONLEARNING_CONTROL | "none" | 0 | MPC_V2_CAPABILITY | 18/18 | MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR |
| time_aware_capability | TIME_AWARE (V2 Capability) | G_IMPROVED_NONLEARNING_CONTROL | "none" | 0 | MPC_V2_CAPABILITY | 18/18 | MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR |
| fixed_price_mpc_v2 | FixedPrice MPC V2 (V2 Capability) | G_IMPROVED_NONLEARNING_CONTROL | "none" | 0 | MPC_V2_CAPABILITY | 18/18 | MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR |
| adaptive_price_mpc_v2 | AdaptivePrice MPC V2 (V2 Capability) | G_IMPROVED_NONLEARNING_CONTROL | "none" | 0 | MPC_V2_CAPABILITY | 18/18 | MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR |
| slacktrader_v1 | SlackTrader (V1 stopped before training) | H_UNSUCCESSFUL_FOLLOWUP | "NOT_STARTED" | 0 | four-task MPC Development preflight | 16/16 | RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT |
| temporal_continue | CONTINUE_FEEDBACK temporal options (V1) | H_UNSUCCESSFUL_FOLLOWUP | "none" | 0 | TEMPORAL_OPTIONS_CAPABILITY | 18/18 | SEMANTIC_OR_INTEGRITY_STOP |
| temporal_reset | RESET_FEEDBACK temporal options (V1) | H_UNSUCCESSFUL_FOLLOWUP | "none" | 0 | TEMPORAL_OPTIONS_CAPABILITY | 18/18 | SEMANTIC_OR_INTEGRITY_STOP |
| temporal_commit | COMMIT_FEEDBACK temporal options (V1) | H_UNSUCCESSFUL_FOLLOWUP | "none" | 0 | TEMPORAL_OPTIONS_CAPABILITY | 18/18 | SEMANTIC_OR_INTEGRITY_STOP |
| counterfactual_predictor | Counterfactual Option Value predictor (V1 pilot) | H_UNSUCCESSFUL_FOLLOWUP | "Fit/Calibration/Test route panels; not RL seeds" | 0 | CF_OPTION_TEST_V1 states | NOT_RECORDED/NOT_RECORDED | INSUFFICIENT_IMPROVEMENT_OPPORTUNITY |

Fields not supported by immutable artifacts are `NOT_RECORDED`; they are not imputed. Full observation, action, physics-information, objective, integrity, checkpoint, and evidence-path fields are in the JSON.
