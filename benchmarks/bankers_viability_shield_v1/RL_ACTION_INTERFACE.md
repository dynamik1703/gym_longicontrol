# Proposed future policy contract

Input: original causal 8-vector; disclosed action-propagated position, velocity, acceleration, elapsed time and physical memory; remaining route and deadline; current 21-bit safe-action mask; optionally causal certificate slacks. No full route, seed, oracle, future realized state or outcome-derived feature.

Output: one integer in [0,20] selecting exactly [-1,-0.9,...,1]. Invalid choices must have zero probability before sampling. The provided masked_distribution utility excludes them from normalization and entropy. Any future discrete SAC target expectation and PPO log-probability/entropy must use the corresponding current or successor mask. Replacing a sampled invalid action is not an equivalent learning rule. A final full native certificate runs before execution; discrepancy is a safety-integrity event.

The proposed controlled experiment shares a compact discrete SAC or PPO learner, network capacity, update count, reward, native action menu, causal observation contract, training tasks/budgets, evaluation tasks and physical environment. Treatments: no shield, inherited post-shield, BankerShield pre-mask. The declared mask is the observation-contract exception. Record proposed and executed actions and account explicitly for the induced sampling/execution differences. Compare signed energy only on feasible outcomes and include strong causal physical baselines.

No learner is implemented or trained here. GCSL/Constrained RL V2 interface adaptations are separate scientific changes. A speed certificate does not certify the 140 s deadline. Future training or Validation requires separate authorization regardless of this study's classification.
