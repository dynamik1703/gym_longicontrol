# Prospective masked-RL protocol (no training authorization)
The scientific recommendation is gated by the sealed V2 result. This document
implements no learner, optimizer, replay buffer, training rollout or evaluation.

A suitable first controlled proposal is discrete SAC on the same native grid:
three arms (unshielded discrete RL, inherited native post-shield, BankerShield
pre-mask), three fixed seeds (11,22,33), 300,000 native steps per seed per arm,
fixed two-layer 128-wide networks, identical optimizer/hyperparameter schedule,
original scalar environment reward, training task streams and evaluation routes.
These are prospective choices to freeze under separate authorization, not tuned
V2 methods. A single three-seed arm is 900,000 native training steps; the full
three-arm comparison is 2,700,000 steps and nine learner runs. Runtime estimates
for one masked arm must not be misreported as estimates for all nine runs.

All arms receive the causal ordinary 8D observation and identical declared causal
state/progress/time features, propagated only from reset and executed actions.
Declare exact internally propagated VehicleState, normalized position/remaining
route distance, elapsed/remaining deadline time and previous native action.
The policy chooses a discrete index in the 21-element grid. Use the actual
current-state safe mask for the masked arm's policy distribution: invalid logits
have exactly zero sampling probability; normalize over valid actions only;
exclude invalid terms from policy loss, entropy, entropy target and critic
next-state expectation. Retain/reconstruct the causal next-state mask for targets.
Masking only a sampled action by replacement is post-shielding, not pre-masking.
Immediately before execution, reverify the selected full native certificate;
empty mask or discrepancy stops the run. Propagate from the actually executed
action in every arm, including post-shield replacements.

Use identical integrity controls, exactly-once attempt ledgers and route-clean
training streams, with all V2 Development tasks protected from future training.
Do not reuse unopened Validation or Paper content for design or calibration.
Freeze the future routes, training streams, reward, widths and budgets before
any training outcomes. Report completion, deadline, exact endpoint speed and
energy separately; feasibility uses the original evaluator. Energy comparisons
use feasible trajectories and jointly feasible task pairs only. Preserve raw
success denominators and unresolved physical feasibility.

Resource planning must include native mask + final certificate, policy/critic
updates and next-state masks, full evidence retention policy, inherited-post
cost, evaluation and integrity replay. The V2 resource report provides measured
online and two-mask projections for one 900k-step masked arm. A projection below
24h for mask computation alone does not certify the full training budget.
No V2 result supplies a universal deadline guarantee or guarantees RL success.
