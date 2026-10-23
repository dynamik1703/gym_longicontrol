# Draft Manuscript Outline

## 1. Introduction and research question

Frame longitudinal control as a joint reliability-and-efficiency problem in
which completion, deadlines, and speed limits are hard requirements. Ask what
enables reliable satisfaction and what learning adds after physical structure
is available. State the scoped thesis and negative-result policy.

## 2. Related work and scope

Position the work at the intersection of constrained/safe RL, goal and future
conditioning, model-based RL, hybrid residual control, and optimization-based
vehicle control. Define the causal information boundary. Defer external claims
until the gaps in `RELATED_WORK_GAPS.md` are researched and cited.

## 3. LongiControl benchmark and hard requirements

Specify the 0.1 s dynamics, causal observation, native continuous action,
completion condition, deadline, right-endpoint speed check, and energy metric.
Distinguish route identity from full-task identity and describe the train,
Validation, capability, holdout, and still-sealed Paper roles.

## 4. Experimental protocol and reproducibility

Describe frozen code/configuration, seeds 11/29/47, immutable ledgers,
controller-task accounting, RSR, separate compliance outcomes, paired energy,
and integrity incidents. Explain `PAIRED_SAME_TASKS`, `PAIRED_SAME_STREAM`,
`CONTEXTUAL_ONLY`, and `NOT_COMPARABLE`.

## 5. Baseline RL and reward formulation findings

Synthesize scalar SAC/PPO, reward-factorial, action-repeat, binary-success, and
LLM-reward findings as bounded failures or weak results. Avoid averaging across
incompatible panels.

## 6. Future-aware and constrained learning findings

Lead with the matched Same-Information comparison (0/54 versus 23/54), then
separate Requirement-Conditioned, Goal-SAC/HER, GCSL, and contrastive findings.
Explain why completion alone did not imply requirement satisfaction.

## 7. Physical feasibility references

Present the causal and privileged physical references on their common 36-task
holdout. Treat the privileged reference as contextual. Explain why these
results motivate physical structure without proving global safety or
optimality.

## 8. Physics-shielded Hybrid RL

Document the V1 incident, the amended V2 design, raw 35/36 for all five arms,
and task-paired energy effects for H11/H29/H47. Report all seeds and the missed
target; do not select H47 post hoc.

## 9. Strong causal MPC comparisons

Compare CAUSAL60, TIME_AWARE, FixedPrice, and AdaptivePrice on the 18-task
capability panel. Decompose the 5.422% Adaptive-versus-TIME_AWARE effect into
the 4.644% Fixed-versus-TIME_AWARE architecture contrast and the smaller
0.801% Adaptive-versus-Fixed contrast, subject to joint-change confounding.

## 10. Physical infeasibility in generated tasks

Give the discrete proof for 10200029, general audit method, 187 witness / 9
proof / 215 unresolved counts, accepted-versus-rejected separation, and impact
on historical interpretations. Raw denominators never change.

## 11. Negative follow-up studies and interpretation

Report learned-dynamics resource termination, MVE degradation, SlackTrader
preflight stop, Temporal Options integrity/semantic stop, and Counterfactual
opportunity failure. Separate “not tested,” “failed under budget,” and “failed
preregistered gate.”

## 12. Limitations and threats to validity

Use `THREATS_TO_VALIDITY.md`: shared routes, few seeds, post-hoc feasibility,
incomplete task serialization, heterogeneous panels, descriptive intervals,
and finite-panel physical-reference evidence.

## 13. Implications for reliable learning-based controllers

Argue for feasibility before efficiency, explicit constraints, causal physical
shields, matched strong non-learning baselines, per-failure identities, and
exactly-once evaluation. Describe learned residuals as a hypothesis requiring
a final matched test.

## 14. Conclusion and future research

Conclude only that physical structure and explicit constraints were decisive
on tested panels. Leave the incremental value of learning unresolved pending a
separately authorized sealed evaluation. Identify principled opportunity
estimation and statistically efficient task generation as future work.
