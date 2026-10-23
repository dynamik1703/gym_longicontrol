# Working Title and Provisional Abstract

## Working title

**Hard Constraints First: What Learning Adds After Physical Feasibility in
Longitudinal Control**

## Provisional abstract (249 words)

Reliable longitudinal control couples a sparse terminal objective with hard
deadline and speed constraints. We study which design choices enable such
requirements to be met, and whether reinforcement learning adds measurable
benefit once strong physical structure is available. LongiControl records are
consolidated across reward-based, constrained, goal-conditioned, model-based,
physics-reference, physics-shielded, and non-learning optimization studies.
All comparisons retain their original task panels and denominators. In the
closest matched learning comparison, a dense-deadline constrained method
achieved 23/54 requirement-satisfaction outcomes while Same-Information
Scalar SAC achieved 0/54. Goal-conditioned supervised learning achieved 0/54;
several other RL variants were weak or unsuccessful under tested budgets.
Observation-limited and
privileged physical references each achieved 36/36 on a separate holdout, but
the privileged reference is not an information-matched competitor. On amended
Hybrid Validation V2, a causal controller, a time-aware controller, and three
frozen hybrid policies each achieved a raw 35/36; the hybrid policies reduced
paired energy relative to the time-aware controller by 1.04--1.70% on the 35
jointly feasible tasks. A retrospective audit proves the common failed task
physically infeasible without changing raw RSR. Improved non-learning MPC
achieved 18/18 and reduced energy by 5.42% relative to the time-aware baseline,
but only 0.80% relative to FixedPrice MPC, limiting attribution to learning.
These findings support a scoped conclusion: explicit constraints and causal
physical structure close reliability gaps on the tested panels, whereas the
incremental value of learning over strong matched optimization remains
unresolved. We provide a reproducible feasibility audit, evidence registry,
and sealed draft protocol for a separately authorized final evaluation.

## Research questions

1. Which formulations reliably satisfy completion, deadline, and speed
   requirements under a common causal information contract?
2. How much of the observed benefit follows from explicit constraints and
   physical structure rather than from learning itself?
3. Does a frozen learned residual improve energy use over the strongest
   information-matched non-learning controller without reducing raw RSR?
4. How do physically impossible generated tasks affect interpretation while
   preserving the original evaluation denominator?
5. Which negative results delimit, rather than disprove, the plausible value
   of future information, model-based updates, and option-value learning?

## Proposed contributions

- A hash- and task-aware synthesis of 34 completed method/configuration rows
  without new training or controller episodes.
- A conservative discrete-time infeasibility proof and an audit separating
  witnesses, proofs, and unresolved tasks.
- Matched evidence that explicit constrained learning can outperform a
  Same-Information scalar formulation on the tested stream.
- An attribution analysis separating architectural MPC improvement from the
  smaller observed increment associated with learned hybrid policies.
- A sealed, exactly-once draft protocol for a human-authorized final Paper
  evaluation.
