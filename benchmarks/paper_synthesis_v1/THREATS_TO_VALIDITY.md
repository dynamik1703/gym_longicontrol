# Threats to Validity

## Construct validity

- RSR is a conjunction of completion, deadline, and speed compliance. It is
  primary because energy alone rewards non-completion, but it does not measure
  comfort, robustness to unmodeled disturbances, or real-world safety.
- Feasible-only signed energy is conditional on joint success and can describe
  efficiency only after reliability is reported. It is not a universal energy
  lower bound.
- A recorded successful trajectory is an existence witness, not evidence that
  a controller or task distribution is generally easy.
- The infeasibility lower bound is sufficient, not necessary. `UNRESOLVED`
  must not be relabeled feasible.

## Internal validity

- Same-Information Scalar versus Constrained is the closest learning contrast,
  but differs in more than one optimization/formulation component.
- FixedPrice/AdaptivePrice versus TIME_AWARE changes both profile-family and
  terminal treatment. Only AdaptivePrice versus FixedPrice isolates their
  recorded price adaptation contrast, and even that is algorithm-specific.
- Hybrid seeds share the same evaluation routes, so 105 successful
  controller-task pairs are not 105 independent routes.
- Hybrid Validation V1 lost two completed results and performed one reset
  without action. It remains `INCIDENT_ABORTED`; V2 is explicitly amended.
- Temporal Options did not retain `completion_bound_s`; its result remains a
  semantic/integrity stop rather than positive evidence.
- Some historical artifacts lack canonical serialized task identities,
  per-task results, gradient counts, or immutable checkpoint hashes. Missing
  fields are `NOT_RECORDED`, never inferred.

## External validity

- Panels are small generated-task sets under one longitudinal dynamics model,
  action rate, speed semantics, and initial state. Finite-panel success is not
  population-wide reliability or a real-vehicle safety theorem.
- Cross-panel results (for example, 36/36 physical holdout and 35/36 Hybrid
  Validation) cannot rank controllers because populations and information
  contracts differ.
- The historical feasibility corpus is selected by artifact availability and
  study filters. Nine proofs among 411 tasks do not estimate generator-wide
  prevalence.
- Results may not transfer to noisy sensing, actuation delay, friction changes,
  traffic interactions, or adversarial disturbances.

## Statistical conclusion validity

- Three training seeds provide limited information about optimizer variation.
- Historical confidence intervals are descriptive task-bootstrap summaries;
  shared tasks and training-stream reuse preclude naive independent-sample
  inference.
- Feasible-only comparisons may change composition when controllers have
  different failure sets. The raw RSR and failure identities therefore precede
  every energy result.
- Many exploratory studies and comparisons were performed. Historical results
  are not retroactively converted into confirmatory tests.
- A failed acceptance gate bounds the tested design; it does not prove a
  hypothesis mathematically impossible.

## Retrospective-analysis risk

The feasibility audit was performed after historical outcomes existed. Its
proofs use task physics rather than controller behavior, but any
feasibility-conditional RSR is explicitly post-hoc. The original denominator
is always retained, including task 10200029. Paper tasks 4000--4017 were not
used to tune the proof or estimate prevalence.

## Reproducibility risk

The repository records configs and results unevenly across its history.
Constrained V2 checkpoint files exist locally but their byte hashes were not
embedded in the original result record. Hybrid checkpoints have stronger
recorded hash provenance. Final inclusion requires a human decision about the
acceptable provenance threshold; missing models must not be silently retrained.
