# Draft Preregistered Final-Paper Evaluation Protocol

**State:** draft for protocol review only. It neither authorizes nor implements
Paper evaluation. Paper numeric identities 4000--4017 remain
`SEALED_UNINSPECTED`; no task content, route, hash, serialization, reset, or
step is accessed by this protocol draft.

## 1. Scientific question and frozen scope

The primary question is whether frozen learned Hybrid residuals add efficiency
relative to the strongest information-matched non-learning controller while
preserving hard-requirement satisfaction. Secondary questions decompose the
non-learning architecture and describe training-seed variation. The proposed
arms are in `PAPER_EVAL_ARM_PROPOSAL.md`; a later human authorization must
freeze the final set before any Paper content is constructed.

## 2. Evaluation population and unit

- The population is the 18 pre-reserved Paper numeric identities 4000--4017 in
  their already frozen order. This draft records only the numeric exclusion.
- A task is the primary paired unit. Controller-task executions are repeated
  measurements on tasks, not independent routes.
- Each authorized controller-task pair is executed at most once. There are no
  tuning, warm-up, replacement, or rerun episodes on Paper content.
- All original requirements and production dynamics remain unchanged.

## 3. Endpoints

### Primary endpoint

Raw Requirement Satisfaction Rate (RSR): number of Paper tasks satisfying all
original completion, deadline, and speed requirements divided by 18. The raw
denominator is never changed by a later feasibility classification.

For every arm also report, separately:

- completion count/18;
- deadline-compliance count/18;
- speed-compliance count/18;
- exact task identities of each failure and cross-controller failure overlap;
- travel time and maximum speed violation for every completed attempt.

### Primary efficiency estimand

For a treatment controller and AdaptivePrice baseline, on task `i` where both
are feasible under the original requirements, define

`saving_i = 100 * (E_adaptive_i - E_treatment_i) / abs(E_adaptive_i)`.

The reported estimate is the arithmetic mean of `saving_i`, accompanied by
paired count, distinct task count, median, range, and the mean/range of
`E_adaptive_i - E_treatment_i` in kWh. Zero or numerically degenerate baseline
energy is reported as an integrity problem rather than repaired by a changed
formula.

For the Hybrid method-level estimand, first average `saving_i` across H11,
H29, and H47 on tasks where AdaptivePrice and all three Hybrid seeds are
feasible, then average those task-level seed means across tasks. Per-seed
contrasts remain required secondary results. This prevents the shared task set
from being counted three times and prevents best-seed selection.

Energy is never used to declare superiority when lower raw feasibility is
hidden by conditioning. Any Hybrid-only failure on a task completed by
AdaptivePrice is reported before the energy estimate and blocks an unqualified
“reliability preserved” claim.

## 4. Predeclared comparisons

1. **Primary learned contrast:** the three-seed Hybrid method-level estimand
   versus AdaptivePrice, conditional on all four being feasible, preceded by
   all four raw RSRs and failure overlap.
2. **Training-seed variation:** H11, H29, and H47 separately versus
   AdaptivePrice; no seed is selected or discarded.
3. **Architectural decomposition:** FixedPrice versus TIME_AWARE,
   AdaptivePrice versus FixedPrice, and CAUSAL60 versus TIME_AWARE.
4. **Reliability contrasts:** paired raw success for each Hybrid seed versus
   AdaptivePrice.
5. Any conditionally authorized Constrained V2 or privileged arm is secondary
   and uses the same raw-denominator and pairing rules.

The primary set and contrast order must be frozen before the seal is opened.

## 5. Uncertainty and multiplicity

- Raw counts and proportions are always reported. Each paired RSR contrast is
  summarized by the paired risk difference and discordant task identities.
- Exact two-sided McNemar tests use the conditional binomial distribution over
  discordant pairs. The three seed-specific Hybrid-versus-Adaptive tests form
  one family and use Holm correction. With no discordant pairs, report that
  fact rather than an exaggerated guarantee.
- Energy means receive a deterministic 20,000-replicate percentile bootstrap
  over task identities. All treatment values for a sampled task, including the
  three Hybrid seeds, travel together. The PRNG algorithm, software version,
  and seed derived from a frozen analysis-config SHA-256 are persisted.
- The primary method-level energy contrast may additionally use a two-sided
  exact paired randomization test over task-level seed-mean differences.
  Seed-specific energy tests are exploratory and Holm-adjusted as a family.
- Intervals describe this frozen task panel; population inference is not
  asserted. Medians, ranges, absolute differences, and raw paired values remain
  available even if an interval is unstable.
- No post-outcome power analysis, favorable one-sided switch, or denominator
  revision is permitted.

## 6. Physical-feasibility classification

After original outcomes are durably finalized, the preregistered analyzer in
`feasibility.py` may classify an authorized task as `FEASIBLE_WITNESS`,
`PROVEN_INFEASIBLE`, or `UNRESOLVED` using the frozen production semantics and
conservative numerical margins. It must be adapted only through a separately
reviewed future protocol that opens Paper task metadata under explicit
authorization; this draft does not do so.

- Controller failure is never an infeasibility proof.
- A witness must match the exact route and full task and satisfy the original
  completion, deadline, and speed requirements.
- Proven impossible and unresolved tasks stay in the raw 18-task RSR.
- A feasibility-conditional RSR may be reported only as a secondary,
  explicitly retrospective analysis with its exact denominator and proof/witness
  classes. Both original and conditional results appear together.
- No outcomes may influence proof constants, controller inclusion, or task
  replacement.

## 7. Exactly-once execution safeguards for a future authorization

These controls are requirements, not claims about current implementation:

1. Obtain explicit written Paper execution authorization naming branch,
   commit, owners, arms, and task numeric identities.
2. Freeze source tree, environment, configs, ordered arm/task schedule,
   checkpoint bytes, and all SHA-256 values before opening content.
3. Acquire exclusive process ownership through an atomic lock containing run
   identity, host, process, and start time; fail closed on ambiguity.
4. Build an immutable pair ledger. Persist and fsync `STARTED` with unique
   attempt ID **before** the only permitted reset.
5. Permit neither another reset nor a new attempt for a `STARTED` pair. Resume
   only from a verified full native environment/controller/RNG state from that
   same attempt; otherwise record a permanent incident and stop.
6. Record every native step, action, state needed for physical accounting,
   speed-limit observation, energy contribution, termination, and all safety
   checks. Retain `completion_bound_s` explicitly.
7. Persist the complete pair result to a temporary file, fsync, independently
   verify identities/accounting/checksums, then atomically rename and mark the
   ledger pair `COMPLETE` with result hash.
8. Never re-execute a completed, failed, or ambiguous pair. Unexpected process
   loss, missing state, hash mismatch, duplicate ownership, or incomplete
   physical accounting causes an integrity stop, not an automatic rerun.
9. An independent read-only verifier reconstructs aggregate records solely
   from complete hashed pair results and checks the frozen denominator.
10. After the schedule finishes or stops, permanently seal task content,
    ledger, logs, results, and incident record; revoke execution capability.

These safeguards specifically prevent recurrence of Hybrid Validation V1's
unpersisted completed episodes/reset incident and Temporal Options V1's missing
`completion_bound_s` evidence.

## 8. Predefined failure reporting

Every non-success receives exactly one primary outcome tuple (completion,
deadline, speed) plus termination reason, task identity, controller identity,
attempt state, integrity status, and physical-feasibility status when available.
Crashes and integrity incidents are not scientific controller failures and are
not rerun; both the intended denominator and missing result are shown. All
deviations and amendments are time-stamped and reported even if unfavorable.

## 9. Reproducibility deliverables

- final authorization and analysis configuration;
- Git commit and clean-tree proof;
- environment/package lock and platform metadata;
- source, config, checkpoint, ordered schedule, ledger, per-pair result, and
  aggregate SHA-256 manifests;
- checkpoint load verification without parameter mutation;
- independent source/result/denominator verifier and its test log;
- complete outcome table and incident register;
- a permanent state proving the Paper set is closed after use.

## 10. Authorization boundary

Committing or approving this draft does not authorize final Paper evaluation.
No executable runner is included. An authorized implementation and its safety
tests must be reviewed in a separate task before the first task construction or
environment reset.
