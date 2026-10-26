# Preregistered V2 Development protocol
Frozen physical task: StochasticTrack-v1, original BMW i3, 1000 m, deadline 140 s,
dt 0.1 s, TimeLimit 1800, native [-1,1], ordinary 8D observation, zero exact speed
tolerance. Authoritative evaluator: is_feasible(EpisodeMetrics, TaskSpecification).
The 21 grid points i/10, i=-10..10, discretize the policy interface only.

A CAUSAL60: frozen original continuous observation-limited action law, executed
unchanged. Diagnostic shadow masks never affect it.
B LEGACY_NATIVE_POST_GREEDY: propose +1, apply inherited native endpoint, known
braking, unknown-preview and deadline predicates including the frozen identical
fallback exception; execute CAUSAL60 replacement when rejected. This native +1
arm is not the original capped target-speed controller and has no universal
legacy safety theorem.
C BANKERS_GREEDY: full certified mask, largest admitted action, fresh final
certificate, exact execution and causal state update.
D BANKERS_MASKED_CAUSAL60: same causal continuous proposal, nearest certified grid
point, equal-distance ties to smaller numeric action, fresh final certificate.
No controller adaptation, calibration, learning or future route input.

Reserve BANKERS_DEV_V2 before constructing any Development task. Proposed scan
11600000 upward; if already claimed, select the next unused million block before
selection. Protect inherited canonical training streams, historical Development,
opened Diagnostic/Validation and legitimately available reserved fingerprints,
all MPC/Hybrid/Chunk/Counterfactual/V1 ledgers, rejected constructed candidates,
internal duplicates and numeric Paper 4000..4017. Never construct Paper content.
Accept the first 18 Policy-R numeric/route/full-task-clean tasks without using
performance. Freeze ordered task serializations, rejection reasons, source/config,
four controllers and all 72 unique ATTEMPT_01 identities before any reset.

Execute task-paired A/B/C/D in that order, 18 tasks, exactly 72 episodes maximum.
Acquire exclusive writer, durably STARTED before reset; atomically publish full
native evidence, read disk and independently replay all physical records and masks
before COMPLETED. Completed pairs are never rerun. Any ambiguous STARTED pair
forbids restart; no same-attempt recovery protocol is authorized. Commit the
frozen protocol before scientific execution. No source/config mutation afterward.

Gates fixed prospectively:
A conditional proof assumptions and certificate semantics supported.
B no certified executed speed violation, complete required synthetic coverage and
independent Development verification. A violation retains trace and stops as
BANKERS_V2_COUNTEREXAMPLE_FOUND; no margin tuning.
C each Banker arm's ordinary Development distribution: >=25% states have at least
two distinct successor velocities with span >=0.05 m/s, and >=5% states have at
least one physically positive-acceleration admitted action. Report arms and pool.
D D incurs zero additional feasibility failures where A is feasible. This is a
strict grid readiness criterion, not a deadline theorem. C failure alone does
not fail D. Preserve the raw 18-task denominator.
E all 72 uniquely executed, provenance/hashes/counts/independent replay pass.
F total active scientific compute <=21600 s, retained V2 artifacts <=3 GiB.
Before opening Development, project 72*1800 steps at the most expensive measured
synthetic high-speed or full-fixture online+independent replay+evidence I/O cost,
with 25% overhead plus measured preflight compute. Charge all arms that cost.
Stop if either bound is exceeded; do not shrink the experiment.

Report raw RSR, completion/deadline/exact speed/failure IDs, travel/progress,
feasible-only signed net energy (mean/median), jointly feasible paired energy,
actions/backup, authority, mask latency/calls and rejection reasons. Matched
legacy/Banker certificates use identical states, separately from policy outcomes;
disagreement categories: actual unsafe endpoint, uncertifiable backup, horizon,
deadline-only, proposal/domain, numerical/integrity. Uncertifiable is not impossible.
After sealing original results, verified trajectories establish FEASIBLE_WITNESS;
only rigorous physical impossibility proofs establish PROVEN_INFEASIBLE; otherwise
UNRESOLVED. No post-hoc denominator changes or unopened Validation/Paper content.

Principal outcome priority: proof unresolved, counterexample, integrity/resource
stop at the required failed stage; after complete sound evidence, speed-safe but
action-authority-low if C fails, speed-safe but progress-insufficient if D fails,
otherwise READY_FOR_MASKED_RL_DESIGN. No negative result is upgraded. RL projections
are exploratory only: 3 seeds *300k native steps, 24h budget, no execution.
