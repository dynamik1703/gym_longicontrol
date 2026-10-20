# Frozen Protocol

## Canonical task

- route: 1,000 m;
- deadline: 140.0 s;
- maximum speed violation: exactly 0.0 m/s;
- native step: 0.1 s;
- maximum episode length: 1,800 steps;
- action: continuous `[-1, 1]`.

Success is exactly external `is_feasible(EpisodeMetrics, TaskSpecification)`.
No evaluator epsilon is added.

## Evidence classes

- `FEASIBLE_WITNESS`: an action sequence executed and independently replayed in
  the original simulator satisfies every hard requirement.
- `PROVEN_INFEASIBLE`: an independent mathematical lower bound establishes
  impossibility. Optimizer failure is insufficient.
- `UNRESOLVED`: neither of the above.

The only implemented infeasibility certificate is the constraint-free exact
maximum-action completion lower bound exceeding the deadline. It was not
triggered by any study task.

## Lifecycle

1. Construct `PHYSICS_REFERENCE_DEV_V1` from seed 9,800,000 upward.
2. Construct `PHYSICS_REFERENCE_HOLDOUT_V1` from seed 9,900,000 upward.
3. Exclude prior canonical training and legitimately opened evaluation content
   by numeric seed, route hash and full-task hash; never construct Paper.
4. Validate physical fixtures and Development behavior.
5. Run resource gate (≤12 h, ≤12 GB).
6. Freeze controller sources, margins, task and split hashes.
7. Evaluate Development with the frozen implementation.
8. Open the holdout once. Persist per-task results and every feasible Oracle
   action trajectory. Refuse a second holdout invocation.
9. Independently replay each retained Oracle trajectory and require exact action,
   state-trace, metric and final-position parity.

The optional already-opened RL contextual panels are omitted. No RL training,
distillation, checkpoint mutation or Paper evaluation is authorized or run.

## Frozen hashes

- scientific configuration:
  `f63717504a810c75d7080388aeb159505cf667ad13248dbe4a53ec0c4bb94cd1`
- Development split:
  `4551ddfa2f4363aff35323a2ed5994631f5a26bb50948af66f9d78b7558d3d1f`
- Holdout split:
  `c02c4889333eaf3526277b10bc5c33101ac1aa97f9de836379088c661e72a30b`

Per-source hashes are recorded in `freeze.json`; per-witness action and trace
hashes are recorded beside the frozen actions.
