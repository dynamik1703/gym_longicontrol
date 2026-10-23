# Historical Physical-Feasibility Audit

## Result

Across nine committed metadata ledgers, 424 study-task occurrences reduce to
411 unique routes and 411 unique full tasks. Thirteen repeated occurrences are
therefore not counted as independent routes. In these records the route and
full-task counts happen to be equal; the analyzer nevertheless keys and reports
them separately and would not collapse differing requirements.

For all 411 legitimately inspectable tasks:

| Classification | Unique tasks |
| --- | ---: |
| `FEASIBLE_WITNESS` | 187 |
| `PROVEN_INFEASIBLE` | 9 |
| `UNRESOLVED` | 215 |

For the 356 tasks accepted into at least one historical study panel, the counts
are 187, 5, and 164 respectively. The other four proven-infeasible tasks were
scanned but rejected by pre-existing route/full-task reservation policies; they
did not enter those evaluation panels.

The audit matched 628 historical controller-task records to exact serialized
tasks. Of those, 621 are verified feasible trajectory records, representing
187 unique positive task witnesses. Repeated successes by multiple controllers
remain multiple controller-task records but one physical-feasibility witness
identity.

## Proven-infeasible tasks

| Numeric identity/identities | Accepted | Continuous limit-only [s] | Conservative discrete lower bound [s] |
| --- | --- | ---: | ---: |
| 10100003 | yes | 156.000 | 154.217 |
| 10100019 | yes | 144.000 | 141.771 |
| 10200029 | yes | 144.000 | 141.771 |
| 11100013 | yes | 145.200 | 142.436 |
| 11900011 | yes | 143.143 | 140.086 |
| 9100002 | rejected candidate | 144.000 | 141.994 |
| 9900045 | rejected candidate | 158.400 | 156.528 |
| 10300005 / 11100004 | rejected candidate, repeated content | 168.000 | 166.217 |
| 10700014 | rejected candidate | 150.000 | 147.771 |

The table is rendered from `feasibility_audit.json`. Full route/task hashes,
segment values, sources, selection status, and numerical margins are retained
in that file and in `historical_task_status.csv`.

## Formal finding for Hybrid Validation V2 task 10200029

The formal discrete proof in `FEASIBILITY_PROOF.md` is valid. The exact
serialized route/full-task fingerprints reproduce independently. The proof
uses the production 0.1 s timestep, constant-acceleration position expression,
physical velocity and acceleration bounds, right-endpoint checks, transition
crossings, zero initial state, inclusive task inequalities, and explicit
roundoff margins. Its 141.771 s lower bound exceeds 140 s. No controller was
run.

All five Validation V2 controllers failed only this same task and succeeded on
the other 35. Their original raw RSR remains 35/36. A clearly retrospective,
feasibility-conditional description is 35/35 on tasks that now have a feasible
witness; it must never replace the original denominator.

## Effect on historical interpretation

- Hybrid Validation V2's common failure has a physical explanation, not five
  independent controller failures. This strengthens the conclusion that the
  controllers preserved feasibility on the 35 witnessed tasks, while leaving
  the registered 35/36 result intact.
- Accepted identities 10100003 and 10100019 belonged to the earlier Hybrid
  design's frozen prospective Validation metadata. That design study did not
  execute those performance pairs, so there is no historical RSR denominator
  to revise.
- Counterfactual normal sampling recorded late, speed-compliant completions for
  11100013 and 11900011. The proof shows why they could not become feasible by
  controller choice under the original deadline. These are not RL evaluation
  failures and do not change the negative opportunity/predictor classification.
- The remaining historical RL failures cannot generally be attributed to
  physical impossibility. For older studies without exact canonical task
  content or task-paired outcome identities, overlap is
  `NOT_AVAILABLE_UNDER_CURRENT_AUTHORIZATION/EVIDENCE`. Aggregate results such
  as 23/54 or 0/54 were not converted into arbitrary route witnesses.

No causal statement is made from an unpaired cross-panel comparison.

## Coverage by study family

Exact serialized task content is available for constrained-collector audits,
the physical reference, Hybrid design/Validation V2, SlackTrader, causal MPC,
Temporal Options, and Counterfactual Option Value. Per-task witnesses are
available only from the allow-listed records in `feasibility.py`.

Canonical training streams for seeds 11, 29, and 47 and older Scalar, PPO,
reward-engineering, constrained, requirement-conditioned, HER/Goal-SAC, GCSL,
and model-based studies are inventoried in the method/evidence registries.
Where their committed artifacts lack reconstructible task serialization or
task-paired identity, feasibility status remains unavailable rather than being
inferred from aggregate RSR.

## Limitations

- The proof is sufficient, not necessary: `UNRESOLVED` does not mean feasible.
- Witnesses show existence only for the exact task; they do not establish an
  optimal policy or population prevalence.
- Rejected candidates are reported because their already-opened metadata are
  legitimate historical evidence, but they are separated from accepted panels.
- No Paper task was used to estimate infeasibility prevalence.
- The audit covers committed serializations, not every episode ever generated
  in ignored run directories.
