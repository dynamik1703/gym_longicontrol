# Physical Feasibility Reference

Status: **complete**. The independent holdout was opened once after source,
configuration and split freeze.

This is a control/feasibility reference study, not an RL comparison and not a
claim that MPC "beats" RL. It separates:

1. physical feasibility with privileged full-route information;
2. causal controllability through the ordinary 8D sensor plus disclosed memory;
3. learning feasibility, which this study does not test.

## Main result

On `PHYSICS_REFERENCE_HOLDOUT_V1`, both deterministic references satisfied all
requirements on all 36 tasks:

| Reference | RSR | Completion | Deadline | Speed |
| --- | ---: | ---: | ---: | ---: |
| Full-route Physics Oracle | 36/36 | 36/36 | 36/36 | 36/36 |
| Observation-limited causal reference | 36/36 | 36/36 | 36/36 | 36/36 |

Every Oracle result has a persisted action sequence and an exact independent
replay with identical action, state-trace and metric hashes. This establishes
`FEASIBLE_WITNESS` on 36/36 tested holdout tracks. It is an achieved result, not
a certified upper bound over all possible tracks.

The factual interpretation is **PHYSICALLY_SOLVABLE_REFERENCE** and
**CAUSAL_REFERENCE_STRONG** on the tested panels. `ORACLE_GAP_LARGE` is not
supported: paired completion/deadline/speed/RSR differences are zero on every
holdout task. The causal controller is stateful and must not be described as a
memoryless 8D policy.

## Artifacts

- `SOURCE_AUDIT.md`: authoritative simulator semantics and physical findings.
- `DESIGN.md`: both controller methods and information boundaries.
- `PROTOCOL.md`: frozen lifecycle and classification rules.
- `SPLIT_LEDGER.md` / `split_ledger.json`: route-clean construction and hashes.
- `RESOURCE_REPORT.md` / `resource_measurements.json`: execution gates.
- `RESULTS.md` / `results.json`: final scientific summary.
- `development_results.json` and `holdout_results.json`: immutable task records.
- `witnesses/`: 54 independently replayed Oracle action trajectories.

The optional post-hoc physical evaluation on `SAMEINFO_VALIDATION_V1` and
`GCSL_VALIDATION_V1` was not performed. Existing RL results are discussed only
as non-paired context; no RL policy was rerun or modified.

## Reproduction

The holdout record already exists and the runner deliberately refuses another
holdout invocation (`PHYSICS_REFERENCE_HOLDOUT_V1_ONE_OPEN_ONLY`). Safe checks:

```bash
PYTHONPATH=src:. python -m pytest -q \
  benchmarks/physical_feasibility_reference/tests/test_reference.py
PYTHONPATH=src:. ruff check benchmarks/physical_feasibility_reference \
  tests/benchmark/test_physical_feasibility_reference.py
git diff --check
```

No Paper seed (`4000–4017`) was constructed or inspected.
