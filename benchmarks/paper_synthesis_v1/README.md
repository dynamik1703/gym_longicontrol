# LongiControl Paper Synthesis V1

This directory consolidates the completed LongiControl research programme at
source commit `9802f9262ea7f753365fc4bdd20b1fe08379291d`. It is an analysis-only
package: it reads immutable historical records, verifies their hashes, and
produces audit and paper-draft artifacts. It does not import an environment,
construct a reserved Paper task, call `reset()`/`step()`, train a model, or
evaluate a controller.

The authoritative numerical records are JSON/CSV. Markdown files explain and
render those records; they are not independent result sources.

## Artifact map

- `FEASIBILITY_PROOF.md`, `FEASIBILITY_AUDIT.md`,
  `feasibility_audit.json`, and `historical_task_status.csv` contain the
  historical physical-feasibility method and results.
- `method_registry.json`, `METHOD_REGISTRY.md`,
  `paired_comparison_registry.json`, and the master evidence matrix reconcile
  completed methods and comparisons.
- `CLAIM_EVIDENCE_MATRIX.md`, `PAPER_TITLE_ABSTRACT.md`, `PAPER_OUTLINE.md`,
  and the figure/threat/literature registers form the manuscript draft.
- `PAPER_EVAL_ARM_PROPOSAL.md` and `PAPER_EVAL_PROTOCOL_DRAFT.md` are a
  non-executable proposal requiring later human authorization.
- `source_registry.json`, `preparation_status.json`, `verify.py`, and focused
  tests provide machine-checkable provenance and safeguards.

## Authorization boundary

- Historical task serializations already present in committed ledgers may be
  audited.
- Numeric Paper exclusions `4000` through `4017` may be checked only as status
  metadata. Their task content must not be constructed, serialized,
  fingerprinted, or evaluated.
- A controller failure is never treated as an infeasibility proof.
- A positive witness requires an exact, task-linked historical trajectory that
  completed within the original deadline and speed bound.
- No final Paper evaluation is authorized by this package.

## Read-only verification

```text
python -m benchmarks.paper_synthesis_v1.feasibility --check
python -m benchmarks.paper_synthesis_v1.evidence --check
python -m benchmarks.paper_synthesis_v1.verify
python -m pytest -q -p no:cacheprovider benchmarks/paper_synthesis_v1/tests
ruff check benchmarks/paper_synthesis_v1
```

Only the explicit `--write` mode of the two analysis modules updates generated
artifacts in this directory. Neither mode invokes a scientific runner.

`verify.py` also audits locally available checkpoint bytes, but never imports
or deserializes a checkpoint. Missing optional local run storage is reported as
unavailable; a present hash mismatch fails verification.
