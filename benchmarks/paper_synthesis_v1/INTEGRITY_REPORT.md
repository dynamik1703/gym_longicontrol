# Integrity Report

## Scope and provenance

- Synthesis base: commit
  `9802f9262ea7f753365fc4bdd20b1fe08379291d`, tree
  `1452612428175196d218c2e55af24812dd82a0d8`.
- All branch changes are confined to `benchmarks/paper_synthesis_v1/`.
- Historical scientific source, result records, checkpoints, and ledgers were
  not modified or reclassified in place.
- `source_registry.json` records byte hashes for production semantics and key
  immutable headline-result files. The base Git tree identifies the complete
  historical snapshot, not only that subset.

## Read-only reconciliation

The analysis imports no LongiControl environment and makes no environment
`reset()` or `step()` call. It parses nine explicit historical ledgers, guards
Paper numeric identities before content access, independently recomputes route
and full-task identities, and accepts positive witnesses only from allow-listed
task-level records satisfying the original requirements. Aggregate RSR and
failed episodes cannot become witnesses or proofs.

Generated artifacts reproduce 411 unique routes, 411 unique full tasks, 424
study-task occurrences, 187 witness tasks, nine proven-infeasible tasks, and
215 unresolved tasks. The method registry contains 34 configurations in eight
families; the paired registry contains nine comparisons. Markdown renderings
are checked against their JSON/CSV generators.

## Paper seal

Only existing preparation/status metadata were read. Independent status
records report `SEALED_UNINSPECTED`, zero Paper episodes, and zero constructed
content fingerprints where those counters exist. No reserved Paper task was
generated, reconstructed, serialized, fingerprinted, reset, stepped, or
evaluated. The synthesis contains no executable Paper runner and grants no
authorization.

## Checkpoint byte audit

All three Hybrid checkpoint files are locally available and match SHA-256
values already embedded in Hybrid Validation V2 provenance. Constrained V2
seed-11/29/47 model files are also locally available and their bytes now have
recorded candidate hashes, but the original Constrained V2 result record did
not embed those hashes. This weaker provenance is disclosed and left for human
decision; no checkpoint was deserialized or executed.

## Scientific activity counters

| Activity performed by this synthesis | Count |
| --- | ---: |
| New controller-performance episodes | 0 |
| New training episodes | 0 |
| New environment resets | 0 |
| New environment steps | 0 |
| New gradient/supervised updates | 0 |
| Paper content fingerprints | 0 |

## Verification status

Final read-only verification passed:

- 36 focused tests passed, zero failed;
- `python -m benchmarks.paper_synthesis_v1.verify` passed generated-artifact,
  source-hash, Git ancestry/path-isolation, seal-metadata, registry-denominator,
  prohibited-entrypoint, and checkpoint-byte checks;
- all six locally present Hybrid/Constrained checkpoint candidates matched the
  manifest byte hashes;
- Ruff check and format check passed for the package; and
- `git diff --check` passed.

The final commit and clean remote synchronization are checked after this report
is committed. Any subsequent mismatch requires an integrity stop rather than a
silent repair.
