# Split Ledger

Both splits were selected in ascending numeric-seed order using only numeric
seeds and route/full-task hashes. No controller outcome, runtime or solver
difficulty entered selection.

## PHYSICS_REFERENCE_DEV_V1

- candidate start: `9800000`
- accepted: 18 of 19 scanned
- ordered hash:
  `4551ddfa2f4363aff35323a2ed5994631f5a26bb50948af66f9d78b7558d3d1f`
- seeds: `9800000–9800004`, `9800006–9800018`
- rejected: `9800005` (duplicate protected route and full task)

## PHYSICS_REFERENCE_HOLDOUT_V1

- candidate start: `9900000`
- accepted: 36 of 49 scanned
- ordered hash:
  `c02c4889333eaf3526277b10bc5c33101ac1aa97f9de836379088c661e72a30b`
- accepted seeds:
  `9900000, 9900003–9900011, 9900013–9900018, 9900020, 9900022,
  9900023, 9900025–9900027, 9900029, 9900030, 9900033–9900039,
  9900042, 9900044, 9900046–9900048`
- rejected duplicate-content seeds:
  `9900001, 9900002, 9900012, 9900019, 9900021, 9900024, 9900028,
  9900031, 9900032, 9900040, 9900041, 9900043, 9900045`

## Integrity proof

The machine ledger protects 2,766 prior rows and verifies:

- correct 18/36 counts;
- unique numeric seeds, route hashes and full-task hashes within each split;
- Development/Holdout disjointness in all three identities;
- no route/full-task overlap with prior canonical training streams or legitimately
  opened evaluation content;
- no Paper numeric seed;
- no Paper content fingerprint construction.

All checks are true. The holdout was frozen before performance evaluation and
the runner now fails closed because `holdout_results.json` exists.
