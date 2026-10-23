# Split Ledger

`split_ledger.json` is authoritative. Both panels are selected in ascending
numeric-seed order using Policy R against every accepted row discoverable in
historical ledger files and all reachable canonical training tasks.

- Diagnostic namespace: `CHUNK_RL_DIAGNOSTIC_V1`, scan from 11,200,000,
  36 accepted tasks, ordered SHA-256
  `8a4259923ec425c1a31b2f6c9e86b774feee7631cb02dbad58cf42ddb0d4f39e`.
- Validation namespace: `CHUNK_RL_VALIDATION_V1`, scan from 11,300,000,
  36 accepted tasks, ordered SHA-256
  `5db63770520f3ca06b109d2ef7397cef30d02f4c22033a8943dd845f7ea9cc47`.

Selection uses only numeric identity and route/full-task fingerprints. It does
not run a controller or inspect performance. Validation remains
`FROZEN_UNOPENED`; Paper content is not constructed and only numeric seeds
4000–4017 are excluded.

The ledger records 3,825 unique protected historical/training rows from 13
immutable ledger sources. All numeric, route, and full-task disjointness checks
passed.

